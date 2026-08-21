"""Structural rules the built dashboard has to satisfy, checked without a browser.

These exist because of what actually shipped. A stray newline in a regex literal killed a
whole script block; an SVG stroke written as a presentation attribute rendered the upload
icon invisible; a bare ``svg { width: 100% }`` inflated every icon to the width of its
button; and a ``@media (max-width: 700px)`` block placed above the rules it overrode was
inert for days while its numbers were repeatedly adjusted. Every one of those passed a
build, a test run and a deploy, because nothing checked the page as a page.

None of this needs node or a browser: it is the shape of the file, and the shape is what
kept going wrong.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = ROOT / "dashboard" / "index.html"
TEMPLATE = ROOT / "dashboard" / "index.template.html"


@pytest.fixture(scope="module")
def page() -> str:
    if not PAGE.is_file():
        pytest.skip(f"no built page at {PAGE}; run scripts/build_dashboard.py")
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css(page: str) -> str:
    return page.split("</style>")[0]


@pytest.fixture(scope="module")
def body(page: str) -> str:
    return page.split("</style>", 1)[1]


# --------------------------------------------------------------------- the payload


def test_the_data_placeholder_was_substituted(page):
    assert "__DATA__" not in page
    assert "__FONTS__" not in page


def test_the_inlined_data_is_valid_json(page):
    blob = re.search(r"const DATA = (\{.*?\});", page, re.S)
    assert blob, "no DATA assignment in the page"
    data = json.loads(blob.group(1))
    assert {"money", "sweep", "fixed95", "naive", "n_decisions", "demoCsv"} <= set(data)


def test_the_fonts_are_embedded_rather_than_linked(page):
    """A CDN link works on Fly and is blocked when the page is published elsewhere, where
    it would fall back to system faces with no error to notice."""
    assert page.count("@font-face") == 4
    assert "fonts.googleapis.com" not in page


# --------------------------------------------------------------------- ids


def test_every_element_id_the_code_queries_exists_in_the_markup(page):
    queried = set(re.findall(r"""[#$]{1,2}\(['"]#([A-Za-z0-9_-]+)['"]\)""", page))
    queried |= set(re.findall(r"""getElementById\(['"]([A-Za-z0-9_-]+)['"]\)""", page))
    present = set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', page))
    assert not queried - present, f"queried but never rendered: {sorted(queried - present)}"


def test_every_tab_target_resolves_to_a_section(page):
    """``data-goto`` and the tab strip both call ``show()`` with a section name."""
    sections = set(re.findall(r'<section id="([a-z]+)"', page))
    tabs = set(re.findall(r'data-t="([a-z]+)"', page))
    gotos = set(re.findall(r'data-goto="([a-z]+)"', page))
    assert tabs <= sections, f"tab with no section: {sorted(tabs - sections)}"
    assert gotos <= sections, f"link to no section: {sorted(gotos - sections)}"


def test_the_five_tabs_are_all_present(page):
    assert re.findall(r'data-t="([a-z]+)"', page) == [
        "home",
        "calculator",
        "full",
        "why",
        "overview",
    ]


# --------------------------------------------------------------------- the cascade


def test_no_bare_svg_rule_can_inflate_the_icons(css):
    """A CSS width overrides an HTML width attribute.

    ``svg { width: 100% }`` was written for the charts and silently stretched every inline
    icon to the width of its container -- the arrow in a button rendered as tall as the
    label. Charts opt in with ``.fluid`` instead.
    """
    assert not re.search(r"(^|[,{}])\s*svg\s*\{", css, re.M), "a bare `svg {` rule is back"


def test_every_chart_opts_into_scaling_explicitly(page):
    """The other half of that rule: a chart without ``.fluid`` renders at its attribute
    size, which is the opposite failure and just as silent."""
    for tag in re.findall(r"<svg[^>]*viewBox[^>]*>", page):
        if 'class="arrow"' in tag or "width=" in tag:
            continue
        assert 'class="fluid"' in tag, f"chart svg without .fluid: {tag[:80]}"


def test_no_svg_takes_its_stroke_from_a_presentation_attribute(page):
    """``stroke="var(--jade)"`` leaves stroke at its initial ``none`` where the custom
    property does not compute, and the glyph renders invisible with nothing logged."""
    assert 'stroke="var(' not in page
    assert 'fill="var(' not in page


def test_the_wrap_gutter_is_never_set_with_the_padding_shorthand(css):
    """``.wrap { padding: 0 32px }`` sets all four sides.

    ``main`` and ``footer`` both carry ``.wrap``, and a class outranks an element selector,
    so their vertical padding was silently zeroed -- the first heading sat against the tab
    strip and the footer ran into the last card. Overrides use longhands for that reason;
    a shorthand anywhere on a ``.wrap`` selector reintroduces it.
    """
    shorthands = [
        m.group(0)[:60]
        for m in re.finditer(r"\.wrap[^{]*\{[^}]*?\bpadding\s*:", css)
        if "padding-" not in m.group(0)
    ]
    assert len(shorthands) == 1, f"unexpected padding shorthand on a .wrap rule: {shorthands}"


def test_narrow_screen_overrides_come_after_the_rules_they_override(css):
    """Equal specificity means source order decides, and a media query adds none.

    The narrow-screen block sat above the base rules for ``.drop``, ``.filestats``,
    ``.results`` and the rest, so every value in it was inert -- on a phone the page kept
    its desktop sizes while the numbers in that block were adjusted again and again.
    """
    last_block = css.rindex("@media (max-width: 700px)")
    after = css[css.index("}", css.rindex("}", last_block)) :]
    trailing = re.findall(r"\n  ([^@\s{][^{\n]*)\{", after)
    assert not trailing, f"base rules declared after the narrow-screen block: {trailing}"


# --------------------------------------------------------------------- dead weight


def test_no_function_is_declared_twice_in_one_script_block(page):
    """The later declaration wins, so the earlier one is unreachable while still reading
    as live code -- an edit made there changes nothing and says nothing."""
    for i, block in enumerate(re.findall(r"<script>([\s\S]*?)</script>", page)):
        names = re.findall(r"^  function ([A-Za-z0-9_]+)\(", block, re.M)
        dupes = {n for n in names if names.count(n) > 1}
        assert not dupes, f"script block {i} declares {sorted(dupes)} more than once"


def test_only_one_handler_reads_the_calculator_file_input(page):
    """Two readers on one ``change`` event resolved on different async paths, so whether
    results appeared after picking a file was a race."""
    handlers = len(re.findall(r"""\$\('#csvFile'\)\.addEventListener\(\s*'change'""", page))
    handlers += len(re.findall(r"""\bfile\.addEventListener\(\s*'change'""", page))
    assert handlers == 1, f"{handlers} change handlers bound to #csvFile"


def test_every_class_the_page_emits_has_a_rule(page, css, body):
    """Catches the reverse of dead CSS: markup referring to styling that was pruned."""
    emitted = set()
    for attr in re.findall(r'class="([^"]*)"', body):
        # Class lists assembled in JavaScript -- `class="${cls}"`, or a concatenation like
        # `class="' + wasteClass + '"` -- carry variable names, not class names. Only
        # literal attributes can be checked this way.
        if any(ch in attr for ch in "${}+'`"):
            continue
        emitted.update(t for t in attr.split() if re.fullmatch(r"[a-z][\w-]*", t))
    styled = set(re.findall(r"\.([A-Za-z][\w-]*)", css))
    assert not emitted - styled, f"emitted with no rule: {sorted(emitted - styled)}"


def test_the_built_page_matches_its_template(page):
    """The committed page is what gets served; a template edit that was never rebuilt
    ships the old page and looks like the change simply had no effect."""
    template = TEMPLATE.read_text(encoding="utf-8")
    skeleton = re.sub(r"const DATA = \{.*?\};", "const DATA = __DATA__;", page, flags=re.S)
    skeleton = re.sub(r"@font-face\{.*?\}\n?", "", skeleton, flags=re.S)
    template_skeleton = template.replace("__FONTS__\n", "")
    assert skeleton.strip() == template_skeleton.strip(), (
        "dashboard/index.html is out of date with its template -- run "
        "scripts/build_dashboard.py and commit the result"
    )


def test_the_cost_curve_chooses_its_shape_at_draw_time(page):
    """SVG text scales with the viewBox, so a wide box on a phone shrinks the words too.

    An 880x300 box rendered about 97px tall with 3px labels in a phone column. The fix is
    a different viewBox below the breakpoint, which only works if the renderer asks at
    draw time rather than baking one shape in.
    """
    assert "function narrowScreen()" in page
    curve = page[page.index("function renderCurve(") :][:1600]
    assert "narrowScreen()" in curve, "renderCurve no longer adapts to the viewport"
    assert 'font-size="${fs}"' in curve, "label size is no longer tied to the chosen shape"


@pytest.mark.parametrize(
    "tab,marker",
    [
        ("home", ".home-block"),
        ("how much to order", ".drop"),
        ("why these numbers", "h1.why-h1"),
        ("proof it works", ".simchart"),
        ("full forecast", ".updrop"),
    ],
)
def test_every_tab_has_some_narrow_screen_rule(css, tab, marker):
    """Fig 03 shipped with none at all and kept desktop sizes in a phone column.

    Narrow-screen rules live in several blocks, not one -- each tab's sit just after that
    tab's own base rules, which is what keeps them winning. So this looks across every
    ``max-width`` block rather than only the last, and just asserts no tab was forgotten.
    """
    narrow = "".join(
        css[m.start() : css.index("\n  }", m.start())]
        for m in re.finditer(r"@media\s*\(max-width", css)
    )
    assert marker in narrow, f"the {tab} tab has no narrow-screen rule anywhere"
