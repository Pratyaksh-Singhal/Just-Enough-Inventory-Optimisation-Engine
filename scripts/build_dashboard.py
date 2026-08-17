"""Regenerate ``dashboard/index.html`` from the warehouse.

Queries every table the page shows, computes the exact cost sweep, and inlines the result
into the template as JSON. Run after any pipeline change that moves the numbers.
"""

from __future__ import annotations

import base64
import json
import pathlib
import sys

import duckdb
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_engine.config import WAREHOUSE_PATH  # noqa: E402
from inventory_engine.models.quantiles import monotonize  # noqa: E402


def _sweep(con: duckdb.DuckDBPyConnection) -> dict:
    """Precompute shortfall/leftover unit totals per service level.

    Order quantity depends only on the critical ratio, so these two totals are all the page
    needs: ``cost(Cu, Co) = shortfall * Cu + leftover * Co``, exactly. That is what lets the
    slider be exact arithmetic rather than an interpolated approximation.
    """
    raw = con.execute("""
        SELECT fold, item_id, store_id, target_date, quantile, yhat FROM forecast
        WHERE model_name = 'lgbm' AND level = 'item_store' AND reconciled = FALSE
          AND quantile IS NOT NULL
    """).df()
    wide = (
        monotonize(raw)
        .pivot_table(
            index=["fold", "item_id", "store_id", "target_date"], columns="quantile", values="yhat"
        )
        .reset_index()
    )
    actuals = con.execute("""
        SELECT item_id, store_id, date AS target_date, CAST(units AS DOUBLE) AS demand
        FROM fact_sales
    """).df()
    panel = wide.merge(actuals, on=["item_id", "store_id", "target_date"], how="inner")

    levels = np.array(sorted(c for c in wide.columns if isinstance(c, float)))
    grid = panel[levels].to_numpy(float)
    demand = panel["demand"].to_numpy(float)

    def qty_at(cr: float) -> np.ndarray:
        i = np.clip(np.searchsorted(levels, cr) - 1, 0, len(levels) - 2)
        w = (cr - levels[i]) / (levels[i + 1] - levels[i])
        return np.clip(grid[:, i] + w * (grid[:, i + 1] - grid[:, i]), 0, None)

    def totals(order: np.ndarray) -> dict:
        return {
            "short": float(np.maximum(demand - order, 0).sum()),
            "left": float(np.maximum(order - demand, 0).sum()),
            "stockout_rate": float((demand > order).mean()),
        }

    lagged = con.execute("""
        SELECT item_id, store_id, date + INTERVAL 7 DAY AS target_date,
               CAST(units AS DOUBLE) AS naive_qty
        FROM fact_sales
    """).df()
    naive_qty = (
        panel.merge(lagged, on=["item_id", "store_id", "target_date"], how="left")["naive_qty"]
        .fillna(0)
        .to_numpy()
    )
    return {
        "sweep": [
            {"cr": float(cr), **totals(qty_at(cr))}
            for cr in np.round(np.arange(0.10, 0.991, 0.01), 3)
        ],
        "fixed95": totals(qty_at(0.95)),
        "naive": totals(naive_qty),
        "n_decisions": int(len(panel)),
    }


def _demo_csv(con: duckdb.DuckDBPyConnection, days: int = 120) -> str:
    """Real M5 sales history as a ready-to-load CSV for the order calculator.

    Chosen by volume, not at random. A first-time visitor pressing "Load example data" needs
    numbers they can read: sampling evenly across the intermittency bands filled the table
    with products selling under one unit a day, where the mathematically correct answer is
    "order zero" and the tool looks broken rather than right.

    So this leads with six legible movers and appends **one** genuine slow-seller, which is
    what makes the "order on request" row in the results a demonstrated case rather than an
    unexercised branch.
    """
    fast = con.execute(
        """
        SELECT item_id, store_id FROM fact_sales
        WHERE date >= DATE '2016-01-24'
        GROUP BY 1, 2 HAVING avg(units) BETWEEN 4 AND 19
        ORDER BY avg(units) DESC LIMIT 6
        """
    ).fetchall()
    slow = con.execute(
        """
        SELECT f.item_id, f.store_id FROM fact_sales f
        JOIN dim_item_stratum s ON s.item_id = f.item_id
        WHERE f.date >= DATE '2016-01-24' AND s.stratum_name = 'sparse'
        GROUP BY 1, 2 HAVING avg(f.units) BETWEEN 0.3 AND 1.2
        ORDER BY avg(f.units) DESC LIMIT 1
        """
    ).fetchall()

    lines = ["sku,date,units_sold,unit_price"]
    for item, store in fast + slow:
        rows = con.execute(
            """
            SELECT CAST(date AS VARCHAR), units, price FROM fact_sales
            WHERE item_id = ? AND store_id = ? AND price IS NOT NULL
            ORDER BY date DESC LIMIT ?
            """,
            [item, store, days],
        ).fetchall()
        for d, units, price in reversed(rows):
            lines.append(f"{item}-{store},{d},{int(units)},{price:.2f}")
    return "\n".join(lines)


def build() -> dict:
    """Query the warehouse and assemble every figure the dashboard renders.

    Deliberately narrow: this returns the six keys the page actually reads and nothing
    else. It used to return fifteen -- fold metrics, per-stratum MASE, MinT level tables,
    WRMSSE, coherence gaps, SHAP importances, the stratum profile and three example
    series -- all of which fed the technical-details tab. That tab is gone, the material
    lives in the README, and every one of those keys was still being queried, serialised
    and shipped to every visitor to be read by nothing.

    If a section comes back, so does its query. Inlining JSON nobody parses is the kind of
    cost that never shows up as a failure, only as a slower page.
    """
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    try:
        data: dict = {"money": con.execute("SELECT * FROM cost_comparison").df().to_dict("records")}
        # sweep / fixed95 / naive / n_decisions -- the cost simulator and the money table
        data.update(_sweep(con))
        # the order calculator's "Load example data" button
        data["demoCsv"] = _demo_csv(con)
        return data
    finally:
        con.close()


#: Faces vendored into ``dashboard/fonts/`` and embedded at build time, in the order the
#: ``@font-face`` rules are emitted. Latin subsets only -- the page is English.
FONT_FACES: tuple[tuple[str, str, str, str], ...] = (
    ("Barlow", "400", "normal", "barlow-400.woff2"),
    ("Barlow", "600", "normal", "barlow-600.woff2"),
    ("Barlow Condensed", "600", "normal", "barlow-condensed-600.woff2"),
    ("JetBrains Mono", "400", "normal", "jetbrains-mono-400.woff2"),
)


def _font_css() -> str:
    """Build the ``@font-face`` block, with each woff2 inlined as a data URI.

    Embedded rather than linked because the page is served from two places with different
    rules. On Fly a Google Fonts ``<link>`` would work; published as a Claude Artifact the
    CSP blocks font CDNs outright and the page would silently fall back to system faces --
    losing half the design's identity with no error to notice. A self-contained page has
    the same typography wherever it lands.

    The cost is honest and worth stating: ~86 KB of woff2 becomes ~115 KB of base64. The
    files are vendored in the repository rather than fetched during the build, so a build
    does not depend on Google being reachable, and upgrading a face is a visible commit.

    Raises:
        FileNotFoundError: If a face is missing. A silent fallback to system fonts is the
            exact failure this function exists to prevent, so it fails loudly instead.

    """
    rules = []
    for family, weight, style, filename in FONT_FACES:
        path = ROOT / "dashboard" / "fonts" / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"vendored font missing: {path}. The page embeds its faces rather than "
                "linking a CDN; without this file the design silently loses its typography."
            )
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        rules.append(
            f"@font-face{{font-family:'{family}';font-style:{style};font-weight:{weight};"
            f"font-display:swap;src:url(data:font/woff2;base64,{b64}) format('woff2')}}"
        )
    return "\n".join(rules)


def main() -> int:
    """Regenerate ``dashboard/index.html`` from the template and the warehouse."""
    template = (ROOT / "dashboard" / "index.template.html").read_text(encoding="utf-8")
    payload = json.dumps(build(), separators=(",", ":"))
    out = ROOT / "dashboard" / "index.html"
    rendered = template.replace("__DATA__", payload).replace("__FONTS__", _font_css())
    out.write_text(rendered, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
