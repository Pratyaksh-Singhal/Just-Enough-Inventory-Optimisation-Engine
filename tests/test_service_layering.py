"""The API enqueues; the worker trains. Asserted, not assumed.

Tier 1 established this discipline in ``test_no_handler_imports_a_trainer``, which reads
``api/app.py`` and greps for four banned import lines. That worked while there was exactly
one handler module, but it has two limits worth fixing rather than copying: a substring
search misses ``from lightgbm import LGBMRegressor``, and a new handler in a different file
is not checked at all.

So this walks the AST of every request-handling module in both tiers. A new router added
tomorrow is covered the moment it exists, and an import is recognised by what it imports
rather than by how the line was spelled.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import inventory_engine

SRC = Path(inventory_engine.__file__).parent

#: Libraries that fit models. A request handler has no legitimate reason to import one:
#: fitting inside a request handler is the failure mode this whole layering exists to
#: prevent, and it would be found in production as a timeout rather than in review.
TRAINING_LIBRARIES = frozenset(
    {"lightgbm", "statsforecast", "hierarchicalforecast", "mlflow", "sklearn", "shap"}
)

#: Modules that serve requests. Everything reachable from an HTTP handler.
HANDLER_MODULES = (
    SRC / "api" / "app.py",
    SRC / "api" / "deps.py",
    SRC / "service" / "app.py",
    SRC / "service" / "db" / "session.py",
    SRC / "service" / "storage.py",
    SRC / "service" / "gate.py",
    *sorted((SRC / "service" / "routers").glob("*.py")),
)


def imported_roots(path: Path) -> set[str]:
    """Top-level package names imported by ``path``, from the AST rather than the text.

    Catches ``import x``, ``import x.y``, ``from x import y`` and ``from x.y import z``
    alike -- including ones inside a function, which is where a lazy import of a trainer
    would hide.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", HANDLER_MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_handler_imports_a_trainer(path: Path):
    """No request-handling module may import a model-fitting library."""
    offending = sorted(imported_roots(path) & TRAINING_LIBRARIES)
    assert not offending, (
        f"{path.relative_to(SRC)} imports {offending}. Request handlers enqueue; the worker "
        "trains. Move the fitting code into service/worker.py and enqueue a job instead."
    )


@pytest.mark.parametrize("path", HANDLER_MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_handler_imports_a_pipeline_module(path: Path):
    """Handlers must not reach into the fitting pipeline, even by a first-party name.

    Named module by module rather than banning ``inventory_engine.models`` wholesale,
    because that package is not uniformly worker-side. ``models.quantiles`` holds
    :func:`~inventory_engine.models.quantiles.monotonize`, a read-time rearrangement of
    stored quantiles with no fitting in it, and tier 1's ``/forecast`` handler is
    *supposed* to call it -- every consumer of stored quantiles reads them through it or
    occasionally renders q0.9 above q0.95.

    That a pure function lives in a package called ``models`` is the naming problem noted
    in the tier 2 review; it is a tier 1 refactor, not something to paper over by loosening
    this rule until it stops firing.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    banned = {
        "inventory_engine.models.gbm",
        "inventory_engine.models.hybrid",
        "inventory_engine.models.baselines",
        "inventory_engine.hierarchy.mint",
        "inventory_engine.service.pipeline",
        "inventory_engine.service.worker",
    }
    hits = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        hits.update(n for n in names if any(n == b or n.startswith(b + ".") for b in banned))
    assert not hits, f"{path.relative_to(SRC)} imports worker-side module(s) {sorted(hits)}"


def test_the_router_package_is_fully_covered():
    """Every module in ``service/routers`` is in the scanned list.

    Without this, adding ``routers/admin.py`` would silently escape both checks above --
    which is precisely the gap in tier 1's single-file version of this test.
    """
    on_disk = {p for p in (SRC / "service" / "routers").glob("*.py")}
    assert on_disk <= set(HANDLER_MODULES), (
        f"unscanned router module(s): {sorted(p.name for p in on_disk - set(HANDLER_MODULES))}"
    )


def test_the_tier_one_guard_still_holds():
    """Tier 1's own rule, restated here so both tiers fail from one test file."""
    assert not (imported_roots(SRC / "api" / "app.py") & TRAINING_LIBRARIES)


# --------------------------------------------------------------------------- cold start

#: Heavy at import and unnecessary to serve a page, answer a health check or poll a job.
#: pandas alone measured 2.4s of a 4.5s cold start, paid before uvicorn could answer
#: anything -- on a Machine that suspends when idle, that is the first visitor's wait.
DEFERRED_LIBRARIES = frozenset({"pandas", "numpy"})


def test_importing_the_api_does_not_import_pandas():
    """Importing the app must not drag the data stack in with it.

    The three routes that need pandas import it inside the handler, where the request is
    already doing slow work. Everything else -- the dashboard, ``/health``, polling a job --
    starts without it. This is asserted in a subprocess because the test session has almost
    certainly imported pandas already for its own fixtures.
    """
    code = (
        "import sys; import inventory_engine.service.app; "
        "print(','.join(sorted(m for m in "
        f"{sorted(DEFERRED_LIBRARIES)!r} if m in sys.modules)))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    leaked = done.stdout.strip()
    assert not leaked, (
        f"importing the API pulled in {leaked}. Something on the import path reaches the "
        "data stack again -- check for a module-scope import in a router, or a package "
        "__init__ that re-exports one."
    )


def test_the_upload_handler_can_still_reach_pandas():
    """The other half: deferring must not have broken the code that needs it."""
    code = (
        "import inventory_engine.service.routers.full_forecast as m; "
        "import inspect; src = inspect.getsource(m.upload); "
        "assert 'import pandas' in src, 'upload no longer imports pandas anywhere'; "
        "print('ok')"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
