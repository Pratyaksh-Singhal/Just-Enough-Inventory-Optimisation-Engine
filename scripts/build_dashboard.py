"""Regenerate ``dashboard/index.html`` from the warehouse."""

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
    """Precompute shortfall/leftover unit totals per service level."""
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
    """Real M5 sales history as a ready-to-load CSV for the order calculator."""
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
    """Query the warehouse and assemble every figure the dashboard renders."""
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
    """Build the ``@font-face`` block, with each woff2 inlined as a data URI."""
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


def _check_scripts(html: str) -> None:
    """Parse every inline ``<script>`` and fail the build on a syntax error."""
    import re
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    if node is None:
        print(f"  ! node not found: {len(blocks)} script block(s) not syntax-checked")
        return

    with tempfile.TemporaryDirectory() as tmp:
        for i, block in enumerate(blocks):
            path = pathlib.Path(tmp) / f"block{i}.js"
            path.write_text(block, encoding="utf-8")
            done = subprocess.run(
                [node, "--check", str(path)], capture_output=True, text=True, check=False
            )
            if done.returncode != 0:
                raise SystemExit(
                    f"script block {i} does not parse -- refusing to write a "
                    f"dead page: {done.stderr.strip()}"
                )
    print(f"  {len(blocks)} script block(s) parse")


def main() -> int:
    """Regenerate ``dashboard/index.html`` from the template and the warehouse."""
    template = (ROOT / "dashboard" / "index.template.html").read_text(encoding="utf-8")
    payload = json.dumps(build(), separators=(",", ":"))
    out = ROOT / "dashboard" / "index.html"
    rendered = template.replace("__DATA__", payload).replace("__FONTS__", _font_css())
    _check_scripts(rendered)
    out.write_text(rendered, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
