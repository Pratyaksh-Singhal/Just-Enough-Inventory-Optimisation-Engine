/* Run the built dashboard's own functions and print the results as JSON.
 *
 * Driven by tests/test_dashboard_behaviour.py. Node is not a dependency of this project,
 * so that test skips when node is absent -- the same arrangement build_dashboard.py uses
 * for its syntax check, and the same one tests/test_service_postgres.py uses for a
 * database. What this buys is the only layer that executes the page's JavaScript: every
 * other check reads it as text, and a page can be perfectly well-formed and still compute
 * the wrong number.
 *
 * Usage: node tests/js/page_checks.js <path-to-built-index.html>
 */
'use strict';

const fs = require('fs');
const NL = String.fromCharCode(10);
const html = fs.readFileSync(process.argv[2], 'utf8');

/** The first inline <script> block -- the one holding DATA and the calculator. */
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
const src = blocks[0];

/** Lift a top-level function declaration out of the page by brace matching. */
function grab(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('function not found in the built page: ' + name);
  let depth = 0;
  for (let k = src.indexOf('{', i); k < src.length; k++) {
    if (src[k] === '{') depth++;
    else if (src[k] === '}' && --depth === 0) return src.slice(i, k + 1);
  }
  throw new Error('unbalanced braces reading ' + name);
}

const parseCsv = eval('(' + grab('parseCsv') + ')');
const dateKey = eval('(' + grab('dateKey') + ')');
const windowTotals = eval('(' + grab('windowTotals') + ')');
const quantile = eval('(' + grab('quantile') + ')');

const out = {};

// ---------------------------------------------------------------- parsing
const csv = (rows) => rows.join(NL);

out.blocks = blocks.length;

out.unreadable_units_are_skipped = (() => {
  const r = parseCsv(csv(['sku,date,units_sold', 'A,2026-01-01,N/A', 'A,2026-01-02,5']));
  return { rows: r.rows.length, skipped: r.skipped, first_units: r.rows[0] && r.rows[0].units };
})();

out.ragged_row_is_skipped = (() => {
  const r = parseCsv(csv(['sku,date,units_sold', 'A,2026-01-01', 'A,2026-01-02,5']));
  return { rows: r.rows.length, skipped: r.skipped };
})();

out.blank_cell_is_skipped = (() => {
  const r = parseCsv(csv(['sku,date,units_sold', 'A,2026-01-01,', 'A,2026-01-02,5']));
  return { rows: r.rows.length, skipped: r.skipped };
})();

out.zero_is_a_real_sale_of_none = (() => {
  const r = parseCsv(csv(['sku,date,units_sold', 'A,2026-01-01,0', 'A,2026-01-02,5']));
  return { rows: r.rows.length, skipped: r.skipped, first_units: r.rows[0].units };
})();

// A header containing a rival field's keyword must not be claimed by that field.
out.columns = {};
for (const [label, head, row] of [
  ['sales_date_is_a_date', 'product,sales_date,units_sold', 'A,2026-01-01,5'],
  ['store_name_is_a_store', 'date,store_name,item_name,units', '2026-01-01,S1,Milk,5'],
  ['unit_price_beats_units', 'sku,date,units,unit_price', 'A,2026-01-01,5,2.50'],
  ['aliases', 'item,day,qty,price,store', 'A,2026-01-01,5,2.50,S1'],
]) {
  const r = parseCsv(csv([head, row, row.replace('A,', 'B,')]));
  out.columns[label] = r.error ? { error: r.error } : r.rows[0];
}

out.missing_units_column_is_named = (() => {
  const r = parseCsv(csv(['sku,date', 'A,2026-01-01']));
  return { error: r.error || null };
})();

out.missing_sku_column_is_named = (() => {
  const r = parseCsv(csv(['when,howmany', '2026-01-01,5']));
  return { error: r.error || null };
})();

out.header_only_is_refused = (() => {
  const r = parseCsv('sku,date,units_sold');
  return { error: r.error || null };
})();

// ---------------------------------------------------------------- dates
out.dates = {};
for (const v of ['2026-01-05', '5 Jan 2026', '13/01/2026', '05/01/2026', '2026/1/5']) {
  out.dates[v] = dateKey(v);
}
out.unparseable_dates = {};
for (const v of ['week 3', 'banana', '', 'period 12']) {
  out.unparseable_dates[v] = dateKey(v);
}

out.sorted_ddmmyyyy = ['1/5/2026', '10/1/2026', '2/2/2026'].sort((a, b) => {
  const x = dateKey(a), y = dateKey(b);
  if (x == null || y == null) return 0;
  return x - y;
});

// ---------------------------------------------------------------- degenerate maths
out.window_totals_shorter_than_horizon = windowTotals([1, 2, 3], 7).length;
out.window_totals_exactly_horizon = windowTotals([1, 2, 3, 4, 5, 6, 7], 7).length;
out.quantile_of_empty = quantile([], 0.5);
out.quantile_of_one = quantile([4], 0.41);
out.quantile_of_identical = quantile([3, 3, 3, 3], 0.41);

process.stdout.write(JSON.stringify(out, null, 2));
