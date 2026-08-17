# E10 — Ship 🚧

**Goal:** deployed, and a README that survives a 90-second recruiter read.

**Consumes:** everything.

---

### E10-S1 — Deploy ✅

- [x] Fly.io; API + dashboard reachable at https://inventory-optimization-engine.fly.dev/
      — the service serves the built page at `/` so the Full forecast tab shares an origin
      with its own API. A page hosted anywhere else cannot reach it: the Artifact runtime
      blocks every external host and offers no capability to opt out.
- [x] DuckDB file: **neither**. Tier 1 is not deployed. Every figure it produces is inlined
      into the dashboard as JSON at build time by `scripts/build_dashboard.py`, so the live
      endpoints would serve a 324 MB warehouse to nobody. Nothing writes it in production,
      so the nightly-job question does not arise.
- [x] Monthly hosting cost recorded: **≈ $0.32**. One Machine that suspends when idle (a
      volume attaches to exactly one Machine, so the API and the arq worker share one),
      a 1 GB volume at $0.15 which bills whether or not the Machine runs, Postgres on
      Neon's free tier and Redis on Upstash's. An always-on Machine with Fly Managed
      Postgres would have been about $50.

### E10-S2 — README, ordered for 90 seconds ⬜

Order is load-bearing. A recruiter reads top-down and stops early.

1. [ ] One-line business problem + headline result
2. [ ] Live demo link + 30-second GIF **of the cost simulator**, not a chart
3. [ ] The money table
4. [ ] Approach: baselines → model → reconciliation → optimization
5. [ ] Backtest methodology + full metrics with fold variance
6. [ ] **"What broke and how I fixed it"**
7. [ ] Cost/latency: p95 inference, monthly hosting
8. [ ] Limitations + next steps

### E10-S3 — "What broke and how I fixed it" ⬜

**This section matters most.** It is the one part a strong reviewer reads carefully,
because it is the only part that cannot be faked.

- [ ] The intermittent demand problem and what actually worked on the sparse stratum
- [ ] Any leakage bug caught by `test_no_leakage.py`
- [ ] The fold where the model underperformed — named, with the number
- [ ] **What did not work**, kept in rather than edited out
- [ ] Already accumulating in the README from E1: the brief's scope contradiction, the two
      loader bugs the tests caught, and the stratification test that was asserting on
      sampling noise

### E10-S4 — Demo GIF ⬜

- [ ] 30 seconds, cost simulator, sliders moving and cost responding
- [ ] Above the fold in the README
- [ ] Not a chart. The interaction is the pitch.

### E10-S5 — Limitations ⬜

- [ ] Panel is intermittency-stratified, not volume-representative (from E1-S4)
- [ ] Scope is one state, one category
- [ ] Cost assumptions are invented, however carefully sensitivity-tested
- [ ] Currency is USD; the ₹ toggle is presentation only
- [ ] Next steps, concrete enough to be actionable
