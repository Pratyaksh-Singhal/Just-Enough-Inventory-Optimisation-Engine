# E10 — Ship ⬜

**Goal:** deployed, and a README that survives a 90-second recruiter read.

**Consumes:** everything.

---

### E10-S1 — Deploy ⬜

- [ ] Fly.io or Railway; API + dashboard reachable
- [ ] DuckDB file shipped with the image or mounted as a volume — decide and document
      which, because it determines whether the nightly job can write in production
- [ ] Monthly hosting cost recorded

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
