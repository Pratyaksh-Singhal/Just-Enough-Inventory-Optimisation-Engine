# Branching model

Branches map to **SDLC stages**, not to epics. A branch per epic would mean ten
long-lived branches racing each other over the same DuckDB schema, and the phase contract
in [`BACKLOG.md`](BACKLOG.md) already handles cross-phase coordination — that is what the
named tables are for. Duplicating it in branch structure buys nothing and costs merge
conflicts.

## Long-lived branches

| Branch | Stage | Rule |
|---|---|---|
| `main` | Stable | Only phase-complete work. Every commit here has a green test suite and a phase tag. Never committed to directly. |
| `dev` | Integration | The working branch. Day-to-day commits land here. |

That is the whole permanent structure. A third permanent branch (`staging`) is not worth
carrying until E10 gives it something to deploy — see below.

## Short-lived branches

Cut these off `dev` and delete them after merge. Use them when work is risky enough that
you might want to abandon it, not for every change.

| Pattern | Off | Merges to | For |
|---|---|---|---|
| `feature/<slug>` | `dev` | `dev` | A spike or a change large enough to want an escape hatch — e.g. `feature/mint-reconciliation` |
| `fix/<slug>` | `dev` | `dev` | Non-urgent bug fixes |
| `hotfix/<slug>` | `main` | `main` **and** `dev` | Something broken on a deployed build. Rare until E10. |
| `release/<version>` | `dev` | `main` | Deploy preparation from E10 onward |

## The phase boundary

A phase is done when its epic's produced tables exist and its tests pass. At that point:

```bash
git switch main
git merge --no-ff dev -m "Phase N (EN): <what it delivers>"
git tag -a phase-N -m "Phase N (EN): <one-line summary>"
git switch dev
```

`--no-ff` is deliberate: it keeps each phase as a visible merge commit in `main`'s
history, so `git log --first-parent main` reads as a clean list of ten phases rather than
a flat stream of individual commits.

Phases already tagged:

| Tag | Delivers |
|---|---|
| `phase-1` | DuckDB warehouse, intermittency-stratified 720-series scope |
| `phase-2` | Leakage-free feature panel, leakage gate green |

## Why not a staging branch yet

There is nothing deployed, so a `staging` branch would be a branch nothing ever merges
into. It earns its place at E10, when there is a Fly.io/Railway target: at that point
`release/*` → `staging` → `main` gives a real pre-production check. Adding it now would be
ceremony.

## Commit convention

Reference the story ID so a commit traces back to its acceptance criteria:

```
Phase 3 (E3-S2): Croston and TSB for intermittent demand
```

Phase-boundary merges into `main` use the epic, not a story:

```
Phase 3 (E3): baselines
```
