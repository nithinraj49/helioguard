# Phase 7 gates (pre-registered): QAOA scaling benchmark

Registered: 2026-10-30, Phase 7 session 1, before any benchmark run. Before registration I only timed
20 circuit evaluations of `solve_qaoa` at 6, 12 and 16 qubits to size the grid; no quality metric was
looked at. Edit this file after the runs only to add results under "Outcome"; a changed gate needs a new
dated section that says why.

Scope: a classical-simulator benchmark. **No quantum advantage is claimed unless Gate G6 says so, and G6
is fixed to "not claimed" here.** No GPU. The engine solvers (`verispace_engine/qaoa.py`, `qubo.py`)
are not changed to improve numbers; bugs only, and any fix is listed in the report.

## Problem instances (ILLUSTRATIVE spacecraft and storm values)
HelioGuard fleet-scheduling QUBOs from `verispace_engine.qubo.build_qubo`: one action per asset, at most
one `raise_orbit` per ground window, rewards from `helioguard.tools.qaoa_schedule.action_rewards` on the
Feb 2022 storm and the Starlink-like craft, with each asset's altitude drawn uniformly from 190 to 320 km.
The fleet has 3 actions, so its native sizes are 6, 9, 12, 15 qubits; the sizes you asked for (8, 10, 14,
16) are not multiples of 3. Two families therefore:

| Family | Actions | Qubit sizes | Role |
|---|---|---|---|
| B (primary) | raise_orbit, safe_mode_edge_on (2 per asset) | 6, 8, 10, 12, 14, 16 | the requested size ladder |
| A (native) | raise_orbit, safe_mode_edge_on, hold (3 per asset) | 6, 9, 12, 15 | the real fleet QUBO |

Size confounds: family B has fewer constraint bits per asset than A. Do not compare across families.
Windows: `max(2, ceil(assets/2))` ground windows, each asset's window uniform at random.
Instances: 10 per (family, size), instance seeds fixed in the code (`helioguard/qaoa_instances.py`).

## Arms
- **Engine arm (primary):** `verispace_engine.qaoa.solve_qaoa` unchanged, defaults `restarts=3`,
  `maxiter=120` (COBYLA, uniform(0, pi) start), depth p in 1, 2, 3, 4, optimiser seeds 0, 1, 2.
  All sizes in both families.
- **Harness arm (variants):** a numpy statevector simulator of the same circuit (cost layer = diagonal
  phase, mixer = RX(2 beta) on every qubit, normalised Hamiltonian as in the engine) in
  `verispace_engine/qaoa_bench.py`. Optimisers: COBYLA, Nelder-Mead, L-BFGS-B (finite differences).
  Initialisations: `random` (uniform(0, pi), the engine's) and `ramp` (gamma_k = 0.75 (k+0.5)/p,
  beta_k = 0.75 (1 - (k+0.5)/p), restarts 2 and 3 add N(0, 0.1) jitter). p in 1 to 4, seeds 0, 1, 2.
  Same budget as the engine: 3 restarts, at most 120 objective evaluations per restart, every
  evaluation counted. All sizes in both families.
- **Extension (harness arm only, reduced):** family B at 18 and 20 qubits, p = 2, COBYLA, random start,
  5 instances, 3 seeds. Timing-only probes of the engine at 16, 18, 20 qubits. The size limit is reported
  from these runs, not assumed.

## Metrics (per run, then per size)
1. **Single-sample P(optimum):** probability mass of the final state on minimum-energy bitstrings
   (energy <= minimum + 1e-9, the engine's definition). Primary.
2. **Approximation ratio:** (uniform-mean energy - expected energy) / (uniform-mean - minimum), the engine's.
3. **Best-of-64 (secondary), two readings, both reported:**
   (a) *top-64 readout*: the engine's own pick, the best feasible bitstring among the 64 most probable
   ones. It reads the whole probability vector, which hardware cannot do; it is simulator-only.
   Hit = pick energy equals the exact minimum.
   (b) *64 shots*: 1 - (1 - P(optimum))^64, the exact chance that at least one of 64 samples is optimal.
4. Wall time (s) and number of circuit evaluations. Wall times from the parallel run are contaminated by
   core contention; the table of record for timing is a separate serial pass on one instance per size.
5. Harness arm only: probability of a feasible bitstring.

Aggregation: average the 3 seeds within an instance, then 95 percent percentile bootstrap over the
10 instances (10,000 resamples, seed 0). Paired differences are bootstrapped on per-instance differences.
All hit/success statistics are instance-clustered the same way.

## Classical baselines (reported beside every quantum number)
- Exact enumeration (all sizes up to 20) and OR-Tools CP-SAT: optimal rate against the exact optimum,
  solve time.
- Uniform random sampling: P(optimum) = k / 2^n for k optimal states; best-of-64 hit
  1 - (1 - k/2^n)^64; and the same with a sample budget equal to QAOA's circuit-evaluation count.
- Approximation ratio of the uniform distribution is 0 by definition.

## Gates
**G0 (sanity; any failure voids the run).** (i) Exact and OR-Tools objectives agree to 1e-3 on every
instance. (ii) The numpy simulator matches the PennyLane circuit probabilities to 1e-9 (test). (iii) The
harness with COBYLA and random start reproduces `solve_qaoa` expected energy on small cases (test).

**G1 (beats chance).** Per size, family and p (engine arm): the lower CI bound of mean P(optimum) is
above the uniform-sampling P(optimum) of the same instances. Reported pass or fail per cell.

**G2 (solves the problem).** Per size, engine arm, p = 2 (the default): (a) top-64 hit rate point estimate
>= 0.95 with CI lower bound >= 0.90; (b) single-shot: mean P(optimum) >= 0.5. Pass or fail per size and
reading. p = 4 is shown next to it but does not decide the gate. Sizes where a fails are reported as
failures of QAOA at that size, with OR-Tools solving them optimally for comparison.

**G3 (scaling).** Per family and p: least-squares slope of log10 mean P(optimum) against qubit count
(instance-clustered bootstrap CI). "Decays with size" if the CI upper bound is below 0. No extrapolation
beyond the measured sizes is quoted as a result.

**G4 (depth).** Paired difference in P(optimum), p+1 versus p, per size (engine arm). "Helps" if the CI
lower bound is above 0, "hurts" if the upper bound is below 0, otherwise "no detectable change".

**G5 (optimiser and initialisation).** Paired difference versus COBYLA/random at the same size and p
(harness arm). Same three-way verdict as G4. Selecting the best variant per size after seeing results is
descriptive, not a gate.

**G6 (advantage).** Not claimed. A state-vector simulator costs O(2^n) per circuit evaluation, so simulator
wall time says nothing about hardware time, and no hardware time model or classical scaling gap is
measured here. Whatever the numbers show, the report states that no quantum advantage is claimed or
tested for. If QAOA does worse than OR-Tools or exact at a size (expected), that is reported plainly.

## Reporting rules
- Negative results (G1 to G5 failures, depth that hurts, optimisers that stall) go in the README and the
  summary with the same prominence as positive ones.
- 10 instances per cell is small; the report says so and does not over-read CIs that include zero.
- Raw per-run caches under `results/phase7/cache/` are git-ignored; summaries, tables and the report are
  committed.
- Illustrative values are labelled illustrative in code comments and outputs.

## Amendment 1 (2026-10-30, before any benchmark run): wording of G0(iii)
A probe while writing the tests showed that COBYLA is sensitive to float noise of order 1e-16: the numpy
harness and `solve_qaoa` evaluate the same objective (circuit probabilities agree to 1e-16), but their
COBYLA paths are not bit-identical, and at p >= 2 they sometimes end in different local optima. So
"reproduces `solve_qaoa`" in G0(iii) means: at p = 1 on small cases, expected energy within 1e-5 and
P(optimum) within 1e-3. At every depth the benchmark also reports the two arms' COBYLA/random results side
by side (paired per instance), so any systematic difference between arms is visible, not assumed away.

Two more pre-run clarifications found by the G0(i) check on the real instances (nothing had been run):
(1) **Engine bug fixed.** `build_qubo` set the penalty to (max R - min R) + 1, which is below one reward when
rewards are large and similar (these instances: about 1.1 to 1.2 with range 0.14). A second action for one
asset then lowered the energy, so the QUBO minimum was infeasible (exact objective 2.22 to 3.77 against
a feasible optimum 3.44 and so on, 7 of 50 probe instances). The penalty is now max(range, max R, -min R) + 1,
which equals the old value whenever the rewards straddle zero by more than their maximum. Regression tests
are in `tests/test_engine_qubo.py`. All instances are built with the fixed penalty; no benchmark number
comes from the old one.
(2) **G0(i) tolerance.** `solve_ortools` rounds rewards to 1e-3 (`round(r * 1000)`), so on near-ties its
true objective can trail the exact optimum by up to 0.5e-3 per asset. G0(i) now reads: exact optimum is
feasible, and OR-Tools' objective is within 0.5e-3 x assets below it (never above). "OR-Tools optimal" in the
tables means within that bound; the actual gaps are reported. The OR-Tools solver is not changed.

No threshold elsewhere is changed.

## Registration timeline (added at commit time, 2026-10-30)
This file is committed **after the full grid started**. It was not committed before the run, so the git
history alone does not prove ordering; the file timestamps do the work:
- 10:43:58 this file created, all gate text (G0 to G6, metrics, sizes, seeds, arms) written;
- 11:01:25 Amendment 1 added (still before any benchmark number existed);
- 11:03 to 11:04 smoke test of two units (8-qubit family B, instance 0: engine and numpy harness), written
  to a temp directory and not part of the results; I looked at one run's metrics (p = 1) to check that the
  fields were sane, and no gate was changed afterwards;
- 11:11:00 the full grid started (`results/phase7/run.log`).
Gate thresholds, sizes, seeds and instance counts were fixed before 11:11 and have not been changed since.
The only edits after the first draft are Amendment 1 above, written before the grid.

## Amendment 2 (2026-10-30, after the grid started; no gate changed): penalty rule is explicit
`build_qubo` now takes `penalty_rule`: `"sufficient"` (the engine default, P = max(range, max R, -min R) + 1,
the fix described in Amendment 1) or `"legacy"` (the original P = range + 1). Phase 6 paths pass `"legacy"`
and reproduce bit-for-bit (golden digests for all 1,050 train and eval tasks). **Every Phase 7 instance uses
`"sufficient"`.** The grid started (11:11) before this commit. Why the running grid uses "sufficient": its
worker processes imported `verispace_engine.qubo` at start, when the file already held the sufficient
formula (first committed as 9b4b555 after the run began, but present on disk since before 11:01, when
Amendment 1 described it); the explicit parameter added now defaults to that same formula, so the code the
workers ran and the code in this commit build identical matrices for the same input
(`tests/test_engine_qubo.py` pins both rules). Cached baselines show the exact optimum feasible on every
cached instance (G0(i)), which the legacy rule fails on some of these instances. All gate text (G0 to G6,
metrics, sizes, seeds, instance counts) predates the first run; see "Registration timeline".

## Outcome
(Filled in after the runs.)
