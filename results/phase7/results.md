# Phase 7 results: QAOA scaling benchmark

ILLUSTRATIVE spacecraft values; classical statevector simulator; **no quantum advantage is claimed** (G6).
Mean over 10 instances (5 for 18 and 20 qubits), each averaged over 3 optimiser seeds; brackets are 95% percentile-bootstrap CIs over instances. Gates: `docs/phase7_gates.md`.

## G0 sanity

- instances: 110; exact optimum feasible on all: **True**; exact equals landscape minimum: True
- OR-Tools within its reward-rounding bound on all: **True**; largest gap to exact 1.50e-03; instances where it trails exact at all: 11
- simulator fidelity (1e-9 vs PennyLane) and the engine/harness agreement are pinned by `tests/test_qaoa_bench.py`.

## Engine arm (`solve_qaoa`, COBYLA, random start), p = 2, with classical baselines

### Family B (2 actions, requested sizes)

| qubits | P(opt) QAOA | P(opt) uniform | enrichment | approx. ratio | top-64 hit | 64 shots hit | uniform 64 hit | uniform hit, same evals | G1 | G2a | G2b | evals | OR-Tools optimal | OR-Tools ms | exact ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.114 [0.092, 0.133] | 1.56e-02 | 7x | 0.743 [0.693, 0.794] | 1.00 [1.00, 1.00] | 0.93 [0.80, 1.00] | 0.635 | 0.994 | pass | pass | FAIL | 332 | 100% | 3.0 | 0.2 |
| 8 | 0.049 [0.042, 0.056] | 3.91e-03 | 13x | 0.674 [0.630, 0.721] | 1.00 [1.00, 1.00] | 0.89 [0.83, 0.95] | 0.222 | 0.741 | pass | pass | FAIL | 346 | 100% | 3.0 | 0.5 |
| 10 | 0.024 [0.017, 0.031] | 9.77e-04 | 24x | 0.650 [0.593, 0.704] | 0.87 [0.67, 1.00] | 0.70 [0.53, 0.83] | 0.061 | 0.278 | pass | FAIL | FAIL | 334 | 100% | 3.1 | 1.0 |
| 12 | 0.010 [0.008, 0.012] | 2.44e-04 | 40x | 0.665 [0.602, 0.732] | 0.80 [0.67, 0.93] | 0.43 [0.34, 0.51] | 0.016 | 0.079 | pass | FAIL | FAIL | 336 | 100% | 3.4 | 4.8 |
| 14 | 0.005 [0.004, 0.006] | 6.10e-05 | 88x | 0.698 [0.670, 0.728] | 0.63 [0.50, 0.77] | 0.28 [0.23, 0.33] | 0.004 | 0.021 | pass | FAIL | FAIL | 352 | 100% | 6.7 | 40.1 |
| 16 | 0.002 [0.002, 0.003] | 1.53e-05 | 162x | 0.633 [0.593, 0.674] | 0.60 [0.47, 0.77] | 0.14 [0.11, 0.17] | 0.001 | 0.005 | pass | FAIL | FAIL | 343 | 100% | 13.6 | 219.4 |

### Family A (3 actions, native fleet)

| qubits | P(opt) QAOA | P(opt) uniform | enrichment | approx. ratio | top-64 hit | 64 shots hit | uniform 64 hit | uniform hit, same evals | G1 | G2a | G2b | evals | OR-Tools optimal | OR-Tools ms | exact ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.059 [0.043, 0.078] | 1.56e-02 | 4x | 0.618 [0.542, 0.695] | 1.00 [1.00, 1.00] | 0.89 [0.82, 0.95] | 0.635 | 0.994 | pass | pass | FAIL | 330 | 100% | 2.4 | 0.2 |
| 9 | 0.014 [0.012, 0.017] | 1.95e-03 | 7x | 0.602 [0.522, 0.678] | 1.00 [1.00, 1.00] | 0.57 [0.51, 0.63] | 0.118 | 0.471 | pass | pass | FAIL | 326 | 100% | 2.5 | 0.5 |
| 12 | 0.003 [0.002, 0.004] | 2.44e-04 | 12x | 0.533 [0.484, 0.583] | 0.50 [0.37, 0.63] | 0.17 [0.12, 0.22] | 0.016 | 0.078 | pass | FAIL | FAIL | 331 | 100% | 3.8 | 5.9 |
| 15 | 0.001 [0.000, 0.001] | 3.05e-05 | 21x | 0.510 [0.483, 0.543] | 0.30 [0.13, 0.47] | 0.04 [0.03, 0.05] | 0.002 | 0.010 | pass | FAIL | FAIL | 330 | 100% | 6.0 | 92.8 |

Wall times in the parallel run are contended; see the serial timing table below.

## Depth p = 1 to 4 (engine arm): single-sample P(optimum)

### Family B (2 actions, requested sizes)

| qubits | uniform | p=1 | p=2 | p=3 | p=4 | G1 (p=1..4) | G4: 1 to 2 | 2 to 3 | 3 to 4 |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 1.56e-02 | 0.102 [0.095, 0.109] | 0.114 [0.092, 0.133] | 0.115 [0.091, 0.144] | 0.147 [0.121, 0.169] | pass/pass/pass/pass | no detectable change (+0.012) | no detectable change (+0.001) | helps (+0.032) |
| 8 | 3.91e-03 | 0.045 [0.038, 0.052] | 0.049 [0.042, 0.056] | 0.051 [0.039, 0.064] | 0.064 [0.046, 0.084] | pass/pass/pass/pass | no detectable change (+0.004) | no detectable change (+0.002) | no detectable change (+0.013) |
| 10 | 9.77e-04 | 0.020 [0.018, 0.021] | 0.024 [0.017, 0.031] | 0.021 [0.014, 0.027] | 0.028 [0.018, 0.040] | pass/pass/pass/pass | no detectable change (+0.004) | no detectable change (-0.003) | no detectable change (+0.007) |
| 12 | 2.44e-04 | 0.010 [0.009, 0.012] | 0.010 [0.008, 0.012] | 0.013 [0.009, 0.016] | 0.015 [0.010, 0.021] | pass/pass/pass/pass | no detectable change (-0.000) | helps (+0.003) | no detectable change (+0.002) |
| 14 | 6.10e-05 | 0.004 [0.004, 0.005] | 0.005 [0.004, 0.006] | 0.007 [0.006, 0.008] | 0.007 [0.006, 0.008] | pass/pass/pass/pass | no detectable change (+0.001) | no detectable change (+0.001) | no detectable change (+0.000) |
| 16 | 1.53e-05 | 0.002 [0.002, 0.003] | 0.002 [0.002, 0.003] | 0.003 [0.002, 0.003] | 0.004 [0.003, 0.006] | pass/pass/pass/pass | no detectable change (+0.000) | no detectable change (+0.000) | helps (+0.002) |

### Family A (3 actions, native fleet)

| qubits | uniform | p=1 | p=2 | p=3 | p=4 | G1 (p=1..4) | G4: 1 to 2 | 2 to 3 | 3 to 4 |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 1.56e-02 | 0.044 [0.039, 0.049] | 0.059 [0.043, 0.078] | 0.083 [0.057, 0.109] | 0.085 [0.067, 0.109] | pass/pass/pass/pass | no detectable change (+0.015) | helps (+0.024) | no detectable change (+0.002) |
| 9 | 1.95e-03 | 0.011 [0.010, 0.012] | 0.014 [0.012, 0.017] | 0.016 [0.012, 0.020] | 0.019 [0.016, 0.024] | pass/pass/pass/pass | helps (+0.004) | no detectable change (+0.001) | no detectable change (+0.004) |
| 12 | 2.44e-04 | 0.003 [0.002, 0.003] | 0.003 [0.002, 0.004] | 0.005 [0.003, 0.006] | 0.008 [0.006, 0.010] | pass/pass/pass/pass | no detectable change (+0.000) | helps (+0.002) | helps (+0.003) |
| 15 | 3.05e-05 | 0.000 [0.000, 0.001] | 0.001 [0.000, 0.001] | 0.001 [0.001, 0.001] | 0.002 [0.001, 0.002] | pass/pass/pass/pass | helps (+0.000) | no detectable change (+0.000) | helps (+0.001) |

## G3 scaling: slope of log10 P(optimum) per added qubit (negative = decays)

| family | arm | p | slope per qubit | 95% CI | verdict | sizes |
|---|---|---|---|---|---|---|
| B | engine | 1 | -0.168 | [-0.177, -0.160] | decays with size | 6-16 |
| B | engine | 2 | -0.166 | [-0.176, -0.155] | decays with size | 6-16 |
| B | engine | 3 | -0.159 | [-0.172, -0.146] | decays with size | 6-16 |
| B | engine | 4 | -0.153 | [-0.166, -0.141] | decays with size | 6-16 |
| A | engine | 1 | -0.217 | [-0.227, -0.208] | decays with size | 6-15 |
| A | engine | 2 | -0.218 | [-0.238, -0.200] | decays with size | 6-15 |
| A | engine | 3 | -0.208 | [-0.232, -0.186] | decays with size | 6-15 |
| A | engine | 4 | -0.184 | [-0.204, -0.166] | decays with size | 6-15 |
| B | harness | 1 | -0.168 | [-0.177, -0.160] | decays with size | 6-16 |
| B | harness | 2 | -0.168 | [-0.177, -0.158] | decays with size | 6-16 |
| B | harness | 3 | -0.157 | [-0.170, -0.144] | decays with size | 6-16 |
| B | harness | 4 | -0.153 | [-0.166, -0.141] | decays with size | 6-16 |
| A | harness | 1 | -0.217 | [-0.226, -0.208] | decays with size | 6-15 |
| A | harness | 2 | -0.217 | [-0.236, -0.199] | decays with size | 6-15 |
| A | harness | 3 | -0.206 | [-0.231, -0.184] | decays with size | 6-15 |
| A | harness | 4 | -0.186 | [-0.206, -0.168] | decays with size | 6-15 |

## G5 optimiser and initialisation (harness arm, same budget): P(optimum), paired difference vs COBYLA/random

### Family B, p = 2

| qubits | COBYLA/random | COBYLA/ramp | Nelder-Mead/random | Nelder-Mead/ramp | L-BFGS-B/random | L-BFGS-B/ramp |
|---|---|---|---|---|---|---|
| 6 | 0.115 (ref) | 0.099 (-0.015, no detectable change) | 0.117 (+0.002, no detectable change) | 0.123 (+0.008, no detectable change) | 0.124 (+0.009, no detectable change) | 0.127 (+0.012, no detectable change) |
| 8 | 0.049 (ref) | 0.059 (+0.010, helps) | 0.046 (-0.003, no detectable change) | 0.062 (+0.014, helps) | 0.054 (+0.005, no detectable change) | 0.060 (+0.011, no detectable change) |
| 10 | 0.024 (ref) | 0.029 (+0.005, helps) | 0.024 (+0.000, no detectable change) | 0.031 (+0.007, helps) | 0.030 (+0.006, helps) | 0.035 (+0.011, helps) |
| 12 | 0.010 (ref) | 0.016 (+0.007, helps) | 0.011 (+0.002, helps) | 0.016 (+0.006, helps) | 0.015 (+0.006, helps) | 0.016 (+0.006, helps) |
| 14 | 0.005 (ref) | 0.007 (+0.002, helps) | 0.005 (-0.000, no detectable change) | 0.007 (+0.002, helps) | 0.007 (+0.002, helps) | 0.008 (+0.003, helps) |
| 16 | 0.002 (ref) | 0.004 (+0.002, helps) | 0.003 (+0.000, no detectable change) | 0.004 (+0.002, helps) | 0.004 (+0.002, helps) | 0.005 (+0.003, helps) |

### Family B, p = 4

| qubits | COBYLA/random | COBYLA/ramp | Nelder-Mead/random | Nelder-Mead/ramp | L-BFGS-B/random | L-BFGS-B/ramp |
|---|---|---|---|---|---|---|
| 6 | 0.146 (ref) | 0.107 (-0.039, no detectable change) | 0.121 (-0.025, no detectable change) | 0.107 (-0.039, hurts) | 0.144 (-0.001, no detectable change) | 0.157 (+0.011, no detectable change) |
| 8 | 0.061 (ref) | 0.047 (-0.014, no detectable change) | 0.064 (+0.003, no detectable change) | 0.048 (-0.013, no detectable change) | 0.069 (+0.008, no detectable change) | 0.098 (+0.038, helps) |
| 10 | 0.028 (ref) | 0.037 (+0.009, no detectable change) | 0.037 (+0.009, helps) | 0.022 (-0.007, no detectable change) | 0.039 (+0.010, helps) | 0.052 (+0.024, helps) |
| 12 | 0.014 (ref) | 0.013 (-0.001, no detectable change) | 0.017 (+0.003, no detectable change) | 0.010 (-0.004, no detectable change) | 0.018 (+0.004, no detectable change) | 0.027 (+0.013, helps) |
| 14 | 0.007 (ref) | 0.008 (+0.000, no detectable change) | 0.009 (+0.001, no detectable change) | 0.004 (-0.003, hurts) | 0.010 (+0.003, helps) | 0.013 (+0.006, helps) |
| 16 | 0.004 (ref) | 0.003 (-0.001, no detectable change) | 0.006 (+0.002, helps) | 0.003 (-0.002, hurts) | 0.007 (+0.002, helps) | 0.009 (+0.004, helps) |

### Family A, p = 2

| qubits | COBYLA/random | COBYLA/ramp | Nelder-Mead/random | Nelder-Mead/ramp | L-BFGS-B/random | L-BFGS-B/ramp |
|---|---|---|---|---|---|---|
| 6 | 0.057 (ref) | 0.052 (-0.005, no detectable change) | 0.047 (-0.010, no detectable change) | 0.074 (+0.018, no detectable change) | 0.049 (-0.008, no detectable change) | 0.089 (+0.033, helps) |
| 9 | 0.014 (ref) | 0.013 (-0.001, no detectable change) | 0.015 (+0.000, no detectable change) | 0.020 (+0.006, helps) | 0.016 (+0.002, no detectable change) | 0.026 (+0.012, helps) |
| 12 | 0.003 (ref) | 0.004 (+0.001, helps) | 0.004 (+0.001, helps) | 0.006 (+0.003, helps) | 0.006 (+0.003, helps) | 0.008 (+0.006, helps) |
| 15 | 0.001 (ref) | 0.001 (+0.000, helps) | 0.001 (-0.000, no detectable change) | 0.002 (+0.001, helps) | 0.002 (+0.001, helps) | 0.003 (+0.002, helps) |

### Family A, p = 4

| qubits | COBYLA/random | COBYLA/ramp | Nelder-Mead/random | Nelder-Mead/ramp | L-BFGS-B/random | L-BFGS-B/ramp |
|---|---|---|---|---|---|---|
| 6 | 0.087 (ref) | 0.090 (+0.003, no detectable change) | 0.078 (-0.009, no detectable change) | 0.039 (-0.048, hurts) | 0.090 (+0.003, no detectable change) | 0.096 (+0.009, no detectable change) |
| 9 | 0.021 (ref) | 0.024 (+0.004, no detectable change) | 0.019 (-0.002, no detectable change) | 0.010 (-0.011, hurts) | 0.022 (+0.001, no detectable change) | 0.029 (+0.009, helps) |
| 12 | 0.008 (ref) | 0.006 (-0.001, no detectable change) | 0.006 (-0.002, hurts) | 0.003 (-0.005, hurts) | 0.007 (-0.000, no detectable change) | 0.012 (+0.004, helps) |
| 15 | 0.002 (ref) | 0.002 (+0.000, no detectable change) | 0.001 (-0.000, no detectable change) | 0.001 (-0.001, hurts) | 0.002 (+0.000, no detectable change) | 0.004 (+0.003, helps) |

G5 verdict counts over all (family, size, p in {2, 4}) cells:

| variant | helps | hurts | no detectable change |
|---|---|---|---|
| COBYLA/ramp | 7 | 0 | 13 |
| L-BFGS-B/ramp | 16 | 0 | 4 |
| L-BFGS-B/random | 9 | 0 | 11 |
| Nelder-Mead/ramp | 8 | 7 | 5 |
| Nelder-Mead/random | 4 | 1 | 15 |

## Engine (PennyLane) vs harness (numpy), both COBYLA/random: paired difference in P(optimum)

| family | qubits | p | engine | harness | difference [95% CI] |
|---|---|---|---|---|---|
| B | 6 | 1 | 0.102 | 0.102 | -0.000 [-0.000, +0.000] |
| B | 6 | 2 | 0.114 | 0.115 | +0.001 [-0.000, +0.002] |
| B | 6 | 3 | 0.115 | 0.115 | -0.001 [-0.002, +0.001] |
| B | 6 | 4 | 0.147 | 0.146 | -0.001 [-0.002, +0.000] |
| B | 8 | 1 | 0.045 | 0.045 | +0.000 [+0.000, +0.000] |
| B | 8 | 2 | 0.049 | 0.049 | -0.000 [-0.001, +0.000] |
| B | 8 | 3 | 0.051 | 0.051 | -0.001 [-0.003, +0.000] |
| B | 8 | 4 | 0.064 | 0.061 | -0.003 [-0.014, +0.004] |
| B | 10 | 1 | 0.020 | 0.020 | +0.000 [+0.000, +0.000] |
| B | 10 | 2 | 0.024 | 0.024 | +0.000 [-0.000, +0.000] |
| B | 10 | 3 | 0.021 | 0.021 | -0.000 [-0.001, +0.000] |
| B | 10 | 4 | 0.028 | 0.028 | -0.000 [-0.001, +0.001] |
| B | 12 | 1 | 0.010 | 0.010 | +0.000 [-0.000, +0.000] |
| B | 12 | 2 | 0.010 | 0.010 | -0.000 [-0.000, +0.000] |
| B | 12 | 3 | 0.013 | 0.013 | +0.000 [-0.000, +0.001] |
| B | 12 | 4 | 0.015 | 0.014 | -0.001 [-0.002, -0.000] |
| B | 14 | 1 | 0.004 | 0.004 | -0.000 [-0.000, +0.000] |
| B | 14 | 2 | 0.005 | 0.005 | -0.000 [-0.000, +0.000] |
| B | 14 | 3 | 0.007 | 0.007 | +0.000 [-0.000, +0.000] |
| B | 14 | 4 | 0.007 | 0.007 | +0.000 [+0.000, +0.000] |
| B | 16 | 1 | 0.002 | 0.002 | +0.000 [-0.000, +0.000] |
| B | 16 | 2 | 0.002 | 0.002 | -0.000 [-0.000, +0.000] |
| B | 16 | 3 | 0.003 | 0.003 | +0.000 [-0.000, +0.000] |
| B | 16 | 4 | 0.004 | 0.004 | -0.000 [-0.000, +0.000] |
| A | 6 | 1 | 0.044 | 0.044 | -0.000 [-0.001, +0.000] |
| A | 6 | 2 | 0.059 | 0.057 | -0.002 [-0.006, +0.002] |
| A | 6 | 3 | 0.083 | 0.087 | +0.004 [-0.000, +0.011] |
| A | 6 | 4 | 0.085 | 0.087 | +0.002 [-0.004, +0.009] |
| A | 9 | 1 | 0.011 | 0.011 | -0.000 [-0.000, +0.000] |
| A | 9 | 2 | 0.014 | 0.014 | -0.000 [-0.000, +0.000] |
| A | 9 | 3 | 0.016 | 0.015 | -0.000 [-0.000, -0.000] |
| A | 9 | 4 | 0.019 | 0.021 | +0.001 [-0.000, +0.004] |
| A | 12 | 1 | 0.003 | 0.003 | +0.000 [-0.000, +0.000] |
| A | 12 | 2 | 0.003 | 0.003 | -0.000 [-0.000, -0.000] |
| A | 12 | 3 | 0.005 | 0.005 | +0.000 [+0.000, +0.001] |
| A | 12 | 4 | 0.008 | 0.008 | -0.000 [-0.001, +0.000] |
| A | 15 | 1 | 0.000 | 0.000 | -0.000 [-0.000, -0.000] |
| A | 15 | 2 | 0.001 | 0.001 | +0.000 [-0.000, +0.000] |
| A | 15 | 3 | 0.001 | 0.001 | +0.000 [-0.000, +0.000] |
| A | 15 | 4 | 0.002 | 0.002 | +0.000 [-0.000, +0.000] |

## Extension: 18 and 20 qubits (harness arm, p = 2, COBYLA/random, 5 instances)

| qubits | P(opt) | uniform | approx. ratio | top-64 hit | evals | mean run s (contended) |
|---|---|---|---|---|---|---|
| 18 | 0.001 [0.001, 0.001] | 3.81e-06 | 0.581 [0.547, 0.621] | 0.47 [0.33, 0.60] | 339 | 334 |
| 20 | 0.000 [0.000, 0.001] | 9.54e-07 | 0.606 [0.575, 0.637] | 0.47 [0.20, 0.67] | 332 | 1321 |

## Serial timing (nothing else running; one instance per size, p = 2, seed 0)

| qubits | engine s (evals) | numpy harness s | exact ms | OR-Tools ms |
|---|---|---|---|---|
| 6 | 1.08 (360) | 0.22 | 0.1 | 4.5 |
| 8 | 1.28 (360) | 0.23 | 0.1 | 0.9 |
| 10 | 1.33 (321) | 0.23 | 0.3 | 1.5 |
| 12 | 1.85 (360) | 0.42 | 1.3 | 1.0 |
| 14 | 3.16 (360) | 0.91 | 6.9 | 1.1 |
| 16 | 9.16 (360) | 4.41 | 35.7 | 1.1 |

| qubits | engine s per circuit evaluation (incl. setup) | numpy s per evaluation, p = 2 |
|---|---|---|
| 16 | 0.036 | 0.011 |
| 18 | 0.131 | 0.170 |
| 20 | 0.687 | 0.791 |

