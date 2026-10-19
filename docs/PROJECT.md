# HelioGuard

Physics-verified storm decisions for satellites in Earth orbit

Project document, version 1.1, October 2026

Domain: Earth-orbit space (low Earth orbit to geostationary orbit)

Part of the VeriSpace platform, alongside the Quantum-Clock Navigator
(deep space)

Prepared by: \[Founder name\], AI Decision Scientist

## 1. The aim behind this project

This project was started with a clear personal aim: to build a working
prototype that space companies, space engineers, and scientists can see
and use, and that shows a rare combination of skills in one place:
decision science, reinforcement learning, large language models, quantum
computing, and space physics.

The goal is not a concept slide. It is a prototype that solves a problem
companies are actually facing, built honestly enough that an engineer at
SpaceX, NASA, or Space42 would look at it and think: this person
understands our problem, and this could help us.

- **Professional aim:** establish a profile as one of very few people
  combining quantum, AI, and space knowledge in a shipped, verifiable
  product.

- **Industry aim:** give satellite operators a tool that turns
  solar-storm forecasts into protective decisions they can trust and
  audit.

- **Community aim:** release open tools and learning material for
  students of space engineering and heliophysics.

## 2. Executive summary

Solar storms are one of the few hazards that hit every satellite
operator at once. They heat and expand Earth's upper atmosphere,
increasing drag on low-orbit satellites, and they charge spacecraft in
higher orbits. Forecasts of these storms exist. What is missing is the
step from forecast to decision: what each specific satellite, launch,
and ground station should do, and when.

HelioGuard closes that gap. A language model reasons step by step from a
storm alert to asset-specific actions. Every step is checked by a
physics verifier before it counts, and the model is trained on those
verified rewards with Group Variance Policy Optimization (GVPO) using
Verifiable Process Rewards (VPR). A quantum layer built on PennyLane
schedules the response across a whole fleet (QAOA), and a hybrid quantum
neural network learns corrections to the atmosphere model from real
satellite decay. Decisions are designed to remain sound even when the
forecast is wrong.

**Status:** Phase 1 is complete. A working physics verifier replays the
February 2022 Starlink storm, scores AI decision traces step by step,
and has produced the project's first finding: once the storm hit, every
in-orbit option still lost the satellite, and the decision that mattered
was the one made before deployment.

## 3. Where this project sits: Earth-orbit space

VeriSpace has two projects that cover two different regions of space.
This document covers the first.

| **Project**                | **Region**                                                                                             | **Main hazard or need**                                           |
|----------------------------|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| HelioGuard (this document) | Earth-orbit space: low Earth orbit (about 160 to 2,000 km) up to geostationary orbit (about 35,786 km) | Solar storms: drag, charging, radiation, communication disruption |
| Quantum-Clock Navigator    | Deep space: cislunar space, Mars, and beyond                                                           | Navigation without constant help from Earth                       |

HelioGuard's core market is Earth orbit. Its radiation module also
extends naturally to crewed missions beyond Earth's magnetic field, such
as Artemis, which is the bridge to the deep-space project.

## 4. The problem

### 4.1 The physics of a solar storm

The Sun regularly releases solar flares and coronal mass ejections: huge
clouds of magnetised plasma. When one reaches Earth, it disturbs Earth's
magnetic field, causing a geomagnetic storm. Storm strength is measured
by the planetary Kp index (0 to 9) and classified by NOAA from G1
(minor) to G5 (extreme).

- **Drag in low Earth orbit.** Storm energy heats the thermosphere,
  which expands upwards. Satellites suddenly fly through denser air,
  lose speed, and drop. Low satellites, especially freshly launched
  ones, can re-enter.

- **Charging in higher orbits.** Energetic electrons build up charge on
  and inside spacecraft, particularly in geostationary orbit, which can
  cause electrical discharges, anomalies, and outages.

- **Radiation and communications.** Energetic particles damage
  electronics and threaten crews; disturbed ionospheric conditions
  disrupt radio links.

### 4.2 Real events

In February 2022, SpaceX launched 49 Starlink satellites into a low
initial orbit near 210 km. A minor geomagnetic storm the next day raised
drag by up to 50 percent compared with previous launches, according to
SpaceX. The satellites were put into an edge-on safe mode to reduce
drag, but up to 40 re-entered.

In May 2024, the strongest geomagnetic storm in over 20 years struck.
MIT researchers analysing it found that its strength and duration were
poorly predicted even one day ahead, and noted that most space-weather
forecasts provide no uncertainty estimate. Operators therefore make
high-stakes decisions without knowing how far to trust the forecast.

## 5. Why this is a gap for companies

### 5.1 What exists today

- **Forecasts and alerts:** NOAA's Space Weather Prediction Center and
  ESA's Space Weather Office issue indices, alerts, and forecasts.

- **AI forecasting:** this side is crowded. NASA and IBM released Surya,
  an open AI model for solar data, and several research groups build
  storm and solar-wind forecasting models.

- **Early decision tools:** work such as HELIO, presented at the 2026
  Small Satellite Conference, describes an AI layer that turns forecasts
  into risk classifications. That paper explicitly identifies the gap:
  operators still have to translate forecasts into spacecraft-specific
  decisions, which adds delay, especially across large constellations.

### 5.2 What is still missing

1.  **Decisions that hold up when the forecast is wrong.** Existing
    tools predict or classify; few test whether a plan survives if the
    storm is stronger or longer than forecast.

2.  **Reasoning that can be audited.** A risk score cannot be inspected.
    Operators and insurers need to justify decisions afterwards, step by
    step.

3.  **Per-asset, fleet-wide actions.** Turning one alert into
    coordinated actions for hundreds of satellites, ground passes, and
    launches, without conflicts.

**HelioGuard's positioning:** storm decisions you can trust even when
the forecast is wrong, with every step verified by physics.

## 6. Who this helps

| **Audience**                                                                                    | **How HelioGuard helps**                                                |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Low-orbit constellations (Starlink, OneWeb and Eutelsat, Amazon Kuiper, Chinese constellations) | Fleet-wide protective actions; launch and deployment-altitude decisions |
| Geostationary operators (Space42/Yahsat/Thuraya and others)                                     | Charging-risk awareness, protective modes, service planning             |
| Smaller operators without space-weather teams                                                   | Expert-level decisions without an in-house specialist                   |
| Launch providers                                                                                | Go or no-go and deployment-altitude recommendations ahead of storms     |
| Satellite insurers                                                                              | Audit trails and post-storm attribution: was a loss caused by the Sun?  |
| Space agencies and weather offices                                                              | A decision layer on top of the forecasts they already produce           |
| Space engineering students                                                                      | Hands-on tools for drag, orbit decay, and operations decisions          |
| Heliophysics and space-physics students                                                         | A practical bridge from solar physics to real-world impact              |

### 6.1 Is it useful for space engineering and astrophysics students?

Yes, with one precision. Space engineering students benefit directly:
orbital decay, drag, thrust budgets, and operational decision-making are
core course topics, and HelioGuard turns them into interactive,
verifiable tools. Within astrophysics, the relevant students are those
in heliophysics and space physics, because the Sun is the source of
everything HelioGuard models. Students in unrelated areas such as
cosmology or galaxy research will find it less relevant. The open-source
release will include notebooks designed for teaching.

## 7. What HelioGuard does

Inputs are storm alerts and indices, solar data, the operator's fleet
(orbits, mass, area, thrust), and the forecast's uncertainty. Outputs
are, for each asset, a recommended action with timing, a
physics-verified justification, and a plain-English briefing. A human
operator approves every action.

STORM BRIEFING \| forecast Kp 7 (G3), arrival in 18 h, uncertainty: Kp 5
to 8

Batch 12 (210 km, just deployed): DELAY orbit-raise start; enter edge-on
mode at T-2 h

verified: at 210 km raising fails above density x1.2 (drag exceeds 70 mN
thrust)

Launch L-31 (planned T+6 h): HOLD 48 h; deploy at 230 km or above if
launching

verified: raising feasible from 228 km at density x2.0

GEO-3: protective mode; defer non-critical manoeuvres

Plan remains sound across the full forecast range (Kp 5 to 8).

## 8. How it works: technical design

### 8.1 Architecture

Storm data (NOAA SWPC, OMNIWeb, SunPy/SDO) + forecast distribution

\+ Fleet state (orbits from CelesTrak/Space-Track, spacecraft
parameters)

-\> LLM decision policy: step-by-step JSON decision trace

-\> VPR physics verifier: each step checked -\> step rewards

-\> GVPO training loop (offline) -\> improved policy

-\> Robustness check across forecast scenarios

-\> PennyLane QAOA fleet scheduler (vs OR-Tools baseline)

-\> Operator briefing -\> human approval

### 8.2 Physics models

Atmospheric density. Phase 1 uses a piecewise-exponential reference
atmosphere multiplied by a storm factor. Phase 3 replaces it with
NRLMSIS 2.0 through the pymsis package, driven by the real F10.7 solar
flux and Ap geomagnetic indices.

Drag force and orbit decay. For a near-circular orbit of semi-major axis
a, with speed v, density rho, drag coefficient Cd, area A and mass m:

F_drag = 0.5 \* rho \* v^2 \* Cd \* A (N)

v = sqrt(mu / a) (mu = 3.986e14 m^3/s^2)

da/dt = 2 \* F_t \* a^1.5 / (m \* sqrt(mu)) (energy balance; F_t =
thrust - drag)

Re-entry treated as unrecoverable below 150 km

Feasibility. Orbit raising is feasible only when available thrust
exceeds drag in the raising attitude. Safe modes reduce the effective
area A. Phase 3 adds elliptical orbits and full propagation with Orekit.

Storm level. NOAA scale from Kp: below 5 is G0, Kp 5 is G1, up to Kp 9
as G5. Phase 4 adds geostationary charging risk from GOES
energetic-electron flux, using NOAA's high-energy electron alert
thresholds.

### 8.3 Decision trace format

The language model outputs a structured trace. Each step is a claim the
verifier can test:

\[

{"type": "classify_storm", "g_scale": "G1"},

{"type": "predict_decay", "asset": "batch12", "attitude": "raise",

"hours": 12, "altitude_loss_km": 40},

{"type": "check_feasibility", "action": "raise_orbit", "feasible":
false},

{"type": "choose_action", "action": "safe_mode_edge_on"}

\]

### 8.4 The VPR verifier

| **Step type**        | **Verification rule**                                                          | **Reward** |
|----------------------|--------------------------------------------------------------------------------|------------|
| classify_storm       | Matches the NOAA G-scale for the scenario's Kp                                 | 1 or 0     |
| predict_decay        | Within 25% of the physics-propagated altitude loss                             | 1 or 0     |
| check_feasibility    | Matches the thrust-versus-drag comparison                                      | 1 or 0     |
| choose_action        | Feasible, and best among feasible actions by survival, then remaining altitude | 1 or 0     |
| robustness (phase 7) | Chosen plan remains acceptable across sampled forecast scenarios               | 1 or 0     |

The process reward for a trace is the mean of its step rewards. Because
physics checks each step, the reward is dense and free of the noise that
comes from human or model-based judging.

### 8.5 Training with GVPO

1.  Sample a group of k decision traces (for example 8) from the current
    model for each storm scenario.

2.  Score every trace with the verifier to get process rewards.

3.  Compute group-centred rewards (each reward minus the group mean).

4.  Update the model with GVPO, which weights the gradient using these
    rewards within a KL-constrained objective, giving stable training.
    GVPO also allows off-policy training on stored traces, so expensive
    simulations can be reused.

Model: an open model of about 1.5 to 4 billion parameters with LoRA
adapters (via Unsloth), trained with TRL or verl and the GVPO reference
code. This keeps training feasible on a single high-memory GPU.

### 8.6 Decisions that survive forecast error

Each forecast becomes a distribution rather than a single number.
Historical forecast errors are modelled with emcee or PyMC, and
HelioGuard samples many storm scenarios from that distribution.
Candidate plans are scored across all samples, and the recommended plan
maximises worst-case or tail-risk performance (conditional value at
risk) rather than performance under the single most likely forecast.

### 8.7 Quantum layer

Fleet scheduling as a QUBO. Let x(i,a) = 1 if asset i takes action a.
The objective rewards risk reduction and penalises conflicts:

minimise - sum_i,a R(i,a) x(i,a)

\+ P1 \* sum_i ( sum_a x(i,a) - 1 )^2 (one action per asset)

\+ P2 \* sum_w ( load in ground window w over capacity )^2

Small instances are solved with PennyLane's QAOA tools on a simulator
and compared with Google OR-Tools on identical problems. No quantum
speed advantage is claimed.

### 8.8 Quantum machine learning: hybrid QNN for density correction

**The problem it solves.** Even the best atmosphere models are wrong
during storms, often by tens of percent. Those errors flow straight into
decay predictions and therefore into decisions. Real satellites reveal
the truth: their measured orbit decay implies an effective density along
their path. The gap between that and the model is a learnable
correction.

**Target.** The log ratio of effective density (derived from orbit decay
in CelesTrak orbit histories, using the drag equation and an estimated
ballistic coefficient) to NRLMSIS model density. A value of 0.2 means
the real atmosphere was about 22 percent denser than modelled.

**Inputs (6 features, mapped to 6 qubits).** Altitude, solar flux F10.7,
geomagnetic index Ap, hours since storm onset, local solar time, and
latitude band.

**Model.** A hybrid quantum neural network: features are scaled and
encoded into qubit rotation angles (angle embedding), processed by a
shallow trainable entangling circuit, and measured; a small classical
layer turns the measurements into the correction. In PennyLane, the
circuit becomes a PyTorch layer through qml.qnn.TorchLayer, so it trains
with standard tools.

import pennylane as qml, torch

n_qubits, n_layers = 6, 3

dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")

def circuit(inputs, weights):

qml.AngleEmbedding(inputs, wires=range(n_qubits))

qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

return \[qml.expval(qml.PauliZ(w)) for w in range(n_qubits)\]

shapes = {"weights": (n_layers, n_qubits, 3)}

model = torch.nn.Sequential(

qml.qnn.TorchLayer(circuit, shapes), \# quantum layer: 54 trainable
parameters

torch.nn.Linear(n_qubits, 1)) \# classical head -\> log density ratio

**Training.** Adam optimiser on mean squared error. Data is split by
storm event (train, validation, test on different storms) so the model
is judged on storms it has never seen.

**Keeping training healthy.** Shallow circuits, small initial weights,
and local measurements reduce the risk of barren plateaus, where
gradients vanish and training stalls.

**Honest benchmark.** The QNN is compared with a classical neural
network of the same parameter count, and with gradient boosting.
Reported metrics: error in the density correction, improvement in decay
prediction, parameter count, and training time. Whatever the outcome, it
is published. Matching the classical model is still a credible result.

**How it connects to decisions.** A validated correction improves the
density model that the verifier and robustness checks use. It is only
switched on after it beats the uncorrected model on held-out storms, so
the verifier never depends on an unproven model.

### 8.9 Post-storm attribution (phase 6)

When a satellite anomaly or loss occurs, HelioGuard compares its timing
and location against storm data (SpacePy, GOES particle data,
geomagnetic indices) and reasons step by step about whether the Sun was
the likely cause. This module serves operators deciding whether a design
flaw exists, and insurers settling claims.

### 8.10 How the system gets smarter: environment, harness, and learning
loop

The project combines a language model, reinforcement learning, physics
knowledge, and quantum tools. Each part has a different job, and only
one of them makes the system smarter over time: reinforcement learning
against a strong, physics-based environment.

| **Part**                               | **Role, in plain terms**                                                   | **Makes the system smarter?**                  |
|----------------------------------------|----------------------------------------------------------------------------|------------------------------------------------|
| Language model                         | The reasoning mind: understands situations, reasons step by step, explains | Starting point                                 |
| Reinforcement learning (GVPO with VPR) | Practice with a strict teacher: attempts, scores, improves                 | Yes: this is the learning engine               |
| RL environment and verifier            | The teacher and the exams, built on physics                                | Only as good as we make it                     |
| Knowledge (physics models, real data)  | Textbooks and real-world experience                                        | Grows as real data is added                    |
| Quantum (QAOA, QNN)                    | Specialist tools for specific sub-problems                                 | No: sharpens specific parts, not the reasoning |

**What the model becomes.** Training makes the model reason like a
satellite operations and flight-dynamics engineer, within the physics
its verifier contains. It becomes a domain expert, not a general
scientist, and it does not discover new physics on its own.

**Current status.** Partly built. The verifier (the reward function) and
two scenarios (February 2022 storm and a quiet case) exist in Python.
The environment wrapper, scenario generator, harness, and training loop
are planned for phases 2 and 5.

**The environment (built in Claude Code as a first-class deliverable).**
A Gymnasium-style environment with four parts: a scenario generator that
produces thousands of varied, realistic situations (storm strengths and
timings, forecast errors, orbit altitudes, spacecraft types, fleet
sizes, launch schedules); the physics verifier that scores every step;
held-out test scenarios the model never trains on; and real-data
scenarios so expertise matches reality rather than only the simulation.

**The harness.** The machinery around the model: prompt templates, a
strict trace parser, sampling a group of attempts per scenario, sending
each to the verifier, logging rewards, feeding GVPO updates, and running
evaluations. It is built and tested like product code, because it could
become a product in its own right: a benchmark for physics-verified
space reasoning.

### 8.11 Safeguards: making sure more training means real improvement

- **Reward hacking.** If the verifier has loopholes, more training
  teaches the model to exploit them. For example, the decay check
  currently accepts predictions within 25 percent, so a model could
  learn vague answers that always fall inside that band; tolerances will
  tighten and scoring will become graded rather than pass or fail. Every
  verifier rule is tested against deliberately wrong answers that must
  fail.

- **Verification before volume.** Improving the verifier comes before
  adding training tasks; every verifier improvement makes all future
  training more valuable.

- **Generalisation.** Performance is judged only on held-out scenarios
  and real-data replays, never on training scenarios.

- **User feedback.** Users do not train the model simply by using it.
  Operator approvals and rejections can become a later training signal,
  but only after they are checked against physics, because human
  feedback is noisy.

### 8.12 The quantum-tool agent: how decisions are actually made

The final design makes quantum part of the decision itself. The language
model (Qwen) is the decision-maker, and it reasons with tools, including
quantum ones, the way an engineer reasons with solvers. This is a
quantum-augmented agent: Qwen remains a classical model, and the quantum
methods do the sub-problems they suit.

| **Tool**               | **What it returns**                                           | **Kind**                             |
|------------------------|---------------------------------------------------------------|--------------------------------------|
| simulate_decay         | Altitude over time for an asset, action and storm             | Physics                              |
| check_feasibility      | Whether thrust beats drag, with both forces                   | Physics                              |
| min_safe_altitude      | Lowest deployment altitude where raising works                | Physics                              |
| qnn_density_correction | Corrected atmospheric density for the conditions              | Quantum machine learning (PennyLane) |
| qaoa_schedule          | Best fleet-wide protective schedule, plus the OR-Tools result | Quantum optimisation (PennyLane)     |

**Multi-turn RL environment.** Each training episode is a conversation
with the environment: the environment presents a situation; Qwen calls
tools; the environment executes each call and returns the result; Qwen
continues reasoning and finally gives a decision with reasons. The
independent physics verifier then scores every step: whether the right
tool was called, whether its result was used correctly, and whether the
final decision is physically sound. A small cost for unnecessary tool
calls teaches Qwen to use expensive tools, such as QAOA, only when a
problem is genuinely hard.

situation -\> Qwen: reason -\> call tool -\> environment runs tool -\>
result

-\> Qwen: reason -\> (more tool calls) -\> final decision + reasons

-\> physics verifier scores every step -\> GVPO update (training)

-\> at deployment: verifier guardrail -\> operator approves

**Guardrail at deployment.** The same verifier checks every live
recommendation before an operator sees it. Any step that fails is
flagged and the recommendation is blocked.

**Research track.** A small quantum reinforcement-learning policy for
one sub-decision (safe mode versus raise orbit) is offered to Qwen as an
additional tool and benchmarked against a classical policy of the same
size. If it ever proves better, Qwen will learn to use it; if not, the
result is still reported.

## 9. Data sources

| **Source**                                  | **Used for**                                                                    |
|---------------------------------------------|---------------------------------------------------------------------------------|
| NOAA Space Weather Prediction Center        | Kp index, storm alerts, GOES particle and electron flux                         |
| NASA OMNIWeb                                | Historical solar wind and geomagnetic indices for replays                       |
| NASA Solar Dynamics Observatory (via SunPy) | Solar imagery and eruption context                                              |
| CelesTrak and Space-Track                   | Real orbit data (TLEs) before and after storms                                  |
| NRLMSIS 2.0 (via pymsis)                    | Atmospheric density driven by real indices                                      |
| Historical events                           | February 2022 Starlink loss, May 2024 storm, documented geostationary anomalies |

## 10. Technology stack

| **Category**             | **Packages**                                                                                 |
|--------------------------|----------------------------------------------------------------------------------------------|
| Space environment        | SunPy, SpacePy, Astropy, pymsis                                                              |
| Orbits                   | sgp4, Skyfield, Orekit (Python wrapper)                                                      |
| Numerics and uncertainty | NumPy, SciPy, emcee or PyMC                                                                  |
| Reinforcement learning   | GVPO reference code, TRL or verl, Unsloth, Gymnasium                                         |
| Language models          | Hugging Face Transformers                                                                    |
| Quantum                  | PennyLane (QAOA, and hybrid QNNs via qml.qnn.TorchLayer), PyTorch, lightning.qubit simulator |
| Classical baselines      | Google OR-Tools; PyTorch neural networks; gradient boosting                                  |
| Interface                | Interactive HTML dashboard; later a web app                                                  |

## 11. What has been built

Phase 1 consists of a Python package with four modules (atmosphere,
physics, verifier, scenarios), a demo script, and an interactive
storm-replay web page.

| **Response**      | **Quiet atmosphere**   | **February 2022 storm (density x1.5)**               |
|-------------------|------------------------|------------------------------------------------------|
| Raise orbit       | Survives, about 243 km | Not feasible: drag exceeds thrust. Lost at hour 31.8 |
| Edge-on safe mode | Survives, about 166 km | Lost at hour 35.3 (best available)                   |
| Hold              | Lost at hour 19.9      | Lost at hour 13.2                                    |

Under illustrative spacecraft parameters (260 kg, Cd 2.2, 4 square
metres raising area, 1.5 square metres edge-on, 70 mN thrust), raising
becomes feasible from about 218 km in this storm, 228 km at double
density, and 243 km at triple density. A naive AI trace scored 0 of 4
verified steps; a physics-aware trace scored 4 of 4, giving GVPO
group-centred rewards of -0.5 and +0.5.

## 12. How we build it in Claude Code

### 12.1 Working method

- One phase per session. Start each session by asking Claude Code to
  read this document, the README, and the existing code.

- Tests first: each phase begins with tests that encode its "done when"
  criteria.

- Keep a CLAUDE.md file in the repository with project rules:
  illustrative values must be labelled, every quantum result needs a
  classical baseline, and the system never commands spacecraft.

- Commit after every passing phase, with a short results note in the
  README.

### 12.2 Phases

| **Phase**                     | **Work**                                                                                                              | **Done when**                                                               |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| 0\. Setup                     | Project tooling; tests that lock the Phase 1 results                                                                  | Tests pass with no physics changes                                          |
| 1\. Engine core               | Tool registry, trace and tool-call schema, verifier interface (shared VeriSpace engine)                               | Any tool and verifier rule plugs in through one interface                   |
| 2\. Real physics and data     | pymsis density from real indices, Orekit or sgp4 propagation, CelesTrak orbit histories                               | Feb 2022 and May 2024 replays driven by real data                           |
| 3\. Tools                     | simulate_decay, check_feasibility, min_safe_altitude, qaoa_schedule (PennyLane plus OR-Tools), qnn_density_correction | Each tool unit-tested; QAOA and QNN reported against baselines              |
| 4\. Multi-turn RL environment | Scenario generator, tool execution, step verifier, tool-use costs, held-out sets, loophole tests                      | Thousands of scenarios; wrong answers fail every rule                       |
| 5\. Harness                   | Prompt templates, strict parser, rollout sampling, logging, evaluation against rule-based and base-model baselines    | One command runs a full evaluation                                          |
| 6\. GVPO training             | Qwen (1.5B to 4B) with LoRA; curriculum from easy to hard storms; ablations                                           | Trained agent beats the base model and the rule baseline on held-out storms |
| 7\. Quantum benchmarks        | QAOA at larger sizes, QNN on real decay data, quantum RL sub-policy experiment                                        | Published tables against classical methods                                  |
| 8\. Product app               | Backend API, live NOAA and CelesTrak feeds, console UI from the prototype, deployment guardrail                       | An operator can load a fleet and get verified recommendations               |
| 9\. Release                   | README, teaching notebooks, preprint draft, LinkedIn posts                                                            | Public repository and first post                                            |

### 12.3 Example Claude Code prompts

Full session-by-session prompts are in docs/ROADMAP.md and
FIRST_PROMPTS.md in the starter kit.

Phase 3: "Implement qaoa_schedule as a tool: build the QUBO from the
fleet and

ground windows, solve small instances with PennyLane QAOA, and always
return

the OR-Tools solution alongside. Tests first."

Phase 4: "Turn the verifier into a multi-turn environment: reset()
returns a

scenario prompt; step(action) executes a tool call or scores a final
decision."

### 12.4 Repository structure

helioguard/

verispace_engine/ tools.py schema.py verifier_base.py env_base.py
harness/

helioguard/ atmosphere.py physics.py verifier.py scenarios.py

tools/ (decay, feasibility, qaoa_schedule, qnn_density)

env.py data/ (noaa.py, celestrak.py)

training/ gvpo_train.py configs/ evals/

app/ api/ ui/ (from docs/prototype)

notebooks/ tests/ docs/ CLAUDE.md README.md

## 13. How success is measured

- **Decision quality:** assets saved or survival time gained against no
  action and rule-based baselines.

- **Robustness:** performance when the real storm differs from the
  forecast.

- **Reasoning quality:** mean verified-step score of the trained model
  against the base model.

- **Quantum benchmark:** schedule quality and solve time of QAOA against
  OR-Tools.

- **QNN benchmark:** density-correction error of the hybrid QNN against
  a matched classical network, on unseen storms.

- **Real-data validation:** recommendations compared with observed decay
  of real satellites in replayed storms.

## 14. Launch and community

- **LinkedIn:** "Replaying the storm that took out 40 satellites",
  followed by GVPO and VPR results, the QAOA benchmark, and the hybrid
  QNN density-correction results.

- **Open source:** repository with a five-minute quick start, and the
  verifier published as a standalone drag-and-decay tool.

- **Students:** teaching notebooks on orbital decay, storm impacts, and
  decision-making under uncertainty.

- **Research:** an arXiv preprint on physics-verified process rewards
  for space operations.

- **Conferences:** Small Satellite Conference and the AGU Fall Meeting
  (space weather sessions).

## 15. Risks and honesty notes

- Phase 1 spacecraft values are illustrative, not operator
  specifications, and are labelled as such.

- Large operators such as SpaceX have in-house capability; the strongest
  early users are mid-size operators and insurers.

- No quantum computing or quantum machine learning advantage is claimed;
  QAOA and QNN results are always benchmarked against classical methods.

- HelioGuard recommends; humans decide. It never commands spacecraft.

## 16. Glossary

| **Term**                     | **Meaning**                                                                                                |
|------------------------------|------------------------------------------------------------------------------------------------------------|
| Coronal mass ejection (CME)  | A large cloud of magnetised plasma released by the Sun                                                     |
| Kp index and G-scale         | Measures of geomagnetic disturbance (Kp 0 to 9) and NOAA's storm classes (G1 to G5)                        |
| Thermosphere                 | The upper atmosphere layer where low-orbit satellites fly; it expands during storms                        |
| Drag coefficient (Cd)        | A number describing how strongly a shape is slowed by the atmosphere                                       |
| QUBO                         | Quadratic unconstrained binary optimisation: a problem format for quantum optimisers                       |
| QAOA                         | Quantum Approximate Optimization Algorithm, a variational quantum method for QUBO problems                 |
| Quantum neural network (QNN) | A trainable quantum circuit used like a neural network layer; here combined with classical layers (hybrid) |
| Angle embedding              | Encoding each input feature as a rotation angle on a qubit                                                 |
| Barren plateau               | A training problem in quantum circuits where gradients become vanishingly small                            |
| VPR                          | Verifiable Process Rewards: rewarding each reasoning step using a verifier                                 |
| GVPO                         | Group Variance Policy Optimization: a stable method for training language models with rewards              |
| CVaR                         | Conditional value at risk: average outcome in the worst cases, used for robust decisions                   |
