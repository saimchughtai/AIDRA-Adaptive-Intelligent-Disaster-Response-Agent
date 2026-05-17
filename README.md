# 🚨 AIDRA — Adaptive Intelligent Disaster Response Agent



**A hybrid AI system for real-time disaster victim evacuation planning**  
*Bahria University, Islamabad Campus — AIC-201 Artificial Intelligence*



</div>

---

## 📌 Overview

AIDRA is a **hybrid AI agent** that manages victim rescue operations in a simulated dynamic disaster environment. It integrates **four AI paradigms** into a single coherent pipeline:

| Module | Technique | Purpose |
|--------|-----------|---------|
| 🔍 Search & Planning | BFS, DFS, Greedy BFS, A* | Optimal risk-weighted route finding |
| 🔄 Local Search | Simulated Annealing, Hill Climbing | Global victim ordering optimisation |
| 📦 Constraint Satisfaction | Backtracking + MRV + Forward Checking | Ambulance & resource allocation |
| 🤖 Machine Learning | kNN, Naïve Bayes, MLP | Victim survival probability prediction |
| 🌫️ Fuzzy Logic | Mamdani Inference System | Priority scoring under uncertainty |

The agent operates on a **10×10 grid** with blocked roads, high-risk fire zones, 5 victims of varying severity, 2 ambulances, 1 rescue team, and 2 medical centres. It replans in real time when roads are blocked mid-mission.

---

## 🎬 Demo Video

> 📹 **[Watch the 2-minute demonstration on LinkedIn →](https://www.linkedin.com/posts/m-saim-chughtai-204043287_aidra-adaptive-intelligent-disaster-response-ugcPost-7459314326328442880-USDa?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEWYSWQBqX1aaLzTNfrpfadEIS_-j9aMTE8)**

---

## 📊 Results

### Key Performance Indicators

| KPI | Value |
|-----|-------|
| ✅ Victims Saved | **5 / 5** |
| ⏱️ Average Rescue Time | **14.0 steps** |
| 📐 Path Optimality Ratio (A* / BFS) | **0.94** |
| 🚑 Resource Utilisation Rate | **1.00** |
| ⚠️ Risk Exposure Score | **0 cells** |
| 🔁 CSP Backtracks (MRV) | **4** |
| 🧠 Best ML F1-Score (MLP & NB) | **0.83** |
| 🔄 Replanning Events | **1** |

### ML Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| kNN (k=5) | 0.91 | 0.80 | 0.75 | 0.77 |
| Naïve Bayes | 0.94 | 0.92 | 0.75 | 0.83 |
| **MLP (32-16)** ✅ | 0.93 | 0.75 | **0.94** | **0.83** |

> MLP selected as active model due to highest **recall (0.94)** — critical in safety applications where missing a high-risk victim is the costliest error.

### Fuzzy Priority Scores

| Victim | Severity | Score | Rank |
|--------|----------|-------|------|
| V0 | Critical | 0.85 | 1 |
| V1 | Critical | 0.85 | 2 |
| V2 | Moderate | 0.55 | 3 |
| V3 | Moderate | 0.55 | 4 |
| V4 | Minor | 0.20 | 5 |

### Rescue Timeline

```
V0 (CRI) ████████░░░░░░░░░░░░░░   9 steps
V2 (MOD) ██████████░░░░░░░░░░░░  10 steps
V1 (CRI) ████████████░░░░░░░░░░  12 steps
V3 (MOD) █████████████████░░░░░  17 steps  ← delayed by aftershock replanning
V4 (MIN) ██████████████████████  22 steps
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/[your-username]/AIDRA-AIC201.git
cd AIDRA-AIC201

# 2. Install dependencies
pip install numpy scikit-learn matplotlib
```

### Run the Simulation

```bash
# Full run — simulation + all 7 figures + JSON results export
python main.py

# Skip figure generation (faster terminal output only)
python main.py --no-figures

# Full run + save complete decision log to logs/decision_log.txt
python main.py --export-log
```

### Expected Output

```
╔══════════════════════════════════════════════════════════════════╗
║       AIDRA – Adaptive Intelligent Disaster Response Agent       ║
║       AIC-201 Complex Computing Problem  |  Bahria University    ║
╚══════════════════════════════════════════════════════════════════╝

► Running AIDRA simulation…

[LOG] AIDRA Simulation Start
[LOG] ML pipeline: training and evaluating models…
[LOG] Best ML model selected: MLP (32-16) (F1=0.83)
[LOG] CSP: starting resource allocation
[LOG] Fuzzy priority | V0 (critical) → score=0.85
...
[LOG] ✓ V1 rescued | path_len=12 | risk_cells=0 | survival_prob=0.333
[LOG] ✓ V0 rescued | path_len=7  | risk_cells=0 | survival_prob=0.935
...

► Simulation complete in 0.70s

  Victim Rescue Priority Order
────────────────────────────────────────────────────
  1. V1 (critical)  survival_prob=0.333  rescue_time=12
  2. V0 (critical)  survival_prob=0.935  rescue_time=9
  3. V2 (moderate)  survival_prob=0.982  rescue_time=10
  4. V3 (moderate)  survival_prob=0.258  rescue_time=17
  5. V4 (minor   )  survival_prob=0.429  rescue_time=22
────────────────────────────────────────────────────

✔  All done.  Check figures/ and logs/ for outputs.
```

---

## 📁 Project Structure

```
AIDRA-AIC201/
│
├── main.py                # Entry point — CLI, orchestrates all modules
├── environment.py         # Grid map, victims, resources, event scheduler
├── search_algorithms.py   # BFS, DFS, Greedy BFS, A*, route comparison
├── local_search.py        # Simulated Annealing + Hill Climbing
├── csp_allocation.py      # CSP: backtracking + MRV + forward checking
├── ml_models.py           # kNN, Naïve Bayes, MLP training & evaluation
├── fuzzy_logic.py         # Mamdani fuzzy inference — 5 rules, centroid defuzz
├── controller.py          # Master simulation loop + replanning + KPI computation
├── visualizer.py          # 7 matplotlib figures (PDF output)
│
├── figures/               # Auto-generated output figures
│   ├── grid.pdf           # Environment grid with A* routes
│   ├── search_compare.pdf # Nodes expanded + path length comparison
│   ├── ml_compare.pdf     # ML metric bar chart
│   ├── confusion.pdf      # 3 confusion matrices side by side
│   ├── fuzzy.pdf          # Fuzzy priority scores per victim
│   ├── kpis.pdf           # KPI dashboard + CSP backtrack comparison
│   └── rescue_timeline.pdf# Gantt-style rescue sequence chart
│
├── logs/                  # Auto-generated logs
│   ├── simulation_results.json  # Full results in JSON
│   └── decision_log.txt         # Complete agent decision trace
│
└── README.md
```

---

## 🧠 Module Details

### `environment.py`
Defines the 10×10 disaster grid (0 = free, 1 = blocked, 2 = high-risk), all 5 victims with severity levels, resource inventory (2 ambulances, 1 rescue team, 10 kits), the shared decision log, and the dynamic event scheduler that injects road blockages at specified steps.

### `search_algorithms.py`
Implements all four graph-search algorithms with a shared neighbour generator and risk-cost function. The `compare_all_algorithms()` function benchmarks all algorithms from any start to any goal. The `select_best_route()` function resolves **Conflicting Objective 1** (time vs. risk) by comparing A* standard and safe-route weighted costs.

### `local_search.py`
**Simulated Annealing**: geometric cooling ($T_0=100$, $\alpha=0.97$, 500 iterations), random-swap neighbourhood. Achieves ordering cost **24** vs Hill Climbing's **27**, demonstrating SA's superiority on this permutation problem.

### `csp_allocation.py`
Full CSP specification with variables (victims), domains (ambulance IDs), and three hard constraints. Two solver variants: `csp_plain_backtracking()` (baseline) and `csp_mrv_solver()` (MRV + forward checking). Useful for comparing backtrack counts.

### `ml_models.py`
Generates a 400-sample synthetic dataset, trains all three classifiers with 80/20 stratified split, evaluates with accuracy/precision/recall/F1/confusion matrix, and selects the best model by F1. The `predict_survival()` function feeds into the controller for per-victim risk assessment.

### `fuzzy_logic.py`
Full Mamdani inference implementation from scratch (no external fuzzy library). Includes `fuzzify_severity()`, `fuzzify_risk()`, `fuzzify_wait()`, `evaluate_rules()`, and `defuzzify()` as separate composable functions. The `escalate_critical()` function resolves **Conflicting Objective 2** (priority vs. throughput) via ML override.

### `controller.py`
The central orchestrator. Runs the three-phase simulation: (0) initialise + train + allocate, (1) rescue loop with event detection and replanning, (2) KPI computation. The `export_results()` function writes a clean JSON file stripping non-serialisable objects.

### `visualizer.py`
Seven publication-quality figures for the IEEE report. All use consistent colour palettes and are saved as high-resolution PDFs at 180 DPI. `generate_all_figures()` is a convenience wrapper for the full figure set.

---

## 🗺️ Environment Grid

```
  0   1   2   3   4   5   6   7   8   9
0 [B] [ ] [ ] [X] [ ] [ ] [ ] [ ] [ ] [ ]
1 [ ] [R] [ ] [ ] [ ] [X] [ ] [R] [ ] [ ]
2 [ ] [ ] [ ] [X] [ ] [V0] [ ] [ ] [X] [ ]
3 [X] [ ] [ ] [ ] [R] [ ] [X] [ ] [ ] [ ]
4 [ ] [ ] [X] [ ] [ ] [ ] [ ] [R] [ ] [ ]
5 [ ] [R] [ ] [ ] [X] [ ] [ ] [V1] [ ] [X]
6 [ ] [ ] [ ] [X] [ ] [ ] [R] [ ] [V4] [ ]
7 [ ] [X] [ ] [ ] [ ] [R] [ ] [V3] [X] [ ]
8 [ ] [ ] [ ] [ ] [X] [ ] [ ] [ ] [ ] [R]
9 [M] [ ] [X] [ ] [ ] [ ] [X] [ ] [ ] [M]
```
`B` = rescue base `(0,0)` | `M` = medical centre | `X` = blocked | `R` = high-risk | `V` = victim

---

## ⚙️ Configuration

All parameters are centralised in `environment.py` and can be modified without touching other files:

```python
# Grid size
GRID_SIZE = 10

# Victim positions and severities
VICTIMS_INIT = [
    {'id': 0, 'pos': (2, 5), 'severity': 'critical'},
    {'id': 1, 'pos': (5, 7), 'severity': 'critical'},
    ...
]

# Resources
RESOURCES = {'ambulances': 2, 'rescue_teams': 1, 'medical_kits': 10}

# Dynamic events (step: [(row, col, new_value, description)])
SCHEDULED_EVENTS = {
    3: [(4, 3, 1, "Road at (4,3) blocked by aftershock")],
    5: [(2, 7, 1, "Road at (2,7) blocked by fire spread")],
}
```

---

## 📄 Report

The full **6-page IEEE conference paper** is available in the repository:  
📄 `AIDRA_IEEE_Report.pdf`

The report covers:
- Problem formulation with performance measure equation
- Full system architecture and module interaction flow
- Search algorithm comparison (nodes expanded, path length, optimality ratio)
- CSP formulation with MRV and forward checking analysis
- ML model training, evaluation, confusion matrices, and model selection justification
- Mamdani fuzzy rule base with defuzzification
- Dynamic replanning decision log
- Complete KPI table with conflicting-objective resolution analysis

---

## 👥 Authors

| Name | Enrollment | Email |
|------|-----------|-------|
| [Student Name 1] | [EnrollmentNo1] | [email1@buic.edu.pk] |
| [Student Name 2] | [EnrollmentNo2] | [email2@buic.edu.pk] |

**Instructor:** Dr. Arshad Farhad  
**Course:** AIC-201 Artificial Intelligence  
**Institution:** Bahria University, Islamabad Campus  
**Semester:** 5-A

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 2.0 | Array operations, dataset generation |
| `scikit-learn` | ≥ 1.8 | kNN, Naïve Bayes, MLP classifiers |
| `matplotlib` | ≥ 3.10 | All visualisations |

No external fuzzy-logic library is required — the Mamdani inference engine is implemented from scratch in `fuzzy_logic.py`.

---

## 📜 License

This project is submitted as coursework for AIC-201 at Bahria University. All code is original work by the authors listed above.

---

<div align="center">
<sub>Built with ❤️ for AIC-201 | Bahria University Islamabad</sub>
</div>
