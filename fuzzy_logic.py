"""
fuzzy_logic.py
==============
AIDRA – Adaptive Intelligent Disaster Response Agent
AIC-201 CCP | Module 6: Fuzzy Logic Uncertainty Handling

Implements a Mamdani fuzzy inference system to compute a scalar
priority score for each victim from noisy, multi-attribute input data.

Why fuzzy logic?
----------------
Sensor readings in disaster environments are imprecise:
  • A victim tagged "critical" might have uncertainty of ±1 severity class.
  • Risk-zone boundaries shift as fires spread.
  • Wait-time estimates carry ≥30 % error.

Standard probability models require clean numerical distributions;
fuzzy sets handle these *linguistic* uncertainties naturally.

Inference Pipeline
------------------
  Fuzzification → Rule Evaluation → Aggregation → Defuzzification

Inputs (three)
  severity      : numeric code  1 (minor) | 2 (moderate) | 3 (critical)
  risk_exposure : fraction of path through high-risk cells  [0, 1]
  wait_time     : estimated steps before rescue begins       [0, 15]

Output
  priority_score : scalar in [0, 1]  (higher = rescue sooner)

Membership Functions
  All triangular:  trimf(x, a, b, c)
  • severity    → {low, medium, high}
  • risk_exposure → {low, high}
  • wait_time   → {short, long}

Rules (5 Mamdani rules)
  R1: IF sev=HIGH   AND wait=LONG  → priority = 0.90
  R2: IF sev=HIGH   AND wait=SHORT → priority = 0.85
  R3: IF sev=MEDIUM               → priority = 0.55
  R4: IF sev=LOW    AND risk=HIGH  → priority = 0.45
  R5: IF sev=LOW    AND risk=LOW   → priority = 0.20

Defuzzification: weighted centroid
  score = Σ(activation_i × centre_i) / Σ(activation_i)
"""

from environment import SEVERITY, log

# ─────────────────────────────────────────────────────────────────────────────
# MEMBERSHIP FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def trimf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.

       1        /\\
               /  \\
      0 ------/----\\------
             a    b    c

    Returns 0 outside [a, c], peaks at 1 when x == b.
    """
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a + 1e-9)
    return (c - x) / (c - b + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# FUZZIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def fuzzify_severity(sev_code: int) -> dict:
    """
    Fuzzify the numeric severity code (1, 2, 3) into membership degrees.

    Linguistic terms
    ----------------
    low    : peaks at sev=1  (minor)
    medium : peaks at sev=2  (moderate)
    high   : peaks at sev=3  (critical)
    """
    return {
        'low':    trimf(sev_code, 0, 1, 2),
        'medium': trimf(sev_code, 1, 2, 3),
        'high':   trimf(sev_code, 2, 3, 4),
    }


def fuzzify_risk(risk_exposure: float) -> dict:
    """
    Fuzzify risk_exposure (fraction of path in high-risk cells, [0,1]).

    Linguistic terms
    ----------------
    low  : peaks at 0  (safe path)
    high : peaks at 1  (entirely through hazard)
    """
    return {
        'low':  trimf(risk_exposure, -0.1, 0.0, 0.50),
        'high': trimf(risk_exposure,  0.3, 1.0, 1.10),
    }


def fuzzify_wait(wait_time: float) -> dict:
    """
    Fuzzify wait_time (steps before rescue, [0, 15]).

    Linguistic terms
    ----------------
    short : peaks at 0  (immediate rescue)
    long  : peaks at 15 (severe delay)
    """
    return {
        'short': trimf(wait_time, -1.0,  0.0,  7.0),
        'long':  trimf(wait_time,  5.0, 15.0, 16.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUZZY RULE BASE
# ─────────────────────────────────────────────────────────────────────────────

# Each rule: (antecedent_strength_fn, consequent_centre_value)
# Antecedent strength = min(μ_A1, μ_A2, …)  [AND = min operator]

RULES = [
    # centre  description
    (0.90, "IF sev=HIGH   AND wait=LONG  → VERY HIGH priority"),
    (0.85, "IF sev=HIGH   AND wait=SHORT → HIGH priority"),
    (0.55, "IF sev=MEDIUM               → MEDIUM priority"),
    (0.45, "IF sev=LOW    AND risk=HIGH  → MEDIUM-LOW priority"),
    (0.20, "IF sev=LOW    AND risk=LOW   → LOW priority"),
]


def evaluate_rules(sev_mf: dict, risk_mf: dict, wait_mf: dict) -> list[float]:
    """
    Evaluate all five fuzzy rules using min-AND aggregation.

    Returns a list of activation strengths, one per rule.
    """
    activations = [
        min(sev_mf['high'],   wait_mf['long']),    # R1
        min(sev_mf['high'],   wait_mf['short']),   # R2
        sev_mf['medium'],                           # R3
        min(sev_mf['low'],    risk_mf['high']),     # R4
        min(sev_mf['low'],    risk_mf['low']),      # R5
    ]
    return activations


# ─────────────────────────────────────────────────────────────────────────────
# DEFUZZIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def defuzzify(activations: list[float]) -> float:
    """
    Weighted centroid defuzzification.

    score = Σ(activation_i × centre_i) / Σ(activation_i)

    If all activations are zero (degenerate case) returns 0.5 as neutral.
    """
    centres  = [rule[0] for rule in RULES]
    weighted = sum(a * c for a, c in zip(activations, centres))
    total    = sum(activations)
    if total < 1e-9:
        return 0.5
    return round(weighted / total, 4)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def compute_priority_score(
    severity_str:  str,
    risk_exposure: float = 0.1,
    wait_time:     float = 0.0,
) -> float:
    """
    Compute the fuzzy priority score for a single victim.

    Parameters
    ----------
    severity_str  : 'critical' | 'moderate' | 'minor'
    risk_exposure : fraction of planned route through high-risk cells [0, 1]
    wait_time     : estimated steps before rescue can begin [0, 15]

    Returns
    -------
    priority_score ∈ [0, 1]  (higher = rescue sooner)
    """
    sev_code = SEVERITY.get(severity_str, 2)

    sev_mf  = fuzzify_severity(sev_code)
    risk_mf = fuzzify_risk(risk_exposure)
    wait_mf = fuzzify_wait(wait_time)

    activations = evaluate_rules(sev_mf, risk_mf, wait_mf)
    score       = defuzzify(activations)

    return score


def rank_victims(victims: list, grid: list,
                 search_fn=None) -> tuple[list, dict]:
    """
    Score and rank all unrescued victims by fuzzy priority.

    For each victim:
      • risk_exposure = fraction of A* path through high-risk cells
        (requires a path from rescue base → victim; uses search_fn if given)
      • wait_time is initialised to 0 at simulation start; the Controller
        updates it between steps.

    Parameters
    ----------
    victims   : list of victim dicts
    grid      : current environment grid
    search_fn : callable(start, goal, grid) → (path, expanded)
                typically astar; if None, risk_exposure defaults to 0.1

    Returns
    -------
    ranked_victims : list of victim dicts sorted by score descending
    scores         : {victim_id: score}
    """
    from environment import RESCUE_BASE

    scores = {}
    for v in victims:
        if v['rescued']:
            continue

        # Estimate risk exposure from planned A* path
        risk_exposure = 0.1    # default
        if search_fn:
            path, _ = search_fn(RESCUE_BASE, v['pos'], grid, allow_risk=True)
            if path and len(path) > 1:
                n_risk    = sum(1 for p in path if grid[p[0]][p[1]] == 2)
                risk_exposure = n_risk / (len(path) - 1)

        score             = compute_priority_score(v['severity'],
                                                   risk_exposure,
                                                   wait_time=0.0)
        scores[v['id']]   = score

        log(f"Fuzzy priority | V{v['id']} ({v['severity']}) "
            f"risk_exp={risk_exposure:.2f} → score={score}")

    ranked = sorted(
        [v for v in victims if not v['rescued']],
        key=lambda v: -scores.get(v['id'], 0)
    )
    return ranked, scores


def escalate_critical(ranked: list, ml_model,
                      predict_fn, grid: list) -> list:
    """
    Override fuzzy ranking for victims whose ML-predicted survival < 0.50.

    Such victims are moved to the front of the queue regardless of their
    fuzzy score, resolving Conflicting Objective 2 (prioritisation vs
    throughput): a critically endangered victim must not be delayed.

    Parameters
    ----------
    ranked      : list already sorted by fuzzy score (descending)
    ml_model    : fitted classifier with predict_proba
    predict_fn  : function(model, severity, dist, risk, wait, kit) → prob
    grid        : current environment grid

    Returns
    -------
    re_ranked : list with escalated victims at the front
    """
    from environment   import RESCUE_BASE
    from search_algorithms import astar

    escalated = []
    normal    = []

    for v in ranked:
        path, _ = astar(RESCUE_BASE, v['pos'], grid)
        dist     = len(path) - 1 if path else 15
        n_risk   = sum(1 for p in (path or []) if grid[p[0]][p[1]] == 2)
        prob     = predict_fn(ml_model, v['severity'], dist, n_risk, 0, 1)

        v['survival_prob'] = prob

        if prob < 0.50:
            escalated.append(v)
            log(f"ML escalation | V{v['id']} survival_prob={prob:.3f} < 0.5 → moved to front")
        else:
            normal.append(v)

    return escalated + normal