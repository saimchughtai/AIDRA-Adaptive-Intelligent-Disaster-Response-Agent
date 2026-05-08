"""
environment.py
==============
AIDRA – Adaptive Intelligent Disaster Response Agent
AIC-201 CCP | Module 1: Environment Model

Defines the disaster grid, victim data, resource inventory, and the
shared decision log.  All other modules import constants from here.

Cell encoding
-------------
  0  → free       (traversable, cost = 1)
  1  → blocked    (impassable – rubble / wall)
  2  → high-risk  (traversable, cost = 3 – fire / structural hazard)
"""

import copy

# ─────────────────────────────────────────────────────────────────────────────
# GRID CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

GRID_SIZE = 10

# 0 = free, 1 = blocked, 2 = high-risk
BASE_MAP = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 1, 0, 2, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 2, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 2, 0, 0],
    [0, 2, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 2, 0, 0, 0],
    [0, 1, 0, 0, 0, 2, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 2],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
]

RESCUE_BASE      = (0, 0)
MEDICAL_CENTERS  = [(9, 9), (9, 0)]   # two safe medical centres

# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY ENCODING
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY     = {'critical': 3, 'moderate': 2, 'minor': 1}
SEVERITY_REV = {3: 'critical', 2: 'moderate', 1: 'minor'}

CELL_COST = {0: 1, 1: float('inf'), 2: 3}  # cost per cell type

# ─────────────────────────────────────────────────────────────────────────────
# INITIAL VICTIM ROSTER  (2 critical, 2 moderate, 1 minor)
# ─────────────────────────────────────────────────────────────────────────────

VICTIMS_INIT = [
    {
        'id': 0, 'pos': (2, 5), 'severity': 'critical',
        'rescued': False, 'rescue_time': None, 'risk_score': None,
        'survival_prob': None, 'assigned_ambulance': None,
    },
    {
        'id': 1, 'pos': (5, 7), 'severity': 'critical',
        'rescued': False, 'rescue_time': None, 'risk_score': None,
        'survival_prob': None, 'assigned_ambulance': None,
    },
    {
        'id': 2, 'pos': (3, 3), 'severity': 'moderate',
        'rescued': False, 'rescue_time': None, 'risk_score': None,
        'survival_prob': None, 'assigned_ambulance': None,
    },
    {
        'id': 3, 'pos': (7, 2), 'severity': 'moderate',
        'rescued': False, 'rescue_time': None, 'risk_score': None,
        'survival_prob': None, 'assigned_ambulance': None,
    },
    {
        'id': 4, 'pos': (6, 8), 'severity': 'minor',
        'rescued': False, 'rescue_time': None, 'risk_score': None,
        'survival_prob': None, 'assigned_ambulance': None,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE INVENTORY
# ─────────────────────────────────────────────────────────────────────────────

RESOURCES = {
    'ambulances':    2,   # max 2 victims per ambulance at once
    'rescue_teams':  1,   # can service only 1 location at a time
    'medical_kits': 10,   # total kits available
}

# ─────────────────────────────────────────────────────────────────────────────
# DECISION LOG  (shared mutable list; all modules append here)
# ─────────────────────────────────────────────────────────────────────────────

DECISION_LOG: list[str] = []


def log(msg: str) -> None:
    """Append a timestamped message to the shared decision log and print it."""
    DECISION_LOG.append(msg)
    print(f"[LOG] {msg}")


def clear_log() -> None:
    """Reset the log between simulation runs."""
    DECISION_LOG.clear()


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC EVENT SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

# Scheduled events: {step_number: [(row, col, new_cell_value, description), ...]}
SCHEDULED_EVENTS = {
    3: [(4, 3, 1, "Road at (4,3) blocked by aftershock")],
    5: [(2, 7, 1, "Road at (2,7) blocked by fire spread")],
}


def apply_events(grid: list, step: int) -> tuple[list, list[str]]:
    """
    Apply any scheduled environmental events at *step*.

    Parameters
    ----------
    grid : current live grid (will be deep-copied before modification)
    step : current simulation step index

    Returns
    -------
    updated_grid : modified copy of the grid
    events       : list of human-readable event descriptions (empty if none)
    """
    updated_grid = copy.deepcopy(grid)
    events = []

    for row, col, new_val, desc in SCHEDULED_EVENTS.get(step, []):
        if updated_grid[row][col] != new_val:          # avoid duplicate writes
            updated_grid[row][col] = new_val
            events.append(desc)

    return updated_grid, events


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: fresh deep-copies for a new simulation run
# ─────────────────────────────────────────────────────────────────────────────

def fresh_state() -> tuple[list, list, dict]:
    """Return fresh (grid, victims, resources) copies ready for a new run."""
    return (
        copy.deepcopy(BASE_MAP),
        copy.deepcopy(VICTIMS_INIT),
        copy.deepcopy(RESOURCES),
    )