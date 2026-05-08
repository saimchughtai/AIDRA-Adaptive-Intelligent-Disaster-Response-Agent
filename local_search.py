"""
local_search.py
===============
AIDRA – Adaptive Intelligent Disaster Response Agent
AIC-201 CCP | Module 3: Local Search

Implements two local-search metaheuristics for optimising the *rescue
ordering* of victims (which victim should be attempted first, second, …):

  • Simulated Annealing (SA)  – stochastic escape from local optima
  • Hill Climbing (HC)        – deterministic greedy improvement

Both algorithms treat victim ordering as a permutation problem.
The cost of an ordering is the total A*-travel distance along the
sequence starting from RESCUE_BASE.

These results inform (but do not override) the fuzzy priority ranking;
the Controller uses SA/HC as a cross-check on scheduling efficiency.
"""

import math
import random
from search_algorithms import astar
from environment import RESCUE_BASE, log

# ─────────────────────────────────────────────────────────────────────────────
# SHARED COST FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def ordering_cost(order: list, victim_map: dict, grid: list) -> int:
    """
    Compute the total A* travel distance for visiting victims in *order*.

    The agent starts at RESCUE_BASE; after each rescue it moves to the
    next victim's position.  Unreachable victims are penalised with 9999.

    Parameters
    ----------
    order      : list of victim IDs defining the traversal sequence
    victim_map : {id: victim_dict} with 'pos' key
    grid       : current environment grid

    Returns
    -------
    Total path cost (integer)
    """
    pos   = RESCUE_BASE
    total = 0

    for vid in order:
        vpos = victim_map[vid]['pos']
        path, _ = astar(pos, vpos, grid, allow_risk=True)
        if path:
            total += len(path) - 1      # edges, not nodes
            pos    = vpos
        else:
            total += 9999               # severe penalty for unreachable victim

    return total


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATED ANNEALING
# ─────────────────────────────────────────────────────────────────────────────

def simulated_annealing(
    victims:  list,
    grid:     list,
    max_iter: int   = 500,
    T0:       float = 100.0,
    cooling:  float = 0.97,
    seed:     int   = 42,
) -> tuple[list, int]:
    """
    Simulated Annealing for victim rescue ordering.

    Neighbourhood operator : random swap of two positions in the permutation.
    Acceptance criterion   : Metropolis rule  (accept worse if e^{-Δ/T} > U[0,1]).
    Cooling schedule       : geometric  T_{k+1} = α * T_k.

    Parameters
    ----------
    victims  : list of victim dicts (all unrescued)
    grid     : current environment grid
    max_iter : number of SA iterations
    T0       : initial temperature
    cooling  : cooling rate α ∈ (0, 1)
    seed     : random seed for reproducibility

    Returns
    -------
    (best_order, best_cost)
    """
    random.seed(seed)

    ids = [v['id'] for v in victims if not v['rescued']]
    if len(ids) < 2:
        return ids, 0

    victim_map   = {v['id']: v for v in victims}
    current      = ids[:]
    random.shuffle(current)
    current_cost = ordering_cost(current, victim_map, grid)

    best      = current[:]
    best_cost = current_cost
    T         = T0

    log(f"SA start | initial_order={current} | cost={current_cost}")

    for iteration in range(max_iter):
        # Generate neighbour by swapping two random positions
        i, j     = random.sample(range(len(current)), 2)
        neighbour = current[:]
        neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
        nc = ordering_cost(neighbour, victim_map, grid)

        delta = nc - current_cost

        # Accept if better, or probabilistically if worse
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-6)):
            current      = neighbour
            current_cost = nc

        # Track global best
        if current_cost < best_cost:
            best      = current[:]
            best_cost = current_cost

        T *= cooling   # cool down

    log(f"SA final | best_order={best} | best_cost={best_cost}")
    return best, best_cost


# ─────────────────────────────────────────────────────────────────────────────
# HILL CLIMBING
# ─────────────────────────────────────────────────────────────────────────────

def hill_climbing(
    victims:  list,
    grid:     list,
    max_iter: int = 300,
    seed:     int = 42,
) -> tuple[list, int]:
    """
    Steepest-Ascent Hill Climbing for victim rescue ordering.

    At each step all pairwise swaps are evaluated; the best improving swap
    is applied.  Stops when no swap improves cost (local optimum).

    Parameters
    ----------
    victims  : list of victim dicts (all unrescued)
    grid     : current environment grid
    max_iter : maximum number of improvement rounds
    seed     : random seed for initial shuffle

    Returns
    -------
    (best_order, best_cost)
    """
    random.seed(seed + 1)     # different seed from SA for variety

    ids = [v['id'] for v in victims if not v['rescued']]
    if len(ids) < 2:
        return ids, 0

    victim_map   = {v['id']: v for v in victims}
    current      = ids[:]
    random.shuffle(current)
    current_cost = ordering_cost(current, victim_map, grid)

    log(f"HC start | initial_order={current} | cost={current_cost}")

    for _ in range(max_iter):
        best_swap     = None
        best_swap_cost = current_cost

        # Evaluate all O(n²) swaps
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                candidate        = current[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                nc               = ordering_cost(candidate, victim_map, grid)
                if nc < best_swap_cost:
                    best_swap_cost = nc
                    best_swap      = candidate

        if best_swap is None:
            break          # local optimum – no improving swap found

        current      = best_swap
        current_cost = best_swap_cost

    log(f"HC final | best_order={current} | best_cost={current_cost}")
    return current, current_cost


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON HELPER
# ─────────────────────────────────────────────────────────────────────────────

def compare_local_search(victims: list, grid: list) -> dict:
    """
    Run both SA and HC and return a comparison dictionary.

    Returns
    -------
    {
      'SA': {'order': [...], 'cost': int},
      'HC': {'order': [...], 'cost': int},
      'winner': 'SA' | 'HC' | 'tie',
    }
    """
    sa_order, sa_cost = simulated_annealing(victims, grid)
    hc_order, hc_cost = hill_climbing(victims, grid)

    winner = 'tie'
    if   sa_cost < hc_cost: winner = 'SA'
    elif hc_cost < sa_cost: winner = 'HC'

    log(f"Local search comparison | SA cost={sa_cost} | HC cost={hc_cost} | winner={winner}")

    return {
        'SA':     {'order': sa_order, 'cost': sa_cost},
        'HC':     {'order': hc_order, 'cost': hc_cost},
        'winner': winner,
    }