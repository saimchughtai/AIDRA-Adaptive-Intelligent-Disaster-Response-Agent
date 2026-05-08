"""
csp_allocation.py
=================
AIDRA – Adaptive Intelligent Disaster Response Agent
AIC-201 CCP | Module 4: Constraint Satisfaction Problem (CSP)

Models resource allocation as a CSP and solves it with:
  • Backtracking search (baseline)
  • Backtracking + MRV (Minimum Remaining Values) heuristic
  • Backtracking + MRV + Forward Checking (full solver)

CSP Specification
-----------------
Variables  : one per unrescued victim  V = {v0, v1, v2, v3, v4}
Domains    : each victim can be assigned to ambulance 0 or 1
               D(vi) = {0, 1}

Hard Constraints
  C1 – Ambulance capacity  : Σ_{v assigned to a} ≤ 2  ∀a ∈ {0,1}
  C2 – Kit availability    : total assigned ≤ medical_kits
  C3 – Rescue-team limit   : rescue team can handle 1 location at a time
                             (enforced at scheduling level, not CSP)

The solver returns a satisfying assignment {victim_id: ambulance_id}
along with a backtrack count for performance comparison.
"""

from environment import SEVERITY, log

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRAINT CHECKERS
# ─────────────────────────────────────────────────────────────────────────────

def check_capacity(assignment: dict, ambulance: int,
                   max_per_ambulance: int = 2) -> bool:
    """C1: Ambulance may not carry more than *max_per_ambulance* victims."""
    return sum(1 for a in assignment.values() if a == ambulance) < max_per_ambulance


def check_kits(assignment: dict, total_kits: int) -> bool:
    """C2: Number of assigned victims must not exceed available medical kits."""
    return len(assignment) < total_kits


def all_constraints_satisfied(assignment: dict, victim_id: int,
                               ambulance: int, resources: dict) -> bool:
    """
    Check whether assigning *victim_id* → *ambulance* violates any constraint.
    Called before committing the assignment.
    """
    temp = {**assignment, victim_id: ambulance}
    if not check_capacity(temp, ambulance, max_per_ambulance=2):
        return False
    if not check_kits(temp, resources['medical_kits']):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD CHECKING
# ─────────────────────────────────────────────────────────────────────────────

def forward_check(domains: dict, assignment: dict,
                  resources: dict) -> dict | None:
    """
    Prune domains of unassigned variables given the current partial assignment.

    Returns the pruned domains dict, or None if any domain becomes empty
    (constraint violation detected ahead of time).
    """
    pruned = {vid: list(d) for vid, d in domains.items()}

    for vid, dom in pruned.items():
        if vid in assignment:
            continue
        valid = []
        for amb in dom:
            if all_constraints_satisfied(assignment, vid, amb, resources):
                valid.append(amb)
        if not valid:
            return None         # domain wipe-out → dead end, backtrack
        pruned[vid] = valid

    return pruned


# ─────────────────────────────────────────────────────────────────────────────
# MRV (MINIMUM REMAINING VALUES) ORDERING
# ─────────────────────────────────────────────────────────────────────────────

def mrv_select(unassigned: list, domains: dict) -> int:
    """
    Return the victim ID with the fewest remaining valid domain values (MRV).

    Breaks ties by highest severity (degree heuristic proxy): assigning the
    most constrained, most critical victim first reduces wasted search.
    """
    return min(
        unassigned,
        key=lambda vid: (len(domains[vid]), -SEVERITY.get('minor', 1))
    )


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE BACKTRACKING (no heuristics)
# ─────────────────────────────────────────────────────────────────────────────

def csp_plain_backtracking(victims: list, resources: dict) -> tuple[dict, int]:
    """
    Plain backtracking CSP solver — no MRV, no forward checking.
    Used as a performance baseline to measure backtrack count without heuristics.

    Returns
    -------
    (assignment, backtrack_count)
    """
    n_ambulances = resources['ambulances']
    assignment   = {}
    bt_count     = [0]

    def backtrack(idx: int) -> bool:
        if idx == len(victims):
            return True
        v = victims[idx]
        for amb in range(n_ambulances):
            if all_constraints_satisfied(assignment, v['id'], amb, resources):
                assignment[v['id']] = amb
                if backtrack(idx + 1):
                    return True
                del assignment[v['id']]
                bt_count[0] += 1
        return False

    success = backtrack(0)
    if not success:
        log("CSP plain backtracking: no valid assignment found")
    return assignment, bt_count[0]


# ─────────────────────────────────────────────────────────────────────────────
# MRV + FORWARD CHECKING BACKTRACKING (full solver)
# ─────────────────────────────────────────────────────────────────────────────

def csp_mrv_solver(victims: list, resources: dict) -> tuple[dict, int, bool]:
    """
    Backtracking CSP solver with MRV variable ordering and forward checking.

    Algorithm
    ---------
    1.  Initialise domains: every unassigned victim can go to any ambulance.
    2.  Pick the variable (victim) with fewest remaining domain values (MRV).
    3.  For each value (ambulance) in that variable's domain:
          a. Check constraint satisfaction.
          b. Apply forward checking — prune sibling domains.
          c. Recurse.  On failure, restore pruned domains and backtrack.

    The MRV heuristic surfaces the most constrained variable first, failing
    early when the current partial assignment is infeasible, thus reducing
    the total number of backtrack steps compared to arbitrary ordering.

    Returns
    -------
    (assignment, backtrack_count, success)
    """
    n_ambulances = resources['ambulances']
    assignment   = {}
    bt_count     = [0]

    # Initial domains: all ambulances available
    initial_domains = {
        v['id']: list(range(n_ambulances))
        for v in victims
    }

    # Sort victims by severity descending (degree heuristic seed)
    sorted_victims = sorted(victims, key=lambda v: -SEVERITY.get(v['severity'], 1))

    def backtrack(domains: dict) -> bool:
        unassigned = [v['id'] for v in sorted_victims if v['id'] not in assignment]
        if not unassigned:
            return True                        # all assigned → solution found

        vid = mrv_select(unassigned, domains)  # MRV variable selection

        for amb in domains[vid]:
            if not all_constraints_satisfied(assignment, vid, amb, resources):
                continue

            # Tentatively assign
            assignment[vid] = amb

            # Forward check
            pruned = forward_check(domains, assignment, resources)
            if pruned is not None:
                if backtrack(pruned):
                    return True

            # Undo
            del assignment[vid]
            bt_count[0] += 1

        return False

    success = backtrack(initial_domains)
    return assignment, bt_count[0], success


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def allocate_resources(victims: list, resources: dict) -> dict:
    """
    Run the full CSP pipeline:
      1. Solve with MRV + forward checking.
      2. Solve with plain backtracking (baseline comparison).
      3. Log both results.

    Returns
    -------
    {
      'assignment':      {victim_id: ambulance_id, ...},
      'bt_heuristic':    int,   # backtracks with MRV + FC
      'bt_no_heuristic': int,   # backtracks without heuristics
      'success':         bool,
      'ambulance_loads': {0: [victim_ids], 1: [victim_ids]},
    }
    """
    log("CSP: starting resource allocation")

    # ── Full solver (MRV + FC) ──────────────────────────────────────────────
    assignment_h, bt_h, success = csp_mrv_solver(victims, resources)

    # ── Baseline (plain backtracking) ───────────────────────────────────────
    assignment_plain, bt_plain = csp_plain_backtracking(victims, resources)

    log(f"CSP (MRV + FC)        → assignment={assignment_h}, backtracks={bt_h}")
    log(f"CSP (plain BT)        → assignment={assignment_plain}, backtracks={bt_plain}")

    # Build ambulance load summary
    n_amb = resources['ambulances']
    loads = {a: [] for a in range(n_amb)}
    for vid, amb in assignment_h.items():
        loads[amb].append(vid)

    for amb, vids in loads.items():
        log(f"  Ambulance {amb} → victims {vids} "
            f"(load {len(vids)}/{2})")

    return {
        'assignment':       assignment_h,
        'bt_heuristic':     bt_h,
        'bt_no_heuristic':  bt_plain,
        'success':          success,
        'ambulance_loads':  loads,
    }


def get_ambulance(victim_id: int, csp_result: dict) -> int | None:
    """Return the ambulance assigned to *victim_id*, or None."""
    return csp_result['assignment'].get(victim_id)