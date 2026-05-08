"""
controller.py
=============
AIDRA – Adaptive Intelligent Disaster Response Agent
AIC-201 CCP | Module 8: Decision & Replanning Controller

The Controller is the brain of AIDRA.  It orchestrates all five modules:

  1. Environment  – provides initial state and dynamic events
  2. Search       – plans and replans routes via A*
  3. CSP          – allocates ambulances and kits
  4. ML           – estimates victim survival probabilities
  5. Fuzzy Logic  – converts multi-attribute data to priority scores
  6. Local Search – optimises global victim ordering (SA/HC)

Simulation Loop
---------------
  Phase 0 : Train ML models, run CSP, compute fuzzy priorities, run SA/HC.
  Phase 1 : For each rescue step (victim in priority order):
              a. Apply any scheduled environmental events.
              b. Replan A* route on updated grid if events occurred.
              c. Select route (risk-aware vs speed-prioritised).
              d. Predict survival via ML; escalate if needed.
              e. Execute rescue, update victim state, log decisions.
  Phase 2 : Compute KPIs and return full results dict.

Conflicting Objective Resolution
---------------------------------
  Objective 1 (time vs risk)    : resolved by route selection logic in
                                   search_algorithms.select_best_route().
  Objective 2 (priority vs throughput) : resolved by fuzzy ranking +
                                          ML escalation in fuzzy_logic.
"""

import copy
import json
from environment       import (fresh_state, BASE_MAP, RESCUE_BASE,
                               MEDICAL_CENTERS, apply_events, log, clear_log,
                               DECISION_LOG)
from search_algorithms import (compare_all_algorithms, select_best_route,
                               path_risk_cells, path_weighted_cost)
from local_search      import compare_local_search
from csp_allocation    import allocate_resources
from ML_models         import run_ml_pipeline, predict_survival, classify_risk
from fuzzy_logic       import rank_victims, escalate_critical


# ─────────────────────────────────────────────────────────────────────────────
# KPI COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_kpis(victims: list, search_results: dict,
                 ml_results: dict, resources: dict,
                 victims_saved: int, avg_rescue_time: float,
                 bt_h: int, bt_no: int) -> dict:
    """
    Compute all required Key Performance Indicators.

    KPI 1  – Victims Saved (#)
    KPI 2  – Average Rescue Time (steps)
    KPI 3  – Path Optimality Ratio  (A* cost / BFS cost, averaged)
    KPI 4  – Resource Utilisation Rate
    KPI 5  – Risk Exposure Score (total risk cells traversed)
    KPI 6  – CSP Backtrack Counts (with / without MRV heuristic)
    KPI 7  – ML Metrics (accuracy, precision, recall, F1 per model)
    """
    # Path optimality (A* vs BFS)
    ratios = []
    for vid, routes in search_results.items():
        astar_cost = routes.get('A*', {}).get('cost', 0)
        bfs_cost   = routes.get('BFS', {}).get('cost', 1)
        if astar_cost and bfs_cost:
            ratios.append(astar_cost / bfs_cost)

    opt_ratio    = round(sum(ratios) / len(ratios), 4) if ratios else 1.0
    total_risk   = sum(
        search_results[vid].get('A*', {}).get('risk', 0)
        for vid in search_results
    )
    util_rate    = round(resources['ambulances'] / 2, 2)
    best_f1      = max(v['f1'] for v in ml_results.values()) if ml_results else 0

    return {
        'victims_saved':          victims_saved,
        'avg_rescue_time':        round(avg_rescue_time, 2),
        'path_optimality_ratio':  opt_ratio,
        'resource_utilization':   util_rate,
        'risk_exposure_score':    total_risk,
        'csp_bt_heuristic':       bt_h,
        'csp_bt_no_heuristic':    bt_no,
        'ml_best_f1':             best_f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation() -> dict:
    """
    Execute the full AIDRA simulation and return a results dictionary.

    Returns
    -------
    {
      'victims'        : list of victim dicts (with rescue outcomes),
      'grid'           : final grid state,
      'search_results' : {victim_id: {algorithm: metrics}},
      'ml_results'     : {model_name: {accuracy, precision, recall, f1, cm}},
      'csp'            : {assignment, bt_heuristic, bt_no_heuristic, ...},
      'local_search'   : {SA: {order, cost}, HC: {order, cost}, winner},
      'fuzzy'          : {victim_id: priority_score},
      'priority_order' : [victim_ids in rescue sequence],
      'kpis'           : {kpi_name: value},
      'log'            : [decision log strings],
    }
    """
    clear_log()
    log("═" * 60)
    log("AIDRA Simulation Start")
    log("═" * 60)

    # ── Phase 0-A: Initialise state ─────────────────────────────────────────
    grid, victims, resources = fresh_state()
    current_grid = copy.deepcopy(grid)

    # ── Phase 0-B: Train ML models ──────────────────────────────────────────
    log("\n[PHASE 0] Initialisation")
    best_model, best_model_name, ml_results, all_models = run_ml_pipeline()
    log(f"Active ML model: {best_model_name}")

    # ── Phase 0-C: CSP resource allocation ──────────────────────────────────
    csp_result = allocate_resources(victims, resources)

    # ── Phase 0-D: Compute route stats for all victims (initial grid) ────────
    search_results = {}
    for v in victims:
        routes = compare_all_algorithms(RESCUE_BASE, v['pos'], current_grid)
        search_results[v['id']] = routes

    # ── Phase 0-E: Fuzzy priority ranking ────────────────────────────────────
    from search_algorithms import astar as _astar
    ranked_victims, fuzzy_scores = rank_victims(victims, current_grid,
                                                search_fn=_astar)

    # ── Phase 0-F: ML escalation (victims with survival < 0.5 move front) ───
    priority_order = escalate_critical(ranked_victims, best_model,
                                       predict_survival, current_grid)
    log(f"Final priority order: {[v['id'] for v in priority_order]}")

    # ── Phase 0-G: Local search comparison ──────────────────────────────────
    local_search_results = compare_local_search(victims, current_grid)

    # ── Phase 1: Rescue loop ─────────────────────────────────────────────────
    log("\n[PHASE 1] Rescue Execution")
    total_rescue_time = 0
    victims_saved     = 0
    replanning_count  = 0

    for step, v in enumerate(priority_order):
        log(f"\n── Step {step}: attempting rescue of V{v['id']} "
            f"({v['severity']}) at {v['pos']}")

        # Apply dynamic environmental events
        updated_grid, events = apply_events(current_grid, step)

        if events:
            replanning_count += 1
            for ev in events:
                log(f"  ⚠ EVENT: {ev}")
            current_grid = updated_grid
            log(f"  Replanning triggered (event #{replanning_count})")

            # Replan route on updated grid
            routes = compare_all_algorithms(RESCUE_BASE, v['pos'], current_grid)
            search_results[v['id']] = routes
            log(f"  Route replanned for V{v['id']}")

        # Select best route (risk-aware decision)
        chosen_path, chosen_alg, justification = select_best_route(
            search_results[v['id']]
        )
        log(f"  Route: {chosen_alg} | {justification}")

        if chosen_path is None:
            log(f"  ✗ V{v['id']} UNREACHABLE after replanning — skipping")
            continue

        # Compute path metrics
        path_len  = len(chosen_path) - 1
        risk_exp  = path_risk_cells(chosen_path, current_grid)
        wait_est  = step * 2

        # ML survival prediction
        survival_prob = predict_survival(
            best_model, v['severity'], path_len, risk_exp, wait_est, kit_available=1
        )
        risk_cat      = classify_risk(survival_prob)

        # Update victim record
        v['rescue_time']    = path_len + wait_est
        v['risk_score']     = round(1 - survival_prob, 4)
        v['survival_prob']  = survival_prob
        v['assigned_ambulance'] = csp_result['assignment'].get(v['id'], 0)
        v['rescued']        = True

        total_rescue_time += v['rescue_time']
        victims_saved     += 1

        log(f"  ✓ V{v['id']} rescued | path_len={path_len} | "
            f"risk_cells={risk_exp} | wait={wait_est} | "
            f"survival_prob={survival_prob} ({risk_cat} risk) | "
            f"rescue_time={v['rescue_time']}")

    # ── Phase 2: KPI computation ─────────────────────────────────────────────
    log("\n[PHASE 2] KPI Computation")
    avg_rt = total_rescue_time / max(victims_saved, 1)
    kpis   = compute_kpis(
        victims, search_results, ml_results, resources,
        victims_saved, avg_rt,
        csp_result['bt_heuristic'],
        csp_result['bt_no_heuristic'],
    )

    log("\n═" * 30)
    log(f"Victims saved     : {victims_saved}/5")
    log(f"Avg rescue time   : {avg_rt:.1f} steps")
    log(f"Replanning events : {replanning_count}")
    log(f"Path optimality   : {kpis['path_optimality_ratio']}")
    log(f"Best ML F1        : {kpis['ml_best_f1']}")
    log("═" * 60)

    return {
        'victims':         victims,
        'grid':            current_grid,
        'initial_grid':    grid,
        'search_results':  search_results,
        'ml_results':      ml_results,
        'all_ml_models':   all_models,
        'csp':             csp_result,
        'local_search':    local_search_results,
        'fuzzy':           fuzzy_scores,
        'priority_order':  [v['id'] for v in priority_order],
        'kpis':            kpis,
        'log':             list(DECISION_LOG),
        'replanning_count': replanning_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def export_results(results: dict, filepath: str = 'logs/simulation_results.json') -> None:
    """
    Save a JSON-serialisable copy of the results to *filepath*.
    Removes non-serialisable model objects before saving.
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    safe = copy.deepcopy(results)
    safe.pop('all_ml_models', None)   # sklearn objects → not JSON-serialisable

    # Convert tuple keys to strings for JSON
    new_search = {}
    for vid, routes in safe.get('search_results', {}).items():
        new_routes = {}
        for alg, data in routes.items():
            d2 = dict(data)
            d2.pop('path', None)       # path lists can be huge
            new_routes[alg] = d2
        new_search[str(vid)] = new_routes
    safe['search_results'] = new_search

    # Convert fuzzy scores keys to strings
    safe['fuzzy'] = {str(k): v for k, v in safe.get('fuzzy', {}).items()}

    with open(filepath, 'w') as f:
        json.dump(safe, f, indent=2, default=str)
    print(f"[CTRL] Results exported → {filepath}")