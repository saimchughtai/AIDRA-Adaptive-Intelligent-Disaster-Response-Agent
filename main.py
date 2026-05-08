"""
main.py
=======
AIDRA – Adaptive Intelligent Disaster Response Agent
AIC-201 CCP | Entry Point

Usage
-----
  python main.py                   # full run: simulation + figures + report
  python main.py --no-figures      # skip figure generation
  python main.py --export-log      # also save the decision log to a .txt file

Project File Structure
----------------------
  main.py              ← you are here
  environment.py       ← grid, victims, resources, events
  search_algorithms.py ← BFS, DFS, Greedy BFS, A*, route utilities
  local_search.py      ← Simulated Annealing + Hill Climbing
  csp_allocation.py    ← CSP resource allocator (MRV + forward checking)
  ml_models.py         ← kNN, Naive Bayes, MLP training + evaluation
  fuzzy_logic.py       ← Mamdani fuzzy priority scoring
  controller.py        ← simulation loop + KPI computation
  visualizer.py        ← all matplotlib charts (7 figures)
  figures/             ← generated PDF figures
  logs/                ← exported JSON results + decision log
"""

import sys
import os
import time

# ── Ensure project root is on path when running from another directory ───────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import run_simulation, export_results
from visualizer import generate_all_figures

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║       AIDRA – Adaptive Intelligent Disaster Response Agent       ║
║       AIC-201 Complex Computing Problem  |  Bahria University    ║
╚══════════════════════════════════════════════════════════════════╝
"""


def print_kpi_table(kpis: dict) -> None:
    """Pretty-print the KPI table to stdout."""
    print("\n" + "─" * 52)
    print(f"  {'KPI':<38} {'Value':>10}")
    print("─" * 52)
    rows = [
        ("Victims Saved",                  f"{kpis['victims_saved']} / 5"),
        ("Average Rescue Time (steps)",    f"{kpis['avg_rescue_time']:.2f}"),
        ("Path Optimality Ratio (A*/BFS)", f"{kpis['path_optimality_ratio']:.4f}"),
        ("Resource Utilisation Rate",      f"{kpis['resource_utilization']:.2f}"),
        ("Risk Exposure Score",            f"{kpis['risk_exposure_score']}"),
        ("CSP Backtracks (MRV + FC)",      f"{kpis['csp_bt_heuristic']}"),
        ("CSP Backtracks (plain BT)",      f"{kpis['csp_bt_no_heuristic']}"),
        ("Best ML F1-Score",               f"{kpis['ml_best_f1']:.4f}"),
    ]
    for label, value in rows:
        print(f"  {label:<38} {value:>10}")
    print("─" * 52 + "\n")


def print_ml_table(ml_results: dict) -> None:
    """Pretty-print ML results."""
    print("  ML Model Performance")
    print("─" * 62)
    print(f"  {'Model':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("─" * 62)
    for name, r in ml_results.items():
        print(f"  {name:<20} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['f1']:>7.4f}")
    print("─" * 62 + "\n")


def print_priority_order(results: dict) -> None:
    """Show the final victim priority order with justification."""
    print("  Victim Rescue Priority Order")
    print("─" * 52)
    for rank, vid in enumerate(results['priority_order'], 1):
        v = next(x for x in results['victims'] if x['id'] == vid)
        sp = v.get('survival_prob', '—')
        sp_str = f"{sp:.3f}" if isinstance(sp, float) else str(sp)
        rt = v.get('rescue_time', '—')
        print(f"  {rank}. V{vid} ({v['severity']:<8}) "
              f"survival_prob={sp_str}  rescue_time={rt}")
    print("─" * 52 + "\n")


def print_csp_summary(csp: dict) -> None:
    """Show CSP assignment and backtrack counts."""
    print("  CSP Resource Allocation")
    print("─" * 52)
    for amb, vids in csp['ambulance_loads'].items():
        print(f"  Ambulance {amb}: victims {vids}  (load {len(vids)}/2)")
    print(f"  Backtracks – MRV + FC  : {csp['bt_heuristic']}")
    print(f"  Backtracks – Plain BT  : {csp['bt_no_heuristic']}")
    print("─" * 52 + "\n")


def print_local_search(ls: dict) -> None:
    """Show local-search comparison results."""
    print("  Local Search – Rescue Ordering Optimisation")
    print("─" * 52)
    print(f"  SA order: {ls['SA']['order']}  cost={ls['SA']['cost']}")
    print(f"  HC order: {ls['HC']['order']}  cost={ls['HC']['cost']}")
    print(f"  Winner  : {ls['winner']}")
    print("─" * 52 + "\n")


def export_decision_log(log_lines: list, filepath: str = 'logs/decision_log.txt') -> None:
    """Save the full decision log to a text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("AIDRA Decision Log\n")
        f.write("=" * 70 + "\n\n")
        for line in log_lines:
            f.write(line + "\n")
    print(f"[MAIN] Decision log exported → {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(BANNER)

    # Parse simple CLI flags
    gen_figures  = '--no-figures'  not in sys.argv
    export_log   = '--export-log'  in sys.argv

    # ── Run simulation ───────────────────────────────────────────────────────
    print("► Running AIDRA simulation…\n")
    t0      = time.perf_counter()
    results = run_simulation()
    elapsed = time.perf_counter() - t0
    print(f"\n► Simulation complete in {elapsed:.2f}s")

    # ── Console summaries ────────────────────────────────────────────────────
    print_priority_order(results)
    print_kpi_table(results['kpis'])
    print_ml_table(results['ml_results'])
    print_csp_summary(results['csp'])
    print_local_search(results['local_search'])

    # ── Export JSON results ──────────────────────────────────────────────────
    export_results(results)

    # ── Export decision log ──────────────────────────────────────────────────
    if export_log:
        export_decision_log(results['log'])

    # ── Generate figures ─────────────────────────────────────────────────────
    if gen_figures:
        print("► Generating figures…")
        saved = generate_all_figures(results)
        print(f"► {len(saved)} figures saved to figures/\n")
        for p in saved:
            print(f"   {p}")
    else:
        print("► Figure generation skipped (--no-figures)")

    print("\n✔  All done.  Check figures/ and logs/ for outputs.\n")


if __name__ == '__main__':
    main()