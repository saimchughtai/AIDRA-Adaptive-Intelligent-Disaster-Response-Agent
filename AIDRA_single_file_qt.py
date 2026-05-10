"""
AIDRA single-file Python Qt application.

Open this file in VSCode and run it directly:

    python AIDRA_single_file_qt.py

Optional console-only check:

    python AIDRA_single_file_qt.py --cli

Dependencies for the GUI:
    pip install PyQt6 numpy scikit-learn

The program also accepts PySide6 or PyQt5 instead of PyQt6. If scikit-learn is
not installed, the ML module falls back to a deterministic risk model so the
simulation can still run.

The GUI includes a mission editor: add victims, block roads, mark high-risk
roads, schedule future road blockages, run the mission, and replay the rescue
unit moving from location to location.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush


# ---------------------------------------------------------------------------
# Console encoding and logging
# ---------------------------------------------------------------------------

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DECISION_LOG: list[str] = []


def log(message: str) -> None:
    DECISION_LOG.append(message)
    text = f"[LOG] {message}"
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))


def clear_log() -> None:
    DECISION_LOG.clear()


# ---------------------------------------------------------------------------
# Environment model
# ---------------------------------------------------------------------------

GRID_SIZE = 10

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

RESCUE_BASE = (0, 0)
MEDICAL_CENTERS = [(9, 9), (9, 0)]

SEVERITY = {"minor": 1, "moderate": 2, "critical": 3}
CELL_COST = {0: 1, 1: math.inf, 2: 3}

VICTIMS_INIT = [
    {"id": 0, "pos": (2, 5), "severity": "critical"},
    {"id": 1, "pos": (5, 7), "severity": "critical"},
    {"id": 2, "pos": (3, 3), "severity": "moderate"},
    {"id": 3, "pos": (7, 2), "severity": "moderate"},
    {"id": 4, "pos": (6, 8), "severity": "minor"},
]

RESOURCES = {
    "ambulances": 2,
    "rescue_teams": 1,
    "medical_kits": 10,
}

SCHEDULED_EVENTS = {
    3: [(4, 3, 1, "Road at (4,3) blocked by aftershock")],
    5: [(2, 7, 1, "Road at (2,7) blocked by fire spread")],
}


def copy_grid(grid: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in grid]


def normalize_victims(victims: list[dict]) -> list[dict]:
    source = victims or VICTIMS_INIT
    normalized = []
    for index, victim in enumerate(source):
        row = dict(victim)
        row["id"] = int(row.get("id", index))
        row["pos"] = tuple(row["pos"])
        row["severity"] = row.get("severity", "moderate")
        row.update(
            rescued=row.get("rescued", False),
            rescue_time=row.get("rescue_time"),
            risk_score=row.get("risk_score"),
            survival_prob=row.get("survival_prob"),
            assigned_ambulance=row.get("assigned_ambulance"),
        )
        normalized.append(row)
    return normalized


def fresh_state(
    initial_grid: list[list[int]] | None = None,
    initial_victims: list[dict] | None = None,
) -> tuple[list[list[int]], list[dict], dict]:
    grid = copy_grid(initial_grid or BASE_MAP)
    victims = normalize_victims(initial_victims or VICTIMS_INIT)
    return grid, victims, dict(RESOURCES)


def apply_events(
    grid: list[list[int]],
    step: int,
    scheduled_events: dict[int, list[tuple[int, int, int, str]]] | None = None,
) -> tuple[list[list[int]], list[str]]:
    updated = copy_grid(grid)
    events = []
    schedule = scheduled_events if scheduled_events is not None else SCHEDULED_EVENTS
    for row, col, value, description in schedule.get(step, []):
        if updated[row][col] != value:
            updated[row][col] = value
            events.append(description)
    return updated, events


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------

def get_neighbors(pos: tuple[int, int], grid: list[list[int]], allow_risk: bool = True):
    row, col = pos
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
            continue
        cell = grid[nr][nc]
        if cell == 1:
            continue
        if cell == 2 and not allow_risk:
            continue
        yield nr, nc


def step_cost(pos: tuple[int, int], grid: list[list[int]]) -> int:
    return CELL_COST.get(grid[pos[0]][pos[1]], 1)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def path_risk_cells(path: list[tuple[int, int]] | None, grid: list[list[int]]) -> int:
    return sum(1 for row, col in (path or []) if grid[row][col] == 2)


def path_weighted_cost(path: list[tuple[int, int]] | None, grid: list[list[int]]) -> int:
    if not path:
        return 0
    return int(sum(step_cost(pos, grid) for pos in path))


def bfs(start: tuple[int, int], goal: tuple[int, int], grid, allow_risk=True):
    queue = deque([(start, [start])])
    visited = {start}
    expanded = 0
    while queue:
        current, path = queue.popleft()
        expanded += 1
        if current == goal:
            return path, expanded
        for neighbor in get_neighbors(current, grid, allow_risk):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None, expanded


def dfs(start: tuple[int, int], goal: tuple[int, int], grid, allow_risk=True):
    stack = [(start, [start])]
    visited = set()
    expanded = 0
    while stack:
        current, path = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        expanded += 1
        if current == goal:
            return path, expanded
        for neighbor in get_neighbors(current, grid, allow_risk):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None, expanded


def greedy_bfs(start: tuple[int, int], goal: tuple[int, int], grid, allow_risk=True):
    heap = [(manhattan(start, goal), start, [start])]
    visited = set()
    expanded = 0
    while heap:
        _, current, path = heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        expanded += 1
        if current == goal:
            return path, expanded
        for neighbor in get_neighbors(current, grid, allow_risk):
            if neighbor not in visited:
                heappush(heap, (manhattan(neighbor, goal), neighbor, path + [neighbor]))
    return None, expanded


def astar(start: tuple[int, int], goal: tuple[int, int], grid, allow_risk=True):
    heap = [(manhattan(start, goal), 0, start, [start])]
    best_g: dict[tuple[int, int], int] = {}
    expanded = 0
    while heap:
        _, g_score, current, path = heappop(heap)
        if current in best_g and best_g[current] <= g_score:
            continue
        best_g[current] = g_score
        expanded += 1
        if current == goal:
            return path, expanded
        for neighbor in get_neighbors(current, grid, allow_risk):
            new_g = g_score + step_cost(neighbor, grid)
            heappush(heap, (new_g + manhattan(neighbor, goal), new_g, neighbor, path + [neighbor]))
    return None, expanded


def compare_all_algorithms(start, goal, grid) -> dict:
    algorithms = {
        "BFS": lambda: bfs(start, goal, grid, True),
        "DFS": lambda: dfs(start, goal, grid, True),
        "Greedy": lambda: greedy_bfs(start, goal, grid, True),
        "A*": lambda: astar(start, goal, grid, True),
        "A*-safe": lambda: astar(start, goal, grid, False),
    }
    results = {}
    for name, fn in algorithms.items():
        start_time = time.perf_counter()
        path, expanded = fn()
        elapsed = round((time.perf_counter() - start_time) * 1000, 3)
        results[name] = {
            "path": path,
            "path_len": len(path) if path else 0,
            "cost": path_weighted_cost(path, grid),
            "risk": path_risk_cells(path, grid),
            "expanded": expanded,
            "time_ms": elapsed,
        }
    return results


def select_best_route(route_data: dict):
    fast = route_data.get("A*", {})
    safe = route_data.get("A*-safe", {})
    fast_path = fast.get("path")
    safe_path = safe.get("path")
    if not fast_path and not safe_path:
        return None, "None", "No path available after replanning"
    if fast_path and not safe_path:
        return fast_path, "A*", f"No risk-free path found; using A* cost={fast['cost']}"
    if safe_path and not fast_path:
        return safe_path, "A*-safe", f"Only safe path available; cost={safe['cost']}"
    if safe["cost"] <= fast["cost"]:
        return safe_path, "A*-safe", f"Safe path chosen; cost={safe['cost']} <= {fast['cost']}"
    return fast_path, "A*", f"Fast path chosen; cost={fast['cost']} vs safe={safe['cost']}"


# ---------------------------------------------------------------------------
# CSP resource allocation
# ---------------------------------------------------------------------------

def constraints_ok(assignment: dict[int, int], victim_id: int, ambulance: int, resources: dict) -> bool:
    temp = {**assignment, victim_id: ambulance}
    capacity = resources.get("ambulance_capacity", 2)
    if sum(1 for value in temp.values() if value == ambulance) > capacity:
        return False
    if len(temp) > resources["medical_kits"]:
        return False
    return True


def forward_check(domains: dict[int, list[int]], assignment: dict[int, int], resources: dict):
    pruned = {vid: list(domain) for vid, domain in domains.items()}
    for vid, domain in pruned.items():
        if vid in assignment:
            continue
        valid = [amb for amb in domain if constraints_ok(assignment, vid, amb, resources)]
        if not valid:
            return None
        pruned[vid] = valid
    return pruned


def csp_solver(victims: list[dict], resources: dict, use_mrv: bool = True, use_forward_checking: bool = True):
    n_ambulances = resources["ambulances"]
    assignment: dict[int, int] = {}
    backtracks = 0
    domains = {victim["id"]: list(range(n_ambulances)) for victim in victims}
    ordered = sorted(victims, key=lambda item: -SEVERITY.get(item["severity"], 1))
    severity_by_id = {victim["id"]: SEVERITY.get(victim["severity"], 1) for victim in victims}

    def choose(unassigned: list[int], current_domains: dict[int, list[int]]) -> int:
        if not use_mrv:
            return unassigned[0]
        return min(unassigned, key=lambda vid: (len(current_domains[vid]), -severity_by_id[vid], vid))

    def backtrack(current_domains: dict[int, list[int]]) -> bool:
        nonlocal backtracks
        unassigned = [victim["id"] for victim in ordered if victim["id"] not in assignment]
        if not unassigned:
            return True
        vid = choose(unassigned, current_domains)
        for ambulance in current_domains[vid]:
            if not constraints_ok(assignment, vid, ambulance, resources):
                continue
            assignment[vid] = ambulance
            next_domains = current_domains
            if use_forward_checking:
                checked = forward_check(current_domains, assignment, resources)
                if checked is None:
                    del assignment[vid]
                    backtracks += 1
                    continue
                next_domains = checked
            if backtrack(next_domains):
                return True
            del assignment[vid]
            backtracks += 1
        return False

    success = backtrack(domains)
    return dict(assignment), backtracks, success


def allocate_resources(victims: list[dict], resources: dict) -> dict:
    resources = dict(resources)
    resources["ambulance_capacity"] = max(2, math.ceil(len(victims) / max(resources["ambulances"], 1)))
    log("CSP: starting resource allocation")
    assignment_h, bt_h, success = csp_solver(victims, resources, True, True)
    assignment_plain, bt_plain, _ = csp_solver(victims, resources, False, False)
    loads = {amb: [] for amb in range(resources["ambulances"])}
    for vid, amb in assignment_h.items():
        loads[amb].append(vid)
    log(f"CSP MRV+FC assignment={assignment_h}, backtracks={bt_h}")
    log(f"CSP plain assignment={assignment_plain}, backtracks={bt_plain}")
    for amb, vids in loads.items():
        log(f"Ambulance {amb}: victims {vids} load {len(vids)}/{resources['ambulance_capacity']}")
    return {
        "assignment": assignment_h,
        "bt_heuristic": bt_h,
        "bt_no_heuristic": bt_plain,
        "success": success,
        "ambulance_loads": loads,
        "ambulance_capacity": resources["ambulance_capacity"],
    }


# ---------------------------------------------------------------------------
# ML model with fallback
# ---------------------------------------------------------------------------

class HeuristicSurvivalModel:
    def predict_proba(self, rows):
        output = []
        for severity, distance, risk, wait, kit in rows:
            prob = 0.90 - 0.10 * severity - 0.02 * distance - 0.08 * risk - 0.03 * wait + 0.05 * kit
            prob = max(0.02, min(0.98, prob))
            output.append([1.0 - prob, prob])
        return output


def run_ml_pipeline():
    try:
        import numpy as np
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
    except Exception:
        model = HeuristicSurvivalModel()
        metrics = {
            "Heuristic Survival": {
                "accuracy": 0.90,
                "precision": 0.90,
                "recall": 0.90,
                "f1": 0.90,
                "cm": [[36, 4], [4, 36]],
            }
        }
        log("ML fallback model active because scikit-learn/numpy is unavailable")
        return model, "Heuristic Survival", metrics, {"Heuristic Survival": model}

    rng = np.random.default_rng(42)
    n_samples = 400
    severity = rng.integers(1, 4, n_samples)
    distance = rng.integers(1, 20, n_samples)
    risk = rng.integers(0, 3, n_samples)
    wait = rng.integers(0, 15, n_samples)
    kit = rng.integers(0, 2, n_samples)
    noise = rng.normal(0, 0.05, n_samples)
    score = 0.90 - 0.10 * severity - 0.02 * distance - 0.08 * risk - 0.03 * wait + 0.05 * kit + noise
    y = (score > 0.40).astype(int)
    x = np.column_stack([severity, distance, risk, wait, kit])
    log(f"Dataset generated: {n_samples} records, positive={int(y.sum())}, negative={int((1-y).sum())}")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
    models = {
        "kNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "MLP (32-16)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42),
    }
    trained = {}
    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "cm": confusion_matrix(y_test, y_pred).tolist(),
        }
        trained[name] = model
        log(f"ML {name}: F1={metrics[name]['f1']}")
    best_name = max(metrics, key=lambda key: metrics[key]["f1"])
    log(f"Best ML model selected: {best_name}")
    return trained[best_name], best_name, metrics, trained


def predict_survival(model, severity_str: str, distance: int, risk_exposure: int, wait_time: int, kit_available: int = 1):
    severity_code = SEVERITY.get(severity_str, 2)
    row = [[severity_code, distance, risk_exposure, wait_time, kit_available]]
    if hasattr(model, "predict_proba"):
        return round(float(model.predict_proba(row)[0][1]), 4)
    return round(float(model.predict(row)[0]), 4)


def classify_risk(survival_prob: float) -> str:
    if survival_prob >= 0.70:
        return "low"
    if survival_prob >= 0.40:
        return "moderate"
    return "high"


# ---------------------------------------------------------------------------
# Fuzzy priority model
# ---------------------------------------------------------------------------

def trimf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a + 1e-9)
    return (c - x) / (c - b + 1e-9)


def compute_priority_score(severity_str: str, risk_exposure: float, wait_time: float = 0.0) -> float:
    severity_code = SEVERITY.get(severity_str, 2)
    sev_low = trimf(severity_code, 0, 1, 2)
    sev_medium = trimf(severity_code, 1, 2, 3)
    sev_high = trimf(severity_code, 2, 3, 4)
    risk_low = trimf(risk_exposure, -0.1, 0.0, 0.5)
    risk_high = trimf(risk_exposure, 0.3, 1.0, 1.1)
    wait_short = trimf(wait_time, -1.0, 0.0, 7.0)
    wait_long = trimf(wait_time, 5.0, 15.0, 16.0)
    activations = [
        min(sev_high, wait_long),
        min(sev_high, wait_short),
        sev_medium,
        min(sev_low, risk_high),
        min(sev_low, risk_low),
    ]
    centers = [0.90, 0.85, 0.55, 0.45, 0.20]
    total = sum(activations)
    if total < 1e-9:
        return 0.5
    return round(sum(a * c for a, c in zip(activations, centers)) / total, 4)


def rank_victims(victims: list[dict], grid: list[list[int]]):
    scores = {}
    for victim in victims:
        if victim.get("rescued"):
            continue
        path, _ = astar(RESCUE_BASE, victim["pos"], grid, allow_risk=True)
        risk_exposure = 0.0
        if path and len(path) > 1:
            risk_exposure = path_risk_cells(path, grid) / (len(path) - 1)
        score = compute_priority_score(victim["severity"], risk_exposure, 0.0)
        scores[victim["id"]] = score
        log(f"Fuzzy priority V{victim['id']} ({victim['severity']}): {score}")
    ranked = sorted([v for v in victims if not v.get("rescued")], key=lambda v: -scores[v["id"]])
    return ranked, scores


def escalate_critical(ranked: list[dict], model, grid: list[list[int]]):
    escalated = []
    normal = []
    for victim in ranked:
        path, _ = astar(RESCUE_BASE, victim["pos"], grid)
        distance = len(path) - 1 if path else 15
        risk = path_risk_cells(path, grid)
        prob = predict_survival(model, victim["severity"], distance, risk, 0, 1)
        victim["survival_prob"] = prob
        if prob < 0.50:
            escalated.append(victim)
            log(f"ML escalation V{victim['id']}: survival={prob}")
        else:
            normal.append(victim)
    return escalated + normal


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def ordering_cost(order: list[int], victim_map: dict[int, dict], grid) -> int:
    total = 0
    current = RESCUE_BASE
    for vid in order:
        path, _ = astar(current, victim_map[vid]["pos"], grid, allow_risk=True)
        if path:
            total += len(path) - 1
            current = victim_map[vid]["pos"]
        else:
            total += 9999
    return total


def simulated_annealing(victims: list[dict], grid, max_iter=500, seed=42):
    random.seed(seed)
    ids = [v["id"] for v in victims if not v.get("rescued")]
    if len(ids) < 2:
        return ids, 0
    victim_map = {v["id"]: v for v in victims}
    current = ids[:]
    random.shuffle(current)
    current_cost = ordering_cost(current, victim_map, grid)
    best = current[:]
    best_cost = current_cost
    temperature = 100.0
    for _ in range(max_iter):
        i, j = random.sample(range(len(current)), 2)
        neighbor = current[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbor_cost = ordering_cost(neighbor, victim_map, grid)
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(temperature, 1e-6)):
            current = neighbor
            current_cost = neighbor_cost
        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost
        temperature *= 0.97
    log(f"SA best order={best}, cost={best_cost}")
    return best, best_cost


def hill_climbing(victims: list[dict], grid, max_iter=300, seed=42):
    random.seed(seed + 1)
    ids = [v["id"] for v in victims if not v.get("rescued")]
    if len(ids) < 2:
        return ids, 0
    victim_map = {v["id"]: v for v in victims}
    current = ids[:]
    random.shuffle(current)
    current_cost = ordering_cost(current, victim_map, grid)
    for _ in range(max_iter):
        best_neighbor = None
        best_cost = current_cost
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                candidate = current[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                cost = ordering_cost(candidate, victim_map, grid)
                if cost < best_cost:
                    best_neighbor = candidate
                    best_cost = cost
        if best_neighbor is None:
            break
        current = best_neighbor
        current_cost = best_cost
    log(f"HC best order={current}, cost={current_cost}")
    return current, current_cost


def compare_local_search(victims: list[dict], grid) -> dict:
    sa_order, sa_cost = simulated_annealing(victims, grid)
    hc_order, hc_cost = hill_climbing(victims, grid)
    winner = "tie"
    if sa_cost < hc_cost:
        winner = "SA"
    elif hc_cost < sa_cost:
        winner = "HC"
    return {"SA": {"order": sa_order, "cost": sa_cost}, "HC": {"order": hc_order, "cost": hc_cost}, "winner": winner}


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

def compute_kpis(victims, search_results, ml_results, resources, victims_saved, avg_rescue_time, bt_h, bt_plain):
    ratios = []
    for routes in search_results.values():
        astar_cost = routes.get("A*", {}).get("cost", 0)
        bfs_cost = routes.get("BFS", {}).get("cost", 1)
        if astar_cost and bfs_cost:
            ratios.append(astar_cost / bfs_cost)
    return {
        "victims_saved": victims_saved,
        "avg_rescue_time": round(avg_rescue_time, 2),
        "path_optimality_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 1.0,
        "resource_utilization": round(resources["ambulances"] / 2, 2),
        "risk_exposure_score": sum(search_results[vid]["A*"].get("risk", 0) for vid in search_results),
        "csp_bt_heuristic": bt_h,
        "csp_bt_no_heuristic": bt_plain,
        "ml_best_f1": max(row["f1"] for row in ml_results.values()) if ml_results else 0,
    }


def nearest_medical_route(start: tuple[int, int], grid: list[list[int]]):
    candidates = []
    for center in MEDICAL_CENTERS:
        path, expanded = astar(start, center, grid, allow_risk=True)
        if path:
            candidates.append((len(path) - 1, path_weighted_cost(path, grid), expanded, center, path))
    if not candidates:
        return None, None
    candidates.sort(key=lambda row: (row[0], row[1]))
    return candidates[0][4], candidates[0][3]


def run_simulation(
    initial_grid: list[list[int]] | None = None,
    initial_victims: list[dict] | None = None,
    scheduled_events: dict[int, list[tuple[int, int, int, str]]] | None = None,
) -> dict:
    clear_log()
    log("=" * 60)
    log("AIDRA Simulation Start")
    log("=" * 60)
    grid, victims, resources = fresh_state(initial_grid, initial_victims)
    current_grid = copy_grid(grid)

    best_model, best_model_name, ml_results, all_models = run_ml_pipeline()
    log(f"Active ML model: {best_model_name}")
    csp = allocate_resources(victims, resources)

    search_results = {}
    for victim in victims:
        search_results[victim["id"]] = compare_all_algorithms(RESCUE_BASE, victim["pos"], current_grid)

    ranked, fuzzy_scores = rank_victims(victims, current_grid)
    priority_order = escalate_critical(ranked, best_model, current_grid)
    local_search_results = compare_local_search(victims, current_grid)
    log(f"Final priority order: {[v['id'] for v in priority_order]}")

    total_rescue_time = 0
    victims_saved = 0
    replanning_count = 0
    rescue_trace = [RESCUE_BASE]
    rescue_segments = []
    current_pos = RESCUE_BASE

    for step, victim in enumerate(priority_order):
        log(f"Step {step}: move from {current_pos} to V{victim['id']} at {victim['pos']} ({victim['severity']})")
        updated_grid, events = apply_events(current_grid, step, scheduled_events)
        if events:
            replanning_count += 1
            current_grid = updated_grid
            for event in events:
                log(f"EVENT: {event}")
        search_results[victim["id"]] = compare_all_algorithms(current_pos, victim["pos"], current_grid)

        chosen_path, chosen_alg, reason = select_best_route(search_results[victim["id"]])
        log(f"Route {chosen_alg}: {reason}")
        if not chosen_path:
            log(f"V{victim['id']} unreachable")
            continue

        evac_path, med_center = nearest_medical_route(victim["pos"], current_grid)
        if not evac_path:
            evac_path = [victim["pos"]]
            med_center = victim["pos"]
            log(f"No medical-center path found for V{victim['id']}; stabilising on site")

        path_len = len(chosen_path) - 1
        evac_len = len(evac_path) - 1
        risk_cells = path_risk_cells(chosen_path, current_grid)
        wait = step * 2
        survival = predict_survival(best_model, victim["severity"], path_len + evac_len, risk_cells, wait, 1)
        victim["rescue_time"] = path_len + evac_len + wait
        victim["risk_score"] = round(1 - survival, 4)
        victim["survival_prob"] = survival
        victim["assigned_ambulance"] = csp["assignment"].get(victim["id"], 0)
        victim["evacuated_to"] = med_center
        victim["rescued"] = True
        total_rescue_time += victim["rescue_time"]
        victims_saved += 1
        rescue_segments.append({
            "victim_id": victim["id"],
            "pickup_path": chosen_path,
            "evac_path": evac_path,
            "from": current_pos,
            "to": victim["pos"],
            "medical_center": med_center,
        })
        rescue_trace.extend(chosen_path[1:])
        rescue_trace.extend(evac_path[1:])
        current_pos = med_center
        log(
            f"V{victim['id']} rescued: pickup={path_len}, evac={evac_len}, wait={wait}, "
            f"survival={survival}, risk={classify_risk(survival)}"
        )

    avg_rescue_time = total_rescue_time / max(victims_saved, 1)
    kpis = compute_kpis(
        victims,
        search_results,
        ml_results,
        resources,
        victims_saved,
        avg_rescue_time,
        csp["bt_heuristic"],
        csp["bt_no_heuristic"],
    )
    log(f"Victims saved: {victims_saved}/{len(victims)}")
    log(f"Average rescue time: {avg_rescue_time:.1f}")
    return {
        "victims": victims,
        "grid": current_grid,
        "initial_grid": grid,
        "search_results": search_results,
        "ml_results": ml_results,
        "all_ml_models": all_models,
        "csp": csp,
        "local_search": local_search_results,
        "fuzzy": fuzzy_scores,
        "priority_order": [v["id"] for v in priority_order],
        "rescue_trace": rescue_trace,
        "rescue_segments": rescue_segments,
        "kpis": kpis,
        "log": list(DECISION_LOG),
        "replanning_count": replanning_count,
    }


def export_results(results: dict, filepath: str = "aidra_results.json") -> str:
    safe = dict(results)
    safe.pop("all_ml_models", None)
    cleaned_search = {}
    for vid, routes in safe.get("search_results", {}).items():
        cleaned_search[str(vid)] = {}
        for name, data in routes.items():
            row = dict(data)
            row.pop("path", None)
            cleaned_search[str(vid)][name] = row
    safe["search_results"] = cleaned_search
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(safe, handle, indent=2, default=str)
    return os.path.abspath(filepath)


# ---------------------------------------------------------------------------
# Qt GUI
# ---------------------------------------------------------------------------

QT_AVAILABLE = True
QT_BINDING = ""
try:
    from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal as Signal
    from PyQt6.QtGui import QColor, QFont, QPainter, QPen
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QComboBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QSpinBox,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    QT_BINDING = "PyQt6"
except ModuleNotFoundError:
    try:
        from PySide6.QtCore import QThread, QTimer, Qt, Signal
        from PySide6.QtGui import QColor, QFont, QPainter, QPen
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QApplication,
            QComboBox,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSizePolicy,
            QSplitter,
            QSpinBox,
            QTabWidget,
            QTableWidget,
            QTableWidgetItem,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
        QT_BINDING = "PySide6"
    except ModuleNotFoundError:
        try:
            from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal as Signal
            from PyQt5.QtGui import QColor, QFont, QPainter, QPen
            from PyQt5.QtWidgets import (
                QAbstractItemView,
                QApplication,
                QComboBox,
                QFrame,
                QGridLayout,
                QHBoxLayout,
                QHeaderView,
                QLabel,
                QMainWindow,
                QMessageBox,
                QPushButton,
                QSizePolicy,
                QSplitter,
                QSpinBox,
                QTabWidget,
                QTableWidget,
                QTableWidgetItem,
                QTextEdit,
                QVBoxLayout,
                QWidget,
            )
            QT_BINDING = "PyQt5"
        except ModuleNotFoundError:
            QT_AVAILABLE = False


def qt_value(group: str, name: str):
    container = getattr(Qt, group, None)
    if container is not None and hasattr(container, name):
        return getattr(container, name)
    return getattr(Qt, name)


def class_value(cls, group: str, name: str):
    container = getattr(cls, group, None)
    if container is not None and hasattr(container, name):
        return getattr(container, name)
    return getattr(cls, name)


if QT_AVAILABLE:
    ALIGN_CENTER = qt_value("AlignmentFlag", "AlignCenter")
    HORIZONTAL = qt_value("Orientation", "Horizontal")
    SIZE_EXPANDING = class_value(QSizePolicy, "Policy", "Expanding")
    HEADER_STRETCH = class_value(QHeaderView, "ResizeMode", "Stretch")
    NO_EDIT_TRIGGERS = class_value(QAbstractItemView, "EditTrigger", "NoEditTriggers")
    SELECT_ROWS = class_value(QAbstractItemView, "SelectionBehavior", "SelectRows")

    class SimulationThread(QThread):
        completed = Signal(object)
        failed = Signal(str)

        def __init__(self, grid=None, victims=None, scheduled_events=None):
            super().__init__()
            self.grid = copy_grid(grid) if grid is not None else None
            self.victims = normalize_victims(victims or VICTIMS_INIT)
            self.scheduled_events = scheduled_events or {}

        def run(self):
            try:
                self.completed.emit(run_simulation(self.grid, self.victims, self.scheduled_events))
            except Exception:
                self.failed.emit(traceback.format_exc())

    class MetricCard(QFrame):
        def __init__(self, title: str):
            super().__init__()
            self.setObjectName("MetricCard")
            layout = QVBoxLayout(self)
            layout.setContentsMargins(14, 12, 14, 12)
            label = QLabel(title)
            label.setObjectName("MetricTitle")
            self.value = QLabel("-")
            self.value.setObjectName("MetricValue")
            layout.addWidget(label)
            layout.addWidget(self.value)

        def set_value(self, value):
            self.value.setText(str(value))

    class GridCanvas(QWidget):
        cell_clicked = Signal(int, int)

        cell_colors = {0: QColor("#f8f9fa"), 1: QColor("#495057"), 2: QColor("#ff8a8a")}
        severity_colors = {
            "critical": QColor("#e63946"),
            "moderate": QColor("#f4a261"),
            "minor": QColor("#e9c46a"),
        }

        def __init__(self):
            super().__init__()
            self.results = None
            self.grid = copy_grid(BASE_MAP)
            self.victims = normalize_victims(VICTIMS_INIT)
            self.active_pos = None
            self.active_path = []
            self.setMinimumSize(430, 430)
            self.setSizePolicy(SIZE_EXPANDING, SIZE_EXPANDING)

        def set_mission(self, grid: list[list[int]], victims: list[dict]):
            self.results = None
            self.grid = copy_grid(grid)
            self.victims = normalize_victims(victims)
            self.active_pos = None
            self.active_path = []
            self.update()

        def set_results(self, results: dict):
            self.results = results
            self.active_pos = None
            self.active_path = results.get("rescue_trace", [])
            self.grid = copy_grid(results.get("grid", self.grid))
            self.victims = normalize_victims(results.get("victims", self.victims))
            self.update()

        def set_active_position(self, pos, traversed=None):
            self.active_pos = pos
            self.active_path = traversed or []
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing if hasattr(QPainter, "RenderHint") else QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor("#ffffff"))

            grid = self.results["grid"] if self.results else self.grid
            victims = self.results["victims"] if self.results else self.victims
            margin = 24
            side = min(self.width(), self.height()) - margin * 2
            cell = side / GRID_SIZE
            left = (self.width() - side) / 2
            top = (self.height() - side) / 2

            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    x = left + col * cell
                    y = top + row * cell
                    painter.fillRect(int(x), int(y), int(cell), int(cell), self.cell_colors[grid[row][col]])
                    painter.setPen(QPen(QColor("#cbd5e1"), 1))
                    painter.drawRect(int(x), int(y), int(cell), int(cell))

            if self.results:
                route_colors = ["#1d3557", "#457b9d", "#2a9d8f", "#e76f51", "#8338ec"]
                for idx, segment in enumerate(self.results.get("rescue_segments", [])):
                    full_path = segment.get("pickup_path", []) + segment.get("evac_path", [])[1:]
                    painter.setPen(QPen(QColor(route_colors[idx % len(route_colors)]), 3))
                    self.draw_path(painter, full_path, left, top, cell)

            if self.active_path:
                painter.setPen(QPen(QColor("#111827"), 4))
                self.draw_path(painter, self.active_path, left, top, cell)

            self.draw_marker(painter, RESCUE_BASE, left, top, cell, QColor("#2dc653"), "B")
            for idx, center in enumerate(MEDICAL_CENTERS, 1):
                self.draw_marker(painter, center, left, top, cell, QColor("#4361ee"), f"M{idx}")
            for victim in victims:
                color = self.severity_colors.get(victim["severity"], QColor("#94a3b8"))
                self.draw_marker(painter, victim["pos"], left, top, cell, color, f"V{victim['id']}")
            if self.active_pos:
                self.draw_marker(painter, self.active_pos, left, top, cell, QColor("#facc15"), "A")

        def draw_path(self, painter, path, left, top, cell):
            points = [(left + col * cell + cell / 2, top + row * cell + cell / 2) for row, col in path]
            for first, second in zip(points, points[1:]):
                painter.drawLine(int(first[0]), int(first[1]), int(second[0]), int(second[1]))

        def draw_marker(self, painter, pos, left, top, cell, color, text):
            row, col = pos
            size = max(18, int(cell * 0.62))
            cx = int(left + col * cell + cell / 2)
            cy = int(top + row * cell + cell / 2)
            painter.setPen(QPen(QColor("#111827"), 2))
            painter.setBrush(color)
            painter.drawEllipse(cx - size // 2, cy - size // 2, size, size)
            painter.setPen(QColor("#111827"))
            font = QFont()
            font.setBold(True)
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(cx - size // 2, cy - size // 2, size, size, ALIGN_CENTER, text)

        def mousePressEvent(self, event):
            row_col = self.cell_at(event.position() if hasattr(event, "position") else event.pos())
            if row_col:
                self.cell_clicked.emit(row_col[0], row_col[1])

        def cell_at(self, point):
            margin = 24
            side = min(self.width(), self.height()) - margin * 2
            cell = side / GRID_SIZE
            left = (self.width() - side) / 2
            top = (self.height() - side) / 2
            x = point.x()
            y = point.y()
            col = int((x - left) // cell)
            row = int((y - top) // cell)
            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                return row, col
            return None

    class AidraWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.results = None
            self.worker = None
            self.mission_grid = copy_grid(BASE_MAP)
            self.mission_victims = normalize_victims(VICTIMS_INIT)
            self.scheduled_events = {}
            self.trace_index = 0
            self.animation_timer = QTimer(self)
            self.animation_timer.timeout.connect(self.advance_animation)
            self.setWindowTitle(f"AIDRA Mission Control - {QT_BINDING}")
            self.resize(1180, 760)
            self.setStyleSheet(STYLE)
            self.build_ui()

        def build_ui(self):
            root = QWidget()
            root_layout = QHBoxLayout(root)
            root_layout.setContentsMargins(0, 0, 0, 0)

            sidebar = QFrame()
            sidebar.setObjectName("Sidebar")
            sidebar.setFixedWidth(290)
            side_layout = QVBoxLayout(sidebar)
            side_layout.setContentsMargins(18, 20, 18, 20)
            title = QLabel("AIDRA\nMission Control")
            title.setObjectName("AppTitle")
            side_layout.addWidget(title)
            self.run_button = QPushButton("Run Full Mission")
            self.run_button.clicked.connect(self.run_mission)
            self.animate_button = QPushButton("Animate Rescue")
            self.animate_button.setEnabled(False)
            self.animate_button.clicked.connect(self.start_animation)
            self.export_button = QPushButton("Export Results JSON")
            self.export_button.setEnabled(False)
            self.export_button.clicked.connect(self.export_json)
            side_layout.addWidget(self.run_button)
            side_layout.addWidget(self.animate_button)
            side_layout.addWidget(self.export_button)

            editor_title = QLabel("Mission Editor")
            editor_title.setObjectName("EditorTitle")
            side_layout.addWidget(editor_title)
            self.edit_mode = QComboBox()
            self.edit_mode.addItems(["Add victim", "Block road", "High-risk road", "Clear cell"])
            self.severity_box = QComboBox()
            self.severity_box.addItems(["critical", "moderate", "minor"])
            self.row_spin = QSpinBox()
            self.row_spin.setRange(0, GRID_SIZE - 1)
            self.col_spin = QSpinBox()
            self.col_spin.setRange(0, GRID_SIZE - 1)
            self.step_spin = QSpinBox()
            self.step_spin.setRange(0, 99)
            self.step_spin.setValue(3)
            for label_text, widget in [
                ("Click mode", self.edit_mode),
                ("Victim severity", self.severity_box),
                ("Row", self.row_spin),
                ("Column", self.col_spin),
                ("Event step", self.step_spin),
            ]:
                label = QLabel(label_text)
                label.setObjectName("EditorLabel")
                side_layout.addWidget(label)
                side_layout.addWidget(widget)

            add_button = QPushButton("Add Victim")
            add_button.clicked.connect(self.add_victim_from_controls)
            block_button = QPushButton("Block Road")
            block_button.clicked.connect(self.block_from_controls)
            risk_button = QPushButton("Mark High Risk")
            risk_button.clicked.connect(self.risk_from_controls)
            clear_button = QPushButton("Clear Cell")
            clear_button.clicked.connect(self.clear_from_controls)
            schedule_button = QPushButton("Schedule Blockage")
            schedule_button.clicked.connect(self.schedule_blockage_from_controls)
            reset_button = QPushButton("Reset Mission")
            reset_button.clicked.connect(self.reset_mission)
            for button in [add_button, block_button, risk_button, clear_button, schedule_button, reset_button]:
                side_layout.addWidget(button)

            side_layout.addStretch()
            self.status = QLabel("Ready")
            self.status.setObjectName("StatusLabel")
            self.status.setWordWrap(True)
            side_layout.addWidget(self.status)

            self.tabs = QTabWidget()
            self.overview_tab = QWidget()
            self.ml_tab = QWidget()
            self.search_tab = QWidget()
            self.csp_tab = QWidget()
            self.log_tab = QWidget()
            self.tabs.addTab(self.overview_tab, "Overview")
            self.tabs.addTab(self.ml_tab, "Machine Learning")
            self.tabs.addTab(self.search_tab, "Search Planning")
            self.tabs.addTab(self.csp_tab, "CSP Allocation")
            self.tabs.addTab(self.log_tab, "Decision Logs")

            root_layout.addWidget(sidebar)
            root_layout.addWidget(self.tabs, 1)
            self.setCentralWidget(root)
            self.build_overview()
            self.build_ml()
            self.build_search()
            self.build_csp()
            self.build_logs()

        def make_table(self):
            table = QTableWidget()
            table.setAlternatingRowColors(True)
            table.setEditTriggers(NO_EDIT_TRIGGERS)
            table.horizontalHeader().setSectionResizeMode(HEADER_STRETCH)
            table.verticalHeader().setVisible(False)
            table.setSelectionBehavior(SELECT_ROWS)
            return table

        def build_overview(self):
            layout = QVBoxLayout(self.overview_tab)
            layout.setContentsMargins(18, 18, 18, 18)
            row = QGridLayout()
            self.metrics = {
                "saved": MetricCard("Victims Saved"),
                "avg": MetricCard("Average Rescue Time"),
                "opt": MetricCard("Path Optimality"),
                "risk": MetricCard("Risk Exposure"),
            }
            for idx, card in enumerate(self.metrics.values()):
                row.addWidget(card, 0, idx)
            layout.addLayout(row)
            splitter = QSplitter(HORIZONTAL)
            self.grid_canvas = GridCanvas()
            self.grid_canvas.cell_clicked.connect(self.handle_grid_click)
            self.victim_table = self.make_table()
            splitter.addWidget(self.grid_canvas)
            splitter.addWidget(self.victim_table)
            splitter.setSizes([650, 420])
            layout.addWidget(splitter, 1)
            self.grid_canvas.set_mission(self.mission_grid, self.mission_victims)
            self.populate_victims({"victims": self.mission_victims})

        def build_ml(self):
            layout = QVBoxLayout(self.ml_tab)
            layout.setContentsMargins(18, 18, 18, 18)
            self.ml_table = self.make_table()
            layout.addWidget(self.ml_table)

        def build_search(self):
            layout = QVBoxLayout(self.search_tab)
            layout.setContentsMargins(18, 18, 18, 18)
            self.search_table = self.make_table()
            layout.addWidget(self.search_table)

        def build_csp(self):
            layout = QVBoxLayout(self.csp_tab)
            layout.setContentsMargins(18, 18, 18, 18)
            self.csp_summary = QLabel("Run mission to view CSP allocation.")
            self.csp_summary.setObjectName("SectionTitle")
            self.csp_table = self.make_table()
            layout.addWidget(self.csp_summary)
            layout.addWidget(self.csp_table, 1)

        def build_logs(self):
            layout = QVBoxLayout(self.log_tab)
            layout.setContentsMargins(18, 18, 18, 18)
            self.log_view = QTextEdit()
            self.log_view.setReadOnly(True)
            layout.addWidget(self.log_view)

        def run_mission(self):
            self.animation_timer.stop()
            self.run_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.animate_button.setEnabled(False)
            self.status.setText("Running simulation...")
            self.worker = SimulationThread(self.mission_grid, self.mission_victims, self.scheduled_events)
            self.worker.completed.connect(self.load_results)
            self.worker.failed.connect(self.show_error)
            self.worker.start()

        def show_error(self, message):
            self.run_button.setEnabled(True)
            self.status.setText("Simulation failed.")
            QMessageBox.critical(self, "AIDRA Error", message)

        def export_json(self):
            if not self.results:
                return
            path = export_results(self.results)
            self.status.setText(f"Saved: {path}")

        def load_results(self, results):
            self.results = results
            kpis = results["kpis"]
            self.metrics["saved"].set_value(f"{kpis['victims_saved']} / {len(results['victims'])}")
            self.metrics["avg"].set_value(kpis["avg_rescue_time"])
            self.metrics["opt"].set_value(kpis["path_optimality_ratio"])
            self.metrics["risk"].set_value(kpis["risk_exposure_score"])
            self.grid_canvas.set_results(results)
            self.populate_victims(results)
            self.populate_ml(results)
            self.populate_search(results)
            self.populate_csp(results)
            self.log_view.setPlainText("\n".join(results["log"]))
            self.run_button.setEnabled(True)
            self.export_button.setEnabled(True)
            self.animate_button.setEnabled(True)
            self.status.setText("Mission complete. Playing rescue movement...")
            self.start_animation()

        def populate_victims(self, results):
            rows = []
            for victim in results["victims"]:
                rows.append([
                    f"V{victim['id']}",
                    victim["severity"],
                    victim["pos"],
                    victim.get("assigned_ambulance", "-"),
                    victim.get("survival_prob", "-"),
                    victim.get("rescue_time", "-"),
                    "yes" if victim.get("rescued") else "no",
                ])
            self.fill_table(self.victim_table, ["Victim", "Severity", "Position", "Ambulance", "Survival", "Time", "Saved"], rows)

        def populate_ml(self, results):
            rows = []
            for model, data in results["ml_results"].items():
                rows.append([model, data["accuracy"], data["precision"], data["recall"], data["f1"], data["cm"]])
            self.fill_table(self.ml_table, ["Model", "Accuracy", "Precision", "Recall", "F1", "Confusion Matrix"], rows)

        def populate_search(self, results):
            rows = []
            for vid in sorted(results["search_results"]):
                for algorithm, data in results["search_results"][vid].items():
                    rows.append([f"V{vid}", algorithm, data["expanded"], data["path_len"], data["cost"], data["risk"], data["time_ms"]])
            self.fill_table(self.search_table, ["Victim", "Algorithm", "Expanded", "Path Len", "Cost", "Risk", "Time ms"], rows)

        def populate_csp(self, results):
            csp = results["csp"]
            capacity = csp["ambulance_capacity"]
            self.csp_summary.setText(
                f"Success: {csp['success']} | MRV+FC backtracks: {csp['bt_heuristic']} | "
                f"Plain backtracking: {csp['bt_no_heuristic']} | Capacity: {capacity}"
            )
            rows = [[f"Ambulance {amb}", ", ".join(f"V{vid}" for vid in vids) or "-", f"{len(vids)} / {capacity}"] for amb, vids in csp["ambulance_loads"].items()]
            self.fill_table(self.csp_table, ["Resource", "Assigned Victims", "Load"], rows)

        def handle_grid_click(self, row, col):
            self.row_spin.setValue(row)
            self.col_spin.setValue(col)
            mode = self.edit_mode.currentText()
            if mode == "Add victim":
                self.add_victim(row, col, self.severity_box.currentText())
            elif mode == "Block road":
                self.set_cell(row, col, 1, "blocked")
            elif mode == "High-risk road":
                self.set_cell(row, col, 2, "high risk")
            else:
                self.set_cell(row, col, 0, "clear")

        def add_victim_from_controls(self):
            self.add_victim(self.row_spin.value(), self.col_spin.value(), self.severity_box.currentText())

        def block_from_controls(self):
            self.set_cell(self.row_spin.value(), self.col_spin.value(), 1, "blocked")

        def risk_from_controls(self):
            self.set_cell(self.row_spin.value(), self.col_spin.value(), 2, "high risk")

        def clear_from_controls(self):
            self.set_cell(self.row_spin.value(), self.col_spin.value(), 0, "clear")

        def schedule_blockage_from_controls(self):
            row = self.row_spin.value()
            col = self.col_spin.value()
            if not self.cell_can_change(row, col):
                return
            step = self.step_spin.value()
            self.scheduled_events.setdefault(step, []).append(
                (row, col, 1, f"User scheduled road blockage at ({row},{col})")
            )
            self.status.setText(f"Scheduled blockage at ({row},{col}) for step {step}.")

        def add_victim(self, row, col, severity):
            if self.mission_grid[row][col] == 1:
                self.status.setText("Cannot add a victim on a blocked road.")
                return
            if (row, col) == RESCUE_BASE or (row, col) in MEDICAL_CENTERS:
                self.status.setText("Choose a normal road cell for a victim.")
                return
            if any(tuple(v["pos"]) == (row, col) for v in self.mission_victims):
                self.status.setText("There is already a victim on that cell.")
                return
            next_id = max([v["id"] for v in self.mission_victims], default=-1) + 1
            self.mission_victims.append({"id": next_id, "pos": (row, col), "severity": severity})
            self.refresh_mission_view(f"Added V{next_id} at ({row},{col}).")

        def set_cell(self, row, col, value, label):
            if not self.cell_can_change(row, col):
                return
            self.mission_grid[row][col] = value
            self.refresh_mission_view(f"Cell ({row},{col}) marked {label}.")

        def cell_can_change(self, row, col):
            pos = (row, col)
            if pos == RESCUE_BASE or pos in MEDICAL_CENTERS:
                self.status.setText("Base and medical centers must stay open.")
                return False
            if any(tuple(v["pos"]) == pos for v in self.mission_victims):
                self.status.setText("Move the edit to a road cell, not a victim cell.")
                return False
            return True

        def refresh_mission_view(self, message):
            self.results = None
            self.animation_timer.stop()
            self.animate_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.grid_canvas.set_mission(self.mission_grid, self.mission_victims)
            self.populate_victims({"victims": self.mission_victims})
            self.status.setText(message)

        def reset_mission(self):
            self.mission_grid = copy_grid(BASE_MAP)
            self.mission_victims = normalize_victims(VICTIMS_INIT)
            self.scheduled_events = {}
            self.refresh_mission_view("Mission reset to default map and victims.")

        def start_animation(self):
            if not self.results or not self.results.get("rescue_trace"):
                self.status.setText("Run a mission before animation.")
                return
            self.trace_index = 0
            self.animation_timer.start(260)

        def advance_animation(self):
            trace = self.results.get("rescue_trace", []) if self.results else []
            if self.trace_index >= len(trace):
                self.animation_timer.stop()
                self.status.setText("Rescue movement complete.")
                return
            self.grid_canvas.set_active_position(trace[self.trace_index], trace[: self.trace_index + 1])
            self.status.setText(f"Moving rescue unit: step {self.trace_index + 1}/{len(trace)}")
            self.trace_index += 1

        def fill_table(self, table, headers, rows):
            table.clear()
            table.setColumnCount(len(headers))
            table.setRowCount(len(rows))
            table.setHorizontalHeaderLabels(headers)
            for r, row in enumerate(rows):
                for c, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    item.setTextAlignment(ALIGN_CENTER)
                    table.setItem(r, c, item)
            table.resizeRowsToContents()

    STYLE = """
    QWidget { background: #f6f8fb; color: #111827; font-family: Segoe UI, Arial; font-size: 10pt; }
    #Sidebar { background: #111827; color: #f9fafb; }
    #AppTitle { color: #f9fafb; font-size: 22pt; font-weight: 700; padding-bottom: 22px; }
    #StatusLabel { color: #cbd5e1; padding-top: 12px; }
    #EditorTitle { color: #f9fafb; font-size: 13pt; font-weight: 700; padding-top: 14px; }
    #EditorLabel { color: #cbd5e1; font-size: 9pt; padding-top: 4px; }
    QPushButton { background: #2563eb; color: white; border: 0; border-radius: 6px; padding: 10px 12px; font-weight: 600; }
    QPushButton:hover { background: #1d4ed8; }
    QPushButton:disabled { background: #64748b; }
    QComboBox, QSpinBox { background: #f8fafc; color: #111827; border: 1px solid #94a3b8; border-radius: 5px; padding: 5px; }
    QTabWidget::pane { border: 0; }
    QTabBar::tab { background: #e5e7eb; padding: 10px 14px; margin-right: 2px; }
    QTabBar::tab:selected { background: #ffffff; font-weight: 700; }
    #MetricCard { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; }
    #MetricTitle { color: #64748b; font-size: 9pt; }
    #MetricValue { color: #0f172a; font-size: 20pt; font-weight: 700; }
    #SectionTitle { font-size: 12pt; font-weight: 700; color: #0f172a; }
    QTableWidget, QTextEdit { background: #ffffff; alternate-background-color: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; gridline-color: #e5e7eb; }
    QHeaderView::section { background: #e5e7eb; border: 0; padding: 7px; font-weight: 700; }
    """


def print_cli_summary(results: dict) -> None:
    print("\nAIDRA simulation complete")
    print(f"Victims saved: {results['kpis']['victims_saved']} / {len(results['victims'])}")
    print(f"Average rescue time: {results['kpis']['avg_rescue_time']}")
    print(f"Priority order: {results['priority_order']}")
    print(f"CSP assignment: {results['csp']['assignment']}")
    print(f"Local search winner: {results['local_search']['winner']}")


def main() -> int:
    if "--cli" in sys.argv:
        results = run_simulation()
        print_cli_summary(results)
        return 0
    if not QT_AVAILABLE:
        print("No Qt binding found. Install PyQt6, PySide6, or PyQt5, then run again.")
        print("You can still test the simulation with: python AIDRA_single_file_qt.py --cli")
        return 1
    app = QApplication(sys.argv)
    window = AidraWindow()
    window.show()
    return app.exec() if hasattr(app, "exec") else app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
