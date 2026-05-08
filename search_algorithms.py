"""
search_algorithms.py
====================
AIDRA – Adaptive Intelligent Disaster Response Agent
AIC-201 CCP | Module 2: Uninformed & Heuristic Search

Implements:
  • BFS          – breadth-first search (uninformed, optimal on uniform cost)
  • DFS          – depth-first search (uninformed, not cost-optimal)
  • Greedy BFS   – heuristic-guided, fast but not cost-optimal
  • A*           – heuristic + cost-weighted, optimal with admissible heuristic
  • Utilities    – neighbour generation, risk cost, path metrics
"""

import heapq
from collections import deque
from environment import GRID_SIZE, CELL_COST

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_neighbors(pos: tuple, grid: list, allow_risk: bool = True) -> list:
    """
    Return valid adjacent cells (4-directional) from *pos*.

    Parameters
    ----------
    pos        : (row, col) current cell
    grid       : 2-D list with cell values (0=free, 1=blocked, 2=high-risk)
    allow_risk : if False, high-risk cells (value 2) are skipped
    """
    r, c = pos
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
            continue
        cell = grid[nr][nc]
        if cell == 1:
            continue                          # wall / rubble → impassable
        if cell == 2 and not allow_risk:
            continue                          # avoid hazard zones if requested
        neighbors.append((nr, nc))
    return neighbors


def step_cost(pos: tuple, grid: list) -> int:
    """
    Return the movement cost of entering *pos*.

    free cell      → 1
    high-risk cell → 3  (danger penalty + slower traverse)
    """
    return CELL_COST.get(grid[pos[0]][pos[1]], 1)


def manhattan(a: tuple, b: tuple) -> int:
    """Admissible Manhattan-distance heuristic (no diagonal moves)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def path_risk_cells(path: list, grid: list) -> int:
    """Count how many cells in *path* are high-risk (value 2)."""
    return sum(1 for p in path if grid[p[0]][p[1]] == 2)


def path_weighted_cost(path: list, grid: list) -> int:
    """Sum of step costs along the path (risk-weighted)."""
    return sum(step_cost(p, grid) for p in path)


# ─────────────────────────────────────────────────────────────────────────────
# BFS  –  Breadth-First Search
# ─────────────────────────────────────────────────────────────────────────────

def bfs(start: tuple, goal: tuple, grid: list,
        allow_risk: bool = True) -> tuple[list | None, int]:
    """
    Breadth-first search.

    Guaranteed to find the *fewest-step* path (optimal when all edges have
    equal cost).  Not risk-weighted.

    Returns
    -------
    (path, nodes_expanded) or (None, nodes_expanded) if unreachable
    """
    queue          = deque([(start, [start])])
    visited        = {start}
    nodes_expanded = 0

    while queue:
        cur, path = queue.popleft()
        nodes_expanded += 1

        if cur == goal:
            return path, nodes_expanded

        for nb in get_neighbors(cur, grid, allow_risk):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))

    return None, nodes_expanded


# ─────────────────────────────────────────────────────────────────────────────
# DFS  –  Depth-First Search
# ─────────────────────────────────────────────────────────────────────────────

def dfs(start: tuple, goal: tuple, grid: list,
        allow_risk: bool = True) -> tuple[list | None, int]:
    """
    Depth-first search.

    Explores deep paths first; memory-efficient but NOT cost-optimal.
    May return long, winding routes.

    Returns
    -------
    (path, nodes_expanded) or (None, nodes_expanded) if unreachable
    """
    stack          = [(start, [start])]
    visited        = set()
    nodes_expanded = 0

    while stack:
        cur, path = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        nodes_expanded += 1

        if cur == goal:
            return path, nodes_expanded

        for nb in get_neighbors(cur, grid, allow_risk):
            if nb not in visited:
                stack.append((nb, path + [nb]))

    return None, nodes_expanded


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY BEST-FIRST SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def greedy_bfs(start: tuple, goal: tuple, grid: list,
               allow_risk: bool = True) -> tuple[list | None, int]:
    """
    Greedy Best-First Search.

    Always expands the node closest to the goal by heuristic estimate h(n).
    Very fast in practice but NOT guaranteed to find the optimal path.

    Returns
    -------
    (path, nodes_expanded) or (None, nodes_expanded)
    """
    heap           = [(manhattan(start, goal), start, [start])]
    visited        = set()
    nodes_expanded = 0

    while heap:
        _, cur, path = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        nodes_expanded += 1

        if cur == goal:
            return path, nodes_expanded

        for nb in get_neighbors(cur, grid, allow_risk):
            if nb not in visited:
                heapq.heappush(heap, (manhattan(nb, goal), nb, path + [nb]))

    return None, nodes_expanded


# ─────────────────────────────────────────────────────────────────────────────
# A*  –  A-Star Search  (primary planner)
# ─────────────────────────────────────────────────────────────────────────────

def astar(start: tuple, goal: tuple, grid: list,
          allow_risk: bool = True) -> tuple[list | None, int]:
    """
    A* Search with risk-weighted edge costs.

    f(n) = g(n) + h(n)
      g(n) : accumulated weighted cost from start
             (free cell = 1, high-risk cell = 3)
      h(n) : Manhattan distance to goal (admissible)

    Guaranteed to find the cost-optimal path (given admissible heuristic).

    Parameters
    ----------
    allow_risk : if False, high-risk cells are treated as impassable,
                 enabling the "safe route" variant.

    Returns
    -------
    (path, nodes_expanded) or (None, nodes_expanded)
    """
    # heap entry: (f, g, node, path)
    heap           = [(manhattan(start, goal), 0, start, [start])]
    best_g         = {}          # best known g-cost reaching each node
    nodes_expanded = 0

    while heap:
        f, g, cur, path = heapq.heappop(heap)

        if cur in best_g and best_g[cur] <= g:
            continue                          # already found a cheaper path here
        best_g[cur]     = g
        nodes_expanded += 1

        if cur == goal:
            return path, nodes_expanded

        for nb in get_neighbors(cur, grid, allow_risk):
            ng = g + step_cost(nb, grid)
            nf = ng + manhattan(nb, goal)
            heapq.heappush(heap, (nf, ng, nb, path + [nb]))

    return None, nodes_expanded


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE COMPARISON HELPER
# ─────────────────────────────────────────────────────────────────────────────

def compare_all_algorithms(start: tuple, goal: tuple,
                           grid: list) -> dict:
    """
    Run all four search algorithms from *start* to *goal* and return a
    dictionary of per-algorithm statistics for analysis and reporting.

    Returns
    -------
    {
      'BFS':    {'path': [...], 'path_len': int, 'cost': int, 'risk': int, 'expanded': int},
      'DFS':    {...},
      'Greedy': {...},
      'A*':     {...},
      'A*-safe':{...},   # A* avoiding high-risk cells entirely
    }
    """
    import time

    algorithms = {
        'BFS':     lambda: bfs(start, goal, grid, allow_risk=True),
        'DFS':     lambda: dfs(start, goal, grid, allow_risk=True),
        'Greedy':  lambda: greedy_bfs(start, goal, grid, allow_risk=True),
        'A*':      lambda: astar(start, goal, grid, allow_risk=True),
        'A*-safe': lambda: astar(start, goal, grid, allow_risk=False),
    }

    results = {}
    for name, fn in algorithms.items():
        t0 = time.perf_counter()
        path, expanded = fn()
        elapsed = round((time.perf_counter() - t0) * 1000, 3)   # ms

        if path:
            results[name] = {
                'path':     path,
                'path_len': len(path),
                'cost':     path_weighted_cost(path, grid),
                'risk':     path_risk_cells(path, grid),
                'expanded': expanded,
                'time_ms':  elapsed,
            }
        else:
            results[name] = {
                'path': None, 'path_len': 0, 'cost': 0,
                'risk': 0, 'expanded': expanded, 'time_ms': elapsed,
            }

    return results


def select_best_route(route_data: dict) -> tuple[list | None, str, str]:
    """
    Choose between the standard A* path and the risk-avoiding A* path.

    Decision rule
    -------------
    • If the safe path exists and its weighted cost ≤ the standard path's
      cost, prefer the safe path (same or better time, lower risk).
    • Otherwise take the standard A* path and log the speed-vs-risk trade-off.

    Returns
    -------
    (chosen_path, chosen_algorithm_name, justification_string)
    """
    astar_info = route_data.get('A*',      {})
    safe_info  = route_data.get('A*-safe', {})

    astar_path = astar_info.get('path')
    safe_path  = safe_info.get('path')

    if not astar_path and not safe_path:
        return None, 'None', 'No path available after replanning'

    if astar_path and not safe_path:
        return (astar_path, 'A*',
                f"No risk-free path found; using A* "
                f"(cost={astar_info['cost']}, risk_cells={astar_info['risk']})")

    if safe_path and not astar_path:
        return (safe_path, 'A*-safe',
                f"Only safe path available "
                f"(cost={safe_info['cost']}, risk_cells=0)")

    # Both exist – compare weighted costs
    if safe_info['cost'] <= astar_info['cost']:
        return (safe_path, 'A*-safe',
                f"Safe path chosen (cost={safe_info['cost']}) ≤ "
                f"fast path (cost={astar_info['cost']}); "
                f"Objective 1 resolved: lower risk, no extra time")
    else:
        return (astar_path, 'A*',
                f"Fast path chosen (cost={astar_info['cost']}) vs "
                f"safe path (cost={safe_info['cost']}); "
                f"Objective 1 resolved: speed prioritised, "
                f"risk_cells={astar_info['risk']}")