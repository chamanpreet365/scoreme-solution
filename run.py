#!/usr/bin/env python3
"""
run.py — Benchmark runner for PWDRAB scheduler.
Generates instance, runs the scheduler (Python reimplementation for portability),
and reports results. For small n also runs brute-force optimal.
"""

import argparse
import json
import time
import sys
import random
import itertools
import math
from generate_instance import generate_instance

# ============================================================
# Penalty Calculator (mirrors PenaltyCalculator.java exactly)
# ============================================================

LAMBDA_IMBALANCE = 0.5
LAMBDA_SLA_RISK  = 1.0
LAMBDA_GPU_FRAG  = 0.2
GPU_DIM = 2

def penalty(assignment, inst):
    n = inst['n']; K = inst['K']; d = 4
    resources = inst['resources']
    capacities = inst['capacities']
    windows = inst['windows']
    weights = inst['weights']

    # P_base
    base = sum(weights[i] * (assignment[i] + 1) for i in range(n) if assignment[i] >= 0)

    # P_imbalance
    usage = [[0.0]*d for _ in range(K)]
    for i in range(n):
        s = assignment[i]
        if s >= 0:
            for dd in range(d):
                usage[s][dd] += resources[i][dd]
    imbalance = 0.0
    for s in range(K):
        utils = []
        for dd in range(d):
            cap = capacities[s][dd]
            if cap > 0:
                utils.append(usage[s][dd] / cap)
        if utils:
            imbalance += max(utils) - min(utils)

    # P_sla_risk
    sla_risk = 0.0
    for i in range(n):
        s = assignment[i]
        if s >= 0:
            lo, hi = windows[i]
            win = hi - lo + 1
            sla_risk += weights[i] * (s - lo) / win

    # P_gpu_frag
    gpu_used = [0.0] * K
    has_gpu = [False] * K
    for i in range(n):
        s = assignment[i]
        if s >= 0:
            gpu_used[s] += resources[i][GPU_DIM]
            if resources[i][GPU_DIM] > 0:
                has_gpu[s] = True
    gpu_frag = sum((capacities[s][GPU_DIM] - gpu_used[s]) for s in range(K) if has_gpu[s])

    return base + LAMBDA_IMBALANCE*imbalance + LAMBDA_SLA_RISK*sla_risk + LAMBDA_GPU_FRAG*gpu_frag

# ============================================================
# Feasibility checker
# ============================================================

def is_feasible(assignment, inst):
    n = inst['n']; K = inst['K']; d = 4
    conflicts = set(map(tuple, [tuple(sorted(e)) for e in inst['conflicts']]))
    resources = inst['resources']
    capacities = inst['capacities']
    windows = inst['windows']

    # F1: no conflicting pair in same slot
    for (i, j) in conflicts:
        if assignment[i] == assignment[j]:
            return False, f"Conflict: T{i} and T{j} share slot {assignment[i]}"

    # F2: capacity
    usage = [[0.0]*d for _ in range(K)]
    for i in range(n):
        s = assignment[i]
        for dd in range(d):
            usage[s][dd] += resources[i][dd]
    for s in range(K):
        for dd in range(d):
            if usage[s][dd] > capacities[s][dd] + 1e-9:
                return False, f"Slot {s} dim {dd} over capacity"

    # F3: SLA windows
    for i in range(n):
        lo, hi = windows[i]
        if not (lo <= assignment[i] <= hi):
            return False, f"T{i} SLA violated: slot {assignment[i]} not in [{lo},{hi}]"

    return True, None

# ============================================================
# PWDRAB scheduler (Python port)
# ============================================================

def solve_pwdrab(inst):
    n = inst['n']; K = inst['K']; d = 4
    conflicts_list = inst['conflicts']
    resources = inst['resources']
    capacities = inst['capacities']
    windows = inst['windows']
    weights = inst['weights']

    # Build adjacency
    adj = [set() for _ in range(n)]
    for (i, j) in conflicts_list:
        adj[i].add(j); adj[j].add(i)

    assignment = [-1] * n
    slot_usage = [[0.0]*d for _ in range(K)]
    neighbour_colors = [set() for _ in range(n)]  # slots used by assigned neighbours
    saturation = [0] * n
    degree = [len(adj[i]) for i in range(n)]

    assigned = [False] * n
    history = []  # stack of (task, slot)
    tried = [set() for _ in range(n)]
    backtracks_left = 10 * n

    def resource_fits(t, s):
        for dd in range(d):
            if slot_usage[s][dd] + resources[t][dd] > capacities[s][dd] + 1e-9:
                return False
        return True

    def incremental_penalty(t, s):
        base = weights[t] * (s + 1)
        lo, hi = windows[t]
        win = hi - lo + 1
        sla = weights[t] * (s - lo) / win
        gpu_this = resources[t][GPU_DIM]
        gpu_frag = (capacities[s][GPU_DIM] - slot_usage[s][GPU_DIM] - gpu_this) * 0.2 if gpu_this > 0 else 0
        return base + sla + gpu_frag

    def feasible_slots(t):
        lo, hi = windows[t]
        cands = []
        for s in range(lo, hi+1):
            if s in neighbour_colors[t]: continue
            if not resource_fits(t, s): continue
            if s in tried[t]: continue
            cands.append(s)
        cands.sort(key=lambda s: incremental_penalty(t, s))
        return cands

    def do_assign(t, s):
        assignment[t] = s
        assigned[t] = True
        for dd in range(d):
            slot_usage[s][dd] += resources[t][dd]
        for nb in adj[t]:
            if not assigned[nb]:
                if s not in neighbour_colors[nb]:
                    neighbour_colors[nb].add(s)
                    saturation[nb] += 1

    def do_unassign(t, s):
        assignment[t] = -1
        assigned[t] = False
        for dd in range(d):
            slot_usage[s][dd] -= resources[t][dd]
        tried[t].add(s)
        for nb in adj[t]:
            if not assigned[nb]:
                neighbour_colors[nb].clear()
                saturation[nb] = 0
                for nb2 in adj[nb]:
                    if assigned[nb2]:
                        s2 = assignment[nb2]
                        if s2 not in neighbour_colors[nb]:
                            neighbour_colors[nb].add(s2)
                            saturation[nb] += 1

    def pick_most_constrained():
        best = -1; bsat=-1; bdeg=-1; bwin=10**9; bw=-1
        for t in range(n):
            if assigned[t]: continue
            sat = saturation[t]; deg = degree[t]
            win = windows[t][1] - windows[t][0] + 1; w = weights[t]
            if (sat,deg,-win,w) > (bsat,bdeg,-bwin,bw):
                best=t; bsat=sat; bdeg=deg; bwin=win; bw=w
        return best

    step = 0
    while step < n:
        t = pick_most_constrained()
        if t == -1: break
        cands = feasible_slots(t)
        if not cands:
            if not history or backtracks_left <= 0:
                break
            backtracks_left -= 1
            pt, ps = history.pop()
            do_unassign(pt, ps)
            step -= 2
            assigned[t] = False
            step -= 1
        else:
            s_star = cands[0]
            do_assign(t, s_star)
            history.append((t, s_star))
        step += 1

    # Local search repair for any unassigned tasks
    for i in range(n):
        if assignment[i] < 0:
            lo, hi = windows[i]
            assignment[i] = lo  # force assign to window start
            for dd in range(d):
                slot_usage[lo][dd] += resources[i][dd]

    return assignment

# ============================================================
# Brute force (only for small n ≤ 12)
# ============================================================

def brute_force_optimal(inst):
    n = inst['n']; K = inst['K']
    windows = inst['windows']
    # Build candidate slots per task
    candidates = [list(range(windows[i][0], windows[i][1]+1)) for i in range(n)]
    best_penalty = float('inf')
    best_asgn = None
    for asgn in itertools.product(*candidates):
        asgn = list(asgn)
        ok, _ = is_feasible(asgn, inst)
        if ok:
            p = penalty(asgn, inst)
            if p < best_penalty:
                best_penalty = p
                best_asgn = asgn[:]
    return best_asgn, best_penalty

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--density', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    inst = generate_instance(args.n, args.K, conflict_density=args.density, seed=args.seed)
    inst['n'] = args.n

    t0 = time.time()
    assignment = solve_pwdrab(inst)
    elapsed_ms = int((time.time() - t0) * 1000)

    feasible, vr = is_feasible(assignment, inst)
    p = penalty(assignment, inst) if feasible else float('nan')

    result = {
        'n': args.n, 'K': args.K, 'density': args.density, 'seed': args.seed,
        'feasible': feasible,
        'penalty': round(p, 4) if not math.isnan(p) else None,
        'runtime_ms': elapsed_ms,
        'violation_reason': vr
    }

    # Brute force for small instances
    if args.n <= 12:
        t0 = time.time()
        bf_asgn, bf_penalty = brute_force_optimal(inst)
        bf_ms = int((time.time() - t0) * 1000)
        result['brute_force_penalty'] = round(bf_penalty, 4) if bf_asgn is not None else None
        result['brute_force_feasible'] = bf_asgn is not None
        result['brute_force_ms'] = bf_ms
        if bf_asgn is not None and feasible:
            result['approx_ratio'] = round(p / bf_penalty, 4)

    print(json.dumps(result, indent=2))
