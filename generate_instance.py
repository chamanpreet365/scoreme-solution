import random
import json
import sys

def generate_instance(n, K, d=4, conflict_density=0.3, seed=42):
    """Generate a random MSME Credit Pipeline Scheduling instance."""
    random.seed(seed)
    tasks = [f'T{i}' for i in range(n)]
    conflicts = [(i, j) for i in range(n) for j in range(i+1, n)
                 if random.random() < conflict_density]
    cap = [32, 128, 8, 6.0]  # CPU, RAM, GPU, Network
    resources = [[random.uniform(1, cap[dd] // (n // K + 1))
                  for dd in range(4)] for _ in range(n)]
    capacities = [cap[:] for _ in range(K)]
    windows = [(lo := random.randint(0, K-2),
                random.randint(lo+1, K-1)) for _ in range(n)]
    weights = [random.uniform(1, 10) for _ in range(n)]
    return dict(tasks=tasks, conflicts=conflicts,
                resources=resources, capacities=capacities,
                windows=windows, weights=weights, K=K)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--density', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    inst = generate_instance(args.n, args.K, conflict_density=args.density, seed=args.seed)
    out = json.dumps(inst, indent=2)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(out)
    else:
        print(out)
