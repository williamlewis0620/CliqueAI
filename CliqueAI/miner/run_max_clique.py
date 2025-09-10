
import argparse
import json
from max_clique_solver import solve_max_clique_all

def load_edge_list(path):
    edges = []
    n = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
            n = max(n, u+1, v+1)
    return n, edges

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Maximum Clique size and all maximum cliques (bitset B&B with coloring).")
    ap.add_argument("--edges", type=str, required=True, help="Path to edge list file: each line 'u v' (0-indexed).")
    ap.add_argument("--time", type=float, default=30.0, help="Total time budget in seconds (default: 30).")
    ap.add_argument("--enum-cap", type=int, default=None, help="Optional cap on number of max cliques to enumerate.")
    ap.add_argument("--no-reorder", action="store_true", help="Disable degeneracy reordering.")
    args = ap.parse_args()

    n, edges = load_edge_list(args.edges)
    res = solve_max_clique_all(n, edges, time_budget_sec=args.time,
                               enum_cap=args.enum_cap, reorder=not args.no_reorder)
    print(json.dumps(res, indent=2))
