
from typing import List, Tuple, Dict, Optional
import time
import math

def _lsb_index(x: int) -> int:
    return (x & -x).bit_length() - 1

class BitGraph:
    __slots__ = ("n", "adj")
    def __init__(self, n: int, adj: List[int]):
        self.n = n
        self.adj = adj

    @staticmethod
    def from_edges(n: int, edges: List[Tuple[int, int]]) -> "BitGraph":
        adj = [0] * n
        for u, v in edges:
            if u == v:
                continue
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"Edge ({u},{v}) out of bounds for n={n}")
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return BitGraph(n, adj)

    def degrees(self) -> List[int]:
        return [self.adj[v].bit_count() for v in range(self.n)]

    def reorder_by_degeneracy(self):
        n = self.n
        deg = self.degrees()
        remaining = set(range(n))
        order = []
        while remaining:
            v = min(remaining, key=lambda x: deg[x])
            remaining.remove(v)
            order.append(v)
            nbrs = self.adj[v]
            u = nbrs
            while u:
                w = _lsb_index(u)
                if w in remaining:
                    deg[w] -= 1
                u &= u - 1
        invperm = order
        perm = [0] * n
        for new_i, old_v in enumerate(invperm):
            perm[old_v] = new_i
        new_adj = [0] * n
        for old_v in range(n):
            new_v = perm[old_v]
            mask = self.adj[old_v]
            tmp = mask
            rel = 0
            while tmp:
                old_u = _lsb_index(tmp)
                rel |= (1 << perm[old_u])
                tmp &= tmp - 1
            new_adj[new_v] = rel & ~(1 << new_v)
        return BitGraph(n, new_adj), perm, invperm

def _color_sort(P: int, adj: List[int]):
    order = []
    colors = []
    color = 0
    P_work = P
    while P_work:
        color += 1
        Q = P_work
        this_color_mask = 0
        while Q:
            v = _lsb_index(Q)
            v_bit = 1 << v
            order.append(v)
            colors.append(color)
            this_color_mask |= v_bit
            Q &= ~v_bit
            Q &= ~adj[v]
        P_work &= ~this_color_mask
    return order, colors

class MaxCliqueSolver:
    def __init__(self, G: BitGraph):
        self.G = G
        self.n = G.n
        self.adj = G.adj
        self.best_size = 0
        self.best_bits = 0
        self.start_time = 0.0
        self.deadline = math.inf
        self.nodes_expanded = 0

    def _time_ok(self) -> bool:
        return time.perf_counter() < self.deadline

    def max_size(self, time_budget: float = 30.0, init_lb: int = 0):
        self.best_size = init_lb
        self.best_bits = 0
        self.nodes_expanded = 0
        self.start_time = time.perf_counter()
        self.deadline = self.start_time + time_budget
        try:
            P = (1 << self.n) - 1
            self._expand_max(0, 0, P)
            complete = True
        except TimeoutError:
            complete = False
        bits = self.best_bits
        return self.best_size, _bits_to_list(bits), complete

    def _expand_max(self, size: int, R_bits: int, P_bits: int):
        if not self._time_ok():
            raise TimeoutError
        if P_bits == 0:
            if size > self.best_size:
                self.best_size = size
                self.best_bits = R_bits
            return
        order, colors = _color_sort(P_bits, self.adj)
        P_local = P_bits
        for i in range(len(order) - 1, -1, -1):
            if size + colors[i] <= self.best_size:
                break
            v = order[i]
            v_bit = 1 << v
            if not (P_local & v_bit):
                continue
            self.nodes_expanded += 1
            R2 = R_bits | v_bit
            P2 = P_local & self.adj[v]
            if P2 == 0:
                if size + 1 > self.best_size:
                    self.best_size = size + 1
                    self.best_bits = R2
            else:
                self._expand_max(size + 1, R2, P2)
            P_local &= ~v_bit

    def enumerate_all_max(self, omega: int, time_budget: float = 30.0, cap: Optional[int] = None):
        self.nodes_expanded = 0
        self.start_time = time.perf_counter()
        self.deadline = self.start_time + time_budget
        out = []
        try:
            P = (1 << self.n) - 1
            self._expand_enum(0, 0, P, omega, out, cap)
            complete = True
        except TimeoutError:
            complete = False
        return out, complete, self.nodes_expanded

    def _expand_enum(self, size: int, R_bits: int, P_bits: int, target: int, out, cap):
        if not self._time_ok():
            raise TimeoutError
        if P_bits == 0:
            if size == target:
                out.append(_bits_to_list(R_bits))
            return
        order, colors = _color_sort(P_bits, self.adj)
        P_local = P_bits
        for i in range(len(order) - 1, -1, -1):
            if size + colors[i] < target:
                break
            v = order[i]
            v_bit = 1 << v
            if not (P_local & v_bit):
                continue
            self.nodes_expanded += 1
            R2 = R_bits | v_bit
            P2 = P_local & self.adj[v]
            if P2 == 0:
                if size + 1 == target:
                    out.append(_bits_to_list(R2))
                    if cap is not None and len(out) >= cap:
                        raise TimeoutError
            else:
                self._expand_enum(size + 1, R2, P2, target, out, cap)
            P_local &= ~v_bit
            if cap is not None and len(out) >= cap:
                raise TimeoutError

def _bits_to_list(bits: int):
    out = []
    x = bits
    while x:
        v = _lsb_index(x)
        out.append(v)
        x &= x - 1
    return out

def _greedy_lb(adj, n, trials=64):
    # degree-desc start vertices
    verts = sorted(range(n), key=lambda v: adj[v].bit_count(), reverse=True)
    starts = verts[:min(n, trials)]
    best_mask, best_sz = 0, 0
    for s in starts:
        C = 1 << s
        P = adj[s]
        while P:
            # pick v maximizing |N(v) ∩ P|
            tmp, best_v, best_sc = P, -1, -1
            while tmp:
                lsb = tmp & -tmp
                v = lsb.bit_length() - 1
                tmp ^= lsb
                sc = (adj[v] & P).bit_count()
                if sc > best_sc:
                    best_sc, best_v = sc, v
            C |= (1 << best_v)
            P &= adj[best_v]
        if C.bit_count() > best_sz:
            best_sz, best_mask = C.bit_count(), C
    return best_sz, best_mask

def solve_max_clique_all(n, edges, time_budget_sec=30.0, enum_cap=None, reorder=True):
    import time as _time
    if n <= 0:
        return {"omega": 0, "max_cliques": [[]], "complete": True, "runtime_sec": 0.0, "expanded_nodes": 0}
    t0 = _time.perf_counter()
    G = BitGraph.from_edges(n, edges)
    perm = list(range(n))
    invperm = list(range(n))
    if reorder:
        G2, perm, invperm = G.reorder_by_degeneracy()
    else:
        G2 = G
    solver = MaxCliqueSolver(G2)
    t_budget1 = max(0.1, time_budget_sec * 0.4)
    lb_sz, lb_mask = _greedy_lb(G2.adj, G2.n, trials=64)
    omega, witness_rel, complete1 = solver.max_size(time_budget=t_budget1, init_lb=lb_sz)
    if lb_sz > 0 and not witness_rel:
        witness_rel = [i for i in range(G2.n) if (lb_mask >> i) & 1]

    witness_abs = [invperm[v] for v in witness_rel]
    t_elapsed = _time.perf_counter() - t0
    t_remaining = max(0.0, time_budget_sec - t_elapsed)
    cliques_abs = []
    complete2 = True
    if t_remaining > 0 and omega > 0:
        cliques_rel, complete2, _ = solver.enumerate_all_max(omega, time_budget=t_remaining, cap=enum_cap)
        cliques_abs = [[invperm[v] for v in clique] for clique in cliques_rel]
        cliques_abs = sorted({tuple(sorted(c)) for c in cliques_abs})
    total_runtime = _time.perf_counter() - t0
    return {
        "omega": omega,
        "witness": sorted(witness_abs),
        "max_cliques": [list(c) for c in cliques_abs],
        "complete": bool(complete1 and complete2),
        "runtime_sec": total_runtime,
        "expanded_nodes": solver.nodes_expanded,
        "reordered": reorder,
    }

if __name__ == "__main__":
    # tiny demo
    n = 6
    edges = []
    def add_clique(nodes):
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                edges.append((nodes[i], nodes[j]))
    # add_clique([0,1,2,3])
    # add_clique([2,3,4,5])
    # Try to load sample data from ../sample.js
    import os
    import json
    try:
        sample_file_path = os.path.join(os.path.dirname(__file__), "..", "sample.js")
        with open(sample_file_path, 'r') as f:
            sample_data = json.load(f)
        
        if "adjacency_list" in sample_data:
            adjacency_list = sample_data["adjacency_list"]
            n = len(adjacency_list)
            
            # Convert adjacency list to edge list
            edges = []
            for u in range(n):
                for v in adjacency_list[u]:
                    if u < v:  # Avoid duplicate edges for undirected graph
                        edges.append((u, v))
            
            print(f"Loaded sample graph from ../sample.js: {n} vertices, {len(edges)} edges")
            # return n, edges
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Could not load sample data: {e}, falling back to random generation")
    
    print(solve_max_clique_all(n, edges, time_budget_sec=30.0))
