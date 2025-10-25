"""
MOBBO based on Simon (2008) core operators, extended to multi-objective (ZDT suite).
- Non-dominated sorting + crowding distance (NSGA-II) determine ordering.
- Ordering -> lambda, mu (as in Simon) -> migration & species dynamics -> mutation.
- Archive of non-dominated solutions preserved.
Author: (adaptado para tu TFG)
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt

# ----------------------------
# ZDT functions (ZDT1, ZDT2, ZDT3, ZDT4, ZDT6)
# ----------------------------
def zdt1(x):
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x)-1)
    f2 = g * (1.0 - math.sqrt(f1 / g))
    return np.array([f1, f2])

def zdt2(x):
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x)-1)
    f2 = g * (1.0 - (f1 / g)**2)
    return np.array([f1, f2])

def zdt3(x):
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x)-1)
    f2 = g * (1.0 - math.sqrt(f1/g) - (f1/g) * math.sin(10*math.pi*f1))
    return np.array([f1, f2])

def zdt4(x):
    f1 = x[0]
    xi = x[1:]
    g = 1.0 + 10*(len(x)-1) + np.sum(xi**2 - 10*np.cos(4*np.pi*xi))
    f2 = g * (1.0 - math.sqrt(f1/g))
    return np.array([f1, f2])

def zdt6(x):
    f1 = 1 - math.exp(-4*x[0]) * (math.sin(6*math.pi*x[0]))**6
    g = 1.0 + 9.0 * (np.sum(x[1:])/(len(x)-1))**0.25
    f2 = g * (1.0 - (f1/g)**2)
    return np.array([f1, f2])

PROBLEMS = {
    'ZDT1': zdt1,
    'ZDT2': zdt2,
    'ZDT3': zdt3,
    'ZDT4': zdt4,
    'ZDT6': zdt6
}

# ----------------------------
# Utilities: dominance, nondominated sort, crowding
# ----------------------------
def dominates(a, b):
    """Return True if objective vector a dominates b (minimization)."""
    return np.all(a <= b) and np.any(a < b)

def nondominated_sort(pop_objs):
    """
    Fast but simple non-dominated sorting.
    Input: pop_objs: (N, M) array of objectives (minimization).
    Returns: fronts: list of lists (indices). front 0 = best nondominated.
    Also returns front_index list mapping idx->front_number.
    """
    N = pop_objs.shape[0]
    S = [set() for _ in range(N)]
    n = np.zeros(N, dtype=int)
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(pop_objs[p], pop_objs[q]):
                S[p].add(q)
            elif dominates(pop_objs[q], pop_objs[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in list(S[p]):
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    # drop empty last
    if not fronts[-1]:
        fronts.pop()
    # build front_index
    front_index = np.full(N, -1, dtype=int)
    for fi, front in enumerate(fronts):
        for idx in front:
            front_index[idx] = fi
    return fronts, front_index

def crowding_distance(pop_objs, front):
    """
    Crowding distance for individuals in a front (list of indices).
    Returns distances array of same length as front (order matched).
    """
    if len(front) == 0:
        return np.array([])
    objs = pop_objs[front]
    num_obj = objs.shape[1]
    dist = np.zeros(len(front))
    for m in range(num_obj):
        sorted_idx = np.argsort(objs[:, m])
        fmin = objs[sorted_idx[0], m]
        fmax = objs[sorted_idx[-1], m]
        # Set boundary distances to inf to prefer them
        dist[sorted_idx[0]] = np.inf
        dist[sorted_idx[-1]] = np.inf
        if fmax - fmin == 0.0:
            continue
        for k in range(1, len(front)-1):
            prev_val = objs[sorted_idx[k-1], m]
            next_val = objs[sorted_idx[k+1], m]
            dist[sorted_idx[k]] += (next_val - prev_val) / (fmax - fmin)
    return dist

# ----------------------------
# MOBBO: Simon 2008 core + multiobjective ordering
# ----------------------------
class MOBBO_Simon:
    def __init__(self, obj_func, dim=30, lb=0.0, ub=1.0,
                 pop_size=100, max_gen=250, elite_count=2,
                 I=1.0, E=1.0, m_max=0.01, archive_size=200, seed=None):
        """
        obj_func: function(x) -> np.array([f1,f2])
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.obj_func = obj_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.elite_count = elite_count
        self.I = I
        self.E = E
        self.m_max = m_max
        self.archive_size = archive_size

        # initialize population uniformly
        self.pop = np.random.uniform(lb, ub, (pop_size, dim))
        self.pop_objs = np.array([self.obj_func(ind) for ind in self.pop])
        # initial species probabilities (uniform)
        self.species_prob = np.ones(pop_size) / pop_size
        # archive of nondominated solutions as list of tuples (x, objs)
        self.archive = []

    # ---- archive utilities ----
    def update_archive(self):
        """Merge population into archive and keep non-dominated set (bounded)."""
        # combine
        cand = [(tuple(ind), tuple(obs)) for ind, obs in zip(self.pop, self.pop_objs)]
        old = [(tuple(x), tuple(o)) for x,o in self.archive]
        combined = old + cand
        # unique by decision vector (keeps duplicates)
        # evaluate nondominated among combined
        decs = np.array([np.array(o[1]) for o in combined])
        keep = []
        for i in range(len(combined)):
            dominated_flag = False
            for j in range(len(combined)):
                if i == j:
                    continue
                if dominates(decs[j], decs[i]):
                    dominated_flag = True
                    break
            if not dominated_flag:
                keep.append(combined[i])
        # reduce to archive_size by crowding on objectives if needed
        # build arrays
        objs_keep = np.array([np.array(o[1]) for o in keep])
        if len(keep) <= self.archive_size:
            self.archive = [(np.array(k[0]), np.array(k[1])) for k in keep]
        else:
            # Use non-dominated sorting (all are non-dominated actually) -> compute crowding
            # We'll simply compute crowding across the set and sort descending
            # Create artificial front indices
            front = list(range(len(keep)))
            cd = crowding_distance(objs_keep, front)
            # indices of keep sorted by cd descending (prefer larger crowding)
            order = np.argsort(-cd)
            selected = [keep[i] for i in order[:self.archive_size]]
            self.archive = [(np.array(k[0]), np.array(k[1])) for k in selected]

    # ---- main run ----
    def run(self, verbose=True):
        history = []
        for gen in range(self.max_gen):
            # 1) sort by pareto fronts and compute crowding
            fronts, front_index = nondominated_sort(self.pop_objs)
            # compute crowding distance for all individuals (we will convert to per-index array)
            crowding = np.zeros(self.pop_size)
            for front in fronts:
                cd = crowding_distance(self.pop_objs, front)
                # map back
                for i, idx in enumerate(front):
                    crowding[idx] = cd[i]
            # 2) Build ordering (pos) using (front, -crowding) lexicographic sort
            # create array of tuples then argsort
            order_keys = [(front_index[i], -crowding[i]) for i in range(self.pop_size)]
            # argsort by key
            ordered_idx = sorted(range(self.pop_size), key=lambda i: order_keys[i])
            # pos_map: position (0 best ... N-1 worst) used for lambda/mu
            pos_map = np.empty(self.pop_size, dtype=int)
            for pos, idx in enumerate(ordered_idx):
                pos_map[idx] = pos

            # 3) compute lambda and mu as Simon 2008 from pos
            ranks = pos_map  # 0..N-1
            lam = self.I * (1.0 - ranks / (self.pop_size - 1.0))
            mu  = self.E * (ranks / (self.pop_size - 1.0))

            # 4) migration (apply to non-elitist individuals)
            new_pop = self.pop.copy()
            for i in range(self.elite_count, self.pop_size):
                # with probability lambda attempt to immigrate
                if np.random.rand() < lam[i]:
                    # donor selection proportional to mu (emigration)
                    mu_sum = np.sum(mu)
                    # avoid zero division
                    if mu_sum <= 0:
                        donor_idx = np.random.randint(self.pop_size)
                    else:
                        probs = mu / mu_sum
                        # ensure donor != i; if selected equal, pick random other
                        donor_idx = np.random.choice(self.pop_size, p=probs)
                        if donor_idx == i:
                            cand = list(range(self.pop_size))
                            cand.remove(i)
                            donor_idx = random.choice(cand)
                    # random SIV index to copy (single gene copy like Simon)
                    gene = np.random.randint(self.dim)
                    new_pop[i, gene] = self.pop[donor_idx, gene]

            # 5) update population after migration
            self.pop = new_pop
            self.pop_objs = np.array([self.obj_func(ind) for ind in self.pop])

            # 6) species probability dynamics discrete approximation (Simon ODE)
            new_species_prob = np.zeros_like(self.species_prob)
            for s in range(self.pop_size):
                if s == 0:
                    new_species_prob[s] = -(lam[s] + mu[s]) * self.species_prob[s]
                    if s+1 < self.pop_size:
                        new_species_prob[s] += mu[s+1] * self.species_prob[s+1]
                elif s == self.pop_size - 1:
                    new_species_prob[s] = -(lam[s] + mu[s]) * self.species_prob[s]
                    new_species_prob[s] += lam[s-1] * self.species_prob[s-1]
                else:
                    new_species_prob[s] = -(lam[s] + mu[s]) * self.species_prob[s] \
                                          + lam[s-1] * self.species_prob[s-1] \
                                          + mu[s+1] * self.species_prob[s+1]
            # ensure non-negative and normalize
            new_species_prob = np.maximum(new_species_prob, 0.0)
            if new_species_prob.sum() <= 0:
                # fallback to uniform
                new_species_prob = np.ones(self.pop_size) / self.pop_size
            else:
                new_species_prob /= new_species_prob.sum()
            self.species_prob = new_species_prob

            # 7) mutation based on species probability (non-elitists)
            for i in range(self.elite_count, self.pop_size):
                m_i = self.m_max * (1.0 - self.species_prob[i])  # good habitats mutate less
                for d in range(self.dim):
                    if np.random.rand() < m_i:
                        self.pop[i,d] = np.random.uniform(self.lb, self.ub)

            # 8) re-evaluate after mutation
            self.pop_objs = np.array([self.obj_func(ind) for ind in self.pop])

            # 9) update archive
            self.update_archive()

            # record some stats
            if gen % 10 == 0 or gen == self.max_gen - 1:
                # approximate archive size and first front size
                first_front_size = len(fronts[0]) if fronts else 0
                if verbose:
                    print(f"Gen {gen:4d} | archive {len(self.archive):3d} | front1 {first_front_size:3d}")

            history.append((gen, len(self.archive)))

        return self.archive, history

# ----------------------------
# Simple plotting for 2-objective problems
# ----------------------------
def plot_pareto(archive, true_front=None, title='Pareto front'):
    objs = np.array([o for (x,o) in archive])
    plt.figure(figsize=(6,5))
    plt.scatter(objs[:,0], objs[:,1], label='Approx Pareto', s=20)
    if true_front is not None:
        tf = np.array(true_front)
        plt.plot(tf[:,0], tf[:,1], 'r--', label='True front')
    plt.xlabel('f1'); plt.ylabel('f2'); plt.title(title)
    plt.legend(); plt.grid(True)
    plt.show()

# ----------------------------
# Example of execution for a ZDT problem
# ----------------------------
if __name__ == "__main__":
    # configuration
    problem_name = 'ZDT1'   # choose ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
    func = PROBLEMS[problem_name]
    dim = 30
    pop_size = 100
    gen = 250
    seed = 42

    mobbo = MOBBO_Simon(obj_func=func, dim=dim, lb=0.0, ub=1.0,
                       pop_size=pop_size, max_gen=gen, elite_count=2,
                       I=1.0, E=1.0, m_max=0.02, archive_size=300, seed=seed)
    archive, history = mobbo.run(verbose=True)

    # plot results (archive)
    plot_pareto(archive, title=f"MOBBO-Simon archive - {problem_name}")

    # print some archive examples
    print("\nArchive sample (first 10):")
    for i, (x,o) in enumerate(archive[:10]):
        print(f"{i+1:2d}: f = {o}")
