# DSATUR-COP-K-Means (COPKM_Plus)

A fast, solver-free algorithm for semi-supervised (constrained) clustering under hard **must-link (ML)** and **cannot-link (CL)** constraints.

> Described in **Appendix A.4** of:  
> **"Kempe Swap K-Means: A Scalable Near-Optimal Solution for Semi-Supervised Clustering"**  
> Yuxuan Ren, Shijie Deng — Georgia Institute of Technology  
> [arXiv:2603.27417](https://arxiv.org/abs/2603.27417)

---

## Overview

**DSATUR-COP-K-Means** (`COPKM_P`) is a constrained K-Means variant that replaces the naive data-order assignment of classic COP-K-Means with a **DSATUR-ordered, cost-aware** assignment step.  The key insight is that highly constrained data points — those with many cannot-linked neighbours already assigned — are the hardest to place feasibly, so they should be settled *first* while the most cluster colours are still available.

- **Must-link (ML):** two data points must be placed in the same cluster.  ML groups are automatically collapsed into super-nodes exploiting ML transitivity, reducing problem size.
- **Cannot-link (CL):** two data points must be placed in different clusters.  CL constraints between super-nodes induce a graph whose chromatic number determines feasibility.

### Comparison to Related Methods

| Method | Assignment order | Requires solver | Feasibility rate | WCSS quality |
|---|---|---|---|---|
| COP-K-Means (Wagstaff 2001) | Data index order | No | Low on dense CL graphs | Moderate |
| **DSATUR-COP-K-Means (ours)** | **Saturation-first** | **No** | **High** | **Good** |
| KSKM (Ren & Deng 2026) | MWIS-optimal | Gurobi | Very high | Near-optimal |

DSATUR-COP-K-Means sits between classic COP-K-Means and KSKM: it needs no external solver, runs in polynomial time, achieves much higher feasibility rates, and produces substantially better WCSS — making it an excellent baseline or fast initialiser for KSKM.

---

## Algorithm

### Core Idea (Algorithm 7, Appendix A.4)

Like standard K-Means, the algorithm alternates between two steps:

1. **Assignment step (DSATUR_KM):** Super-nodes are processed in DSATUR order — the most saturated super-node (most distinct cannot-link-neighbour colours already assigned) is picked next.  Each super-node is assigned to the **nearest feasible centroid**, i.e. the closest cluster colour not yet used by any cannot-link neighbour.  If no feasible colour exists, the algorithm reports infeasibility.

2. **Centroid update step (CentroidUpdate):** Cluster centroids are recomputed as the mean of all assigned points.  Empty clusters are reseeded using k-means++ sampling.

The loop repeats until WCSS no longer improves, yielding a local optimum.

### Why DSATUR Order Helps

Classic COP-K-Means processes points in their natural index order.  When it reaches a highly constrained point late in the sequence, most of its cannot-link neighbours may already be assigned, potentially exhausting all k colours.  DSATUR order avoids this by settling the most constrained points while the assignment is still flexible.  This **significantly reduces infeasibility** and tends to find lower WCSS solutions.

### Component Decomposition

Before the main loop, the cannot-link graph is decomposed into connected components and classified:
- **Singletons** — unconstrained super-nodes assigned directly to the nearest centroid.
- **Cliques** — fully connected cannot-link groups solved by the Hungarian algorithm (linear sum assignment), guaranteeing one super-node per cluster.
- **General components** — handled by the Numba-JIT DSATUR routine `dsatur_init_numba`.

### Initialisation

Centroids are seeded with **k-means++** (scikit-learn) for well-spread initial positions, which is critical for the quality of the DSATUR assignment in the first iteration.

---

## Installation

### Requirements

```
numpy
scipy
scikit-learn
numba
```

No Gurobi or any other ILP solver is required.

```bash
pip install numpy scipy scikit-learn numba
```

---

## Usage

```python
import numpy as np
from COPKM_plus import COPKM_P

# Data to cluster
data = np.array([...])  # shape (n, d)

# Pairwise constraints (0-indexed data-point indices)
ml = [(2, 6), (9, 10)]   # must-link pairs
cl = [(2, 7), (6, 7)]    # cannot-link pairs

# Run DSATUR-COP-K-Means
membership = COPKM_P(
    random_state=42,
    data=data,
    ml=ml,
    cl=cl,
    k=4,         # number of clusters
)
# membership: cluster label for each data point (0-indexed), or [] if infeasible
```

See [`example.py`](example.py) for a complete runnable example with ARI evaluation.

---

## API Reference

### `COPKM_P(random_state, data, ml, cl, k, verbose=False)`

Main entry point. Runs DSATUR-COP-K-Means until WCSS convergence.

| Parameter | Type | Description |
|---|---|---|
| `random_state` | `int` | NumPy random seed for k-means++ initialisation |
| `data` | `ndarray (n, d)` | Data matrix |
| `ml` | `list[tuple[int,int]]` | Pairwise must-link constraints (0-indexed data points) |
| `cl` | `list[tuple[int,int]]` | Pairwise cannot-link constraints (0-indexed data points) |
| `k` | `int` | Number of clusters |
| `verbose` | `bool` | Reserved for future logging (currently unused) |

**Returns:** `ndarray (n,)` — cluster label per data point (0-indexed), or `[]` if the cannot-link graph's chromatic number exceeds k (no feasible k-colouring exists).

---

### `COPKM(random_state, data, ml, cl, k, verbose=False, time_limit=3600)`

Classic COP-K-Means baseline (data-index-order assignment).  Provided for direct comparison.  Generally slower to converge and more prone to infeasibility than `COPKM_P` on dense constraint graphs.

---

## Key Internal Components

| Function | Role |
|---|---|
| `preprocessing` | Collapses ML pairs into super-nodes via connected components; lifts CL pairs to super-node space; aggregates per-super-node data sums |
| `compute_sums_numba` | Numba-JIT aggregation of coordinate sums and squared-norm sums per super-node |
| `sub_adj_classification` | BFS-based decomposition of the CL graph into singletons, cliques, and general components; pre-slices data arrays per component |
| `DSATUR_KM` | One full DSATUR assignment pass across all components, dispatching to the appropriate routine per component type |
| `dsatur_init_numba` | Numba-JIT cost-aware DSATUR colouring for a single general component: saturation-first ordering, nearest-feasible-centroid assignment |
| `InitAssign_Singletons_and_Cliques` | Vectorised assignment for singletons; Hungarian-algorithm assignment for cliques |
| `CentroidUpdate` | Recomputes centroids from current assignment; reseeds empty clusters via k-means++ |
| `CentroidUpdate_assist` | Numba-JIT inner accumulation loop for CentroidUpdate |
| `distance_matrix` | Vectorised WCSS distance D[v, j] using pre-aggregated super-node sums |
| `kmeans_plusplus_init` | D²-weighted sampling to seed centroids for empty clusters |
| `COPKM_DSATUR` | Inner convergence loop: iterates DSATUR_KM + CentroidUpdate until WCSS stops improving |
| `COPKM_assignmen` | Sequential greedy assignment (data-index order) used by the classic COPKM baseline |
| `recover_ml_from_membership` | Expands super-node assignments back to individual data-point labels |

---

## Performance

On the benchmark datasets from the paper, `COPKM_P` consistently achieves:
- **Near-100% feasibility** across all constraint levels, versus 40–87% for classic COP-K-Means.
- **Substantially lower WCSS** than classic COP-K-Means, with runtimes that are orders of magnitude faster than ILP-based methods (PCCC, BLPKMCC) and KSKM.
- A reliable, high-quality starting solution for KSKM when further optimisation is needed.

---

## Citation

```bibtex
@article{ren2026kskm,
  title   = {Kempe Swap K-Means: A Scalable Near-Optimal Solution for Semi-Supervised Clustering},
  author  = {Ren, Yuxuan and Deng, Shijie},
  journal = {arXiv preprint arXiv:2603.27417},
  year    = {2026}
}
```
