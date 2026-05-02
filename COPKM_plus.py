from collections import deque, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import kmeans_plusplus
import time
import copy
from numba.typed import List
from numba import njit
from typing import Dict

def initiate_membership(n_v_ml):
    """Return a zero-initialised cluster-assignment array of length n_v_ml (int32)."""
    return np.zeros(n_v_ml, dtype = np.int32)

def f2(C):
    """Compute squared L2 norm of each row of C; returns shape (k,) from input (k, d)."""
    C2 = np.einsum('ij,ij->i', C, C)
    return C2

def InitAssign_Singletons_and_Cliques(membership, sub_adjs, C_T, C2):
    """
    Assign singleton and clique components to their optimal clusters during initialisation.

    Singleton super-nodes are each assigned to the nearest centroid (argmin over D).
    Clique components (fully-connected cannot-link groups) are assigned by solving a
    linear sum assignment problem (Hungarian method) so every vertex in the clique lands
    in a distinct cluster while minimising total WCSS.

    Args:
        membership (int32[n_supernodes]): Cluster assignment array, modified in-place.
        sub_adjs (dict): Component classification produced by sub_adj_classification.
        C_T (float64[d, k]): Transposed centroid matrix.
        C2 (float64[k]): Squared L2 norms of each centroid row.

    Returns:
        membership (int32[n_supernodes]): Updated cluster assignments.
    """
    verts, Y, Y2, ml_count_at_Y = sub_adjs['singletons']
    D = distance_matrix(Y, Y2, C_T, C2, ml_count_at_Y)
    best_j = np.argmin(D, axis=1)
    for v, j_new in zip(verts, best_j):
        membership[v] = j_new

    for verts, Y, Y2, ml_count_at_Y in sub_adjs['cliques']:
        D = distance_matrix(Y, Y2, C_T, C2, ml_count_at_Y)
        best_j = assignment_problem_scipy(D)
        for v, j_new in zip(verts, best_j):
            membership[v] = j_new
    return membership

def assignment_problem_scipy(c):
    """
    c: (n, m) cost matrix
    Constraint: each row assigned to exactly one col,
                each col used at most once.
    This is a rectangular assignment; Hungarian works directly.
    """
    # Run Hungarian on the full matrix
    _, col_ind = linear_sum_assignment(c)

    return col_ind


def distance_matrix(Y, Y2, C_T, C2, ml_count_ay_Y):
    """
    Compute the WCSS contribution matrix D[v, j] for each super-node v and cluster j.

    For super-node v containing ml_count[v] data points with coordinate sum Y[v] and
    sum-of-squared-norms Y2[v], the contribution to WCSS when assigned to centroid µj is:
        D[v, j] = ml_count[v] * ||µj||² - 2 * Y[v] @ µj + Y2[v]

    This equals the total squared distance of all points in v from µj.

    Args:
        Y (float64[n, d]): Per-super-node coordinate sums.
        Y2 (float64[n]): Per-super-node sum of squared norms.
        C_T (float64[d, k]): Transposed centroid matrix.
        C2 (float64[k]): Squared L2 norms of each centroid.
        ml_count_ay_Y (int[n]): Number of data points in each super-node.

    Returns:
        D (float64[n, k]): WCSS distance matrix.
    """
    D = np.outer(ml_count_ay_Y, C2)
    D -= 2.0 * (Y @ C_T)
    D += Y2[:, None]
    return D


def kmeans_plusplus_init(empty_clusters, y_values, y_values2, C, C2, n_v_ml, ml_count):
    """
    Seed empty cluster centroids using a k-means++ sampling strategy.

    For each empty cluster, samples a new centroid proportional to the squared distance
    of each super-node from its nearest already-chosen centroid (D²-weighting), then
    updates the running minimum distance array so subsequent empty clusters are sampled
    from the remaining under-represented regions.

    Args:
        empty_clusters (int[:]): Indices of clusters that have no assigned super-nodes.
        y_values (float64[n, d]): Per-super-node coordinate sums.
        y_values2 (float64[n]): Per-super-node sum of squared norms.
        C (float64[k, d]): Full centroid matrix, updated in-place for empty clusters.
        C2 (float64[k]): Squared L2 norms of centroids, updated in-place.
        n_v_ml (int): Total number of super-nodes.
        ml_count (int[n]): Number of data points in each super-node.

    Returns:
        C (float64[k, d]), C2 (float64[k]): Updated centroid matrix and norms.
    """

    # Centroids: (k', d), Centroids2: (k',)
    Centroids  = np.delete(C,  empty_clusters, axis=0)   # axis=0, centroids are rows
    Centroids2 = np.delete(C2, empty_clusters)           # 1D, so axis not needed
    Centroids_T = Centroids.T

    # D: (n, k') distances to currently non-empty centroids
    D = distance_matrix(y_values, y_values2, Centroids_T, Centroids2, ml_count)  # (n, k')
    D2 = np.min(D, axis=1) / ml_count  # (n,) min distance to any existing centroid
    D2 = np.maximum(D2, 0.0)


    # Step 2: choose remaining centroids for empty_clusters
    for j in empty_clusters:
        # Sample next centroid using probability ∝ D2
        probs = D2 / D2.sum()        # (n,)
        next_idx = np.random.choice(n_v_ml, p=probs)
        
        # New centroid from y_values
        c  = y_values[next_idx]             # (d,)  — one data point as 1D row
        c2 = np.dot(c, c)            # scalar = ||c||^2

        # Write back into full centroid arrays
        C[j]  = c                    # C[j]: (d,)
        C2[j] = c2                   # C2[j]: scalar

        # Distances to JUST this new centroid
        # c[None, :] -> (1, d), np.array([c2]) -> (1,)
        # distance_matrix(...): (n, 1) -> ravel() to (n,)
        b = distance_matrix(y_values, y_values2, c[:, None], np.array([c2]), ml_count).ravel()  # (n,)
        b = np.maximum(b, 0.0) / ml_count 
        # Update min squared distance to ANY chosen centroid so far
        D2 = np.minimum(D2, b)       # (n,)

    return C, C2

@njit(cache=True)
def CentroidUpdate_assist(membership, y_values, y_values2, k, ml_count, n_v_ml, d):
    """
    Numba-JIT inner loop: accumulate per-cluster coordinate sums, squared-norm sums,
    and point counts weighted by super-node size (ml_count).

    Args:
        membership (int32[n_supernodes]): Current cluster assignment.
        y_values (float64[n, d]): Per-super-node coordinate sums.
        y_values2 (float64[n]): Per-super-node sum of squared norms.
        k (int): Number of clusters.
        ml_count (int[n]): Super-node sizes.
        n_v_ml (int): Number of super-nodes.
        d (int): Feature dimensionality.

    Returns:
        sums (float64[k, d]), sum_sq (float64[k]), counts (int64[k]).
    """
    sums   = np.zeros((k, d), dtype=y_values.dtype)
    sum_sq = np.zeros((k,),   dtype=y_values2.dtype)
    counts = np.zeros((k,),   dtype=np.int64)

    for i in range(n_v_ml):
        c = membership[i]
        counts[c] += ml_count[i]
        sum_sq[c] += y_values2[i]
        for j in range(d):
            sums[c, j] += y_values[i, j]

    return sums, sum_sq, counts

def CentroidUpdate(membership, y_values, y_values2, k, ml_count, n_v_ml, d):
    """
    Recompute cluster centroids from the current assignment and return the total WCSS.

    For each non-empty cluster j, the centroid is µj = (sum of points in j) / count_j.
    Empty clusters are reseeded via kmeans_plusplus_init so every cluster always has a
    valid centroid for the next assignment step.  Total WCSS is returned rounded to
    5 decimal places.

    Args:
        membership (int32[n_supernodes]): Current cluster assignment.
        y_values (float64[n, d]): Per-super-node coordinate sums.
        y_values2 (float64[n]): Per-super-node sum of squared norms.
        k (int): Number of clusters.
        ml_count (int[n]): Super-node sizes.
        n_v_ml (int): Number of super-nodes.
        d (int): Feature dimensionality.

    Returns:
        C_T (float64[d, k]): Transposed centroid matrix.
        C2 (float64[k]): Squared L2 norms of each centroid.
        total_ssr (float): Total within-cluster sum of squares (rounded to 5 d.p.).
    """

    C  = np.empty((k, d), dtype=y_values.dtype)
    C2 = np.empty((k,),   dtype=y_values.dtype)

    sums, sum_sq, counts = CentroidUpdate_assist(membership, y_values, y_values2, k, ml_count, n_v_ml, d)

    nonempty       = counts > 0
    empty_clusters = np.nonzero(~nonempty)[0]

    C[nonempty]  = sums[nonempty] / counts[nonempty, None]
    C2[nonempty] = f2(C[nonempty])

    if empty_clusters.size > 0:
        C, C2 = kmeans_plusplus_init(empty_clusters, y_values, y_values2, C, C2, n_v_ml, ml_count)

    ssr_nonempty = sum_sq[nonempty] - counts[nonempty] * C2[nonempty]
    total_ssr    = float(ssr_nonempty.sum())

    C_T = C.T
    return C_T, C2, round(total_ssr,5)


def choose_initial_cluster_centers(data, k, random_state):
    """
    Select k initial cluster centroids using the k-means++ algorithm.

    Args:
        data (float64[n, d]): Raw data matrix.
        k (int): Number of clusters.
        random_state (int): Random seed passed to sklearn's kmeans_plusplus.

    Returns:
        C (float64[k, d]): Initial centroid matrix.
    """
    C, _ = kmeans_plusplus(data, k, random_state = random_state)
    return C

@njit(cache=True)
def dsatur_init_numba(membership, vertices, neighbors, D, k, n):
    """
    Cost-aware DSATUR graph colouring for one connected component (Numba-JIT).

    Implements the improved DSATUR assignment rule from Algorithm 5 of the paper:
    super-nodes are processed in non-increasing order of saturation degree (number of
    distinct cluster colours already assigned to their cannot-link neighbours), with
    ties broken by degree then vertex id.  Each selected super-node is assigned to the
    cheapest feasible cluster — the one with the smallest D[v, c] among all colours not
    yet used by any neighbour — rather than the smallest colour index used by classic
    DSATUR.  This produces a feasible k-colouring that also approximates nearest-centroid
    assignment, giving a better starting WCSS than classic DSATUR.

    Returns (membership, 0) immediately if no feasible colour exists for some vertex,
    signalling that k is smaller than the component's chromatic number.

    Args:
        membership (int32[n_supernodes]): Global assignment array, written back on success.
        vertices (int[n]): Global super-node ids for each local vertex index.
        neighbors (List[int[:]]): Cannot-link adjacency list in local vertex indices.
        D (float64[n, k]): WCSS distance of each local vertex to each cluster.
        k (int): Number of clusters (colours).
        n (int): Number of vertices in this component.

    Returns:
        membership (int32[n_supernodes]), success (int): 1 if a feasible colouring was
            found, 0 if infeasible (chromatic number > k).
    """

    # colors[v] = color assigned to local vertex v, -1 = uncolored
    colors = np.full(n, -1, np.int64)

    # degree[v] = len(neighbors[v])
    degree = np.empty(n, np.int64)
    for i in range(n):
        degree[i] = len(neighbors[i])

    # neigh_colors[v, c] = 1 if some neighbor of v has color c
    neigh_colors = np.zeros((n, k), np.uint8)

    # cost_order[v, :] = colors in increasing D[v, c] order, with sentinel k at the end
    cost_order = np.empty((n, k + 1), np.int64)
    for i in range(n):
        # argsort over colors for vertex i
        idx = np.argsort(D[i])
        for j in range(k):
            cost_order[i, j] = idx[j]
        cost_order[i, k] = k  # sentinel (means "no color left")

    n_colored = 0

    while n_colored < n:
        # --- pick vertex v with max saturation, then max degree, then max id ---
        best_v = -1
        best_sat = -1
        best_deg = -1

        for u in range(n):
            if colors[u] != -1:
                continue  # already colored

            # compute saturation = number of distinct neighbor colors
            sat = 0
            for c in range(k):
                if neigh_colors[u, c] != 0:
                    sat += 1

            if (sat > best_sat or
                (sat == best_sat and
                 (degree[u] > best_deg or
                  (degree[u] == best_deg and u > best_v)))):
                best_sat = sat
                best_deg = degree[u]
                best_v = u

        v = best_v
        if v == -1:
            # should not happen, but just in case
            return membership, 0

        # --- pick cheapest color for v not used by its neighbors ---
        used_row = neigh_colors[v]
        chosen_color = -1

        for pos in range(k + 1):
            c = cost_order[v, pos]
            if c == k:  # sentinel = "no available color"
                return membership, 0
            if used_row[c] == 0:
                chosen_color = c
                break

        colors[v] = chosen_color
        n_colored += 1

        # --- update neighbors' saturation info ---
        neighs_v = neighbors[v]
        for t in range(len(neighs_v)):
            w = neighs_v[t]
            if colors[w] == -1:
                neigh_colors[w, chosen_color] = 1

    # write back to global membership using vertices mapping
    for v in range(n):
        membership[vertices[v]] = colors[v]

    return membership, 1

def DSATUR_KM(n_v_ml, sub_adjs, C_T, C2, k):
    """
    Run one full pass of cost-aware DSATUR assignment across all constraint-graph components.

    Dispatches to InitAssign_Singletons_and_Cliques for singleton and clique components,
    and to dsatur_init_numba for general (non-clique) components.  Returns an empty list
    immediately if any component cannot be feasibly coloured with k colours.

    Args:
        n_v_ml (int): Total number of super-nodes.
        sub_adjs (dict): Component classification from sub_adj_classification.
        C_T (float64[d, k]): Transposed centroid matrix.
        C2 (float64[k]): Squared L2 norms of each centroid.
        k (int): Number of clusters.

    Returns:
        membership (int32[n_supernodes]): Cluster assignments, or [] if infeasible.
    """
    membership = initiate_membership(n_v_ml)
    membership = InitAssign_Singletons_and_Cliques(membership, sub_adjs, C_T, C2)

    for vertices, _, Y, Y2, neighbors, n, ml_count_at_Y in sub_adjs['others']:
        D = distance_matrix(Y, Y2, C_T, C2, ml_count_at_Y)
        membership, fitted = dsatur_init_numba(membership, vertices, neighbors, D, k, n)
        if not fitted:
            return []

    return membership


def sub_adj_classification(adj, y_values, y_values2, ml_count):
    """
    Classify each connected component of the cannot-link graph and attach data slices.

    Performs a BFS from each unvisited super-node to identify its connected component,
    then categorises it as:
      - Singleton (size 1): collected into a single batch for vectorised assignment.
      - Clique (every vertex connected to every other): assigned by linear sum assignment.
      - General (everything else): passed to the Numba DSATUR routine.

    Pre-sliced data arrays (Y, Y2, ml_count_at_Y) and local adjacency representations
    are stored alongside each component to avoid repeated indexing inside the hot loop.

    Args:
        adj (dict[int, set]): Cannot-link adjacency list over super-node indices.
        y_values (float64[n, d]): Per-super-node coordinate sums.
        y_values2 (float64[n]): Per-super-node sum of squared norms.
        ml_count (int[n]): Super-node sizes.

    Returns:
        sub_adjs (dict): Keys 'singletons', 'cliques', 'others' with typed entries
            ready for consumption by DSATUR_KM / InitAssign_Singletons_and_Cliques.
    """
    sub_adjs = {'cliques': [], 'others': []}
    singletons = []
    visited = set()
    for s in adj:
        n_vertices = 0
        is_clique = True
        nodes_map = defaultdict()
        sub_adj = defaultdict(set)
        if  s in visited:
            continue
        vertices = []
        q = deque([s])
        visited.add(s)
        while q:
            u = q.popleft()
            vertices.append(u)
            nodes_map[u] = n_vertices
            n_vertices += 1
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        for u in vertices:
            sub_adj[nodes_map[u]] = {nodes_map[v] for v in adj[u] if v in nodes_map}
            if len(adj[u]) < n_vertices - 1:
                is_clique = False

        if n_vertices == 1:
            singletons += vertices
            
        elif is_clique:
            # reuse data
            Y = y_values[vertices]        # shape (m, d)
            ml_count_at_Y = ml_count[vertices]
            verts = np.asarray(vertices, dtype=np.int32) # shape (m,)
            Y2 = y_values2[vertices]      # (m,)
            sub_adjs['cliques'].append((verts, Y, Y2, ml_count_at_Y))
        else:
            Y  = y_values[vertices]                 # (m, d)
            Y2 = y_values2[vertices]                # (m,)
            ml_count_at_Y = ml_count[vertices]
            neighbors = List()
            for v, neighs in sub_adj.items():
                neigh_idx = np.fromiter(neighs, dtype=int)
                neighbors.append(neigh_idx)
            sub_adjs['others'].append((vertices, sub_adj, Y, Y2, neighbors, n_vertices, ml_count_at_Y))

    Y = y_values[singletons]      # shape (m, d)
    ml_count_at_Y = ml_count[singletons]
    verts = np.asarray(singletons, dtype=np.int32) # shape (m,)
    Y2 = y_values2[singletons]         # (m,)
    sub_adjs['singletons'] = (verts, Y, Y2, ml_count_at_Y)
    return sub_adjs

def preprocessing(data, cl, ml):
    """
    Convert pairwise ML/CL constraint lists into super-node data structures.

    Steps:
      1. Build a must-link adjacency list and find its connected components; each
         component becomes one super-node.
      2. Construct ml_map (data point → super-node index) and ml_count (super-node size).
      3. Lift cannot-link pairs from data-point space to super-node space, building the
         cannot-link adjacency list adj over super-nodes.
      4. Aggregate per-super-node coordinate sums (y_values) and sum-of-squared-norms
         (y_values2) via compute_sums_numba.

    Args:
        data (float64[n, d]): Raw data matrix.
        cl (list[tuple[int, int]]): Pairwise cannot-link constraints (0-indexed data points).
        ml (list[tuple[int, int]]): Pairwise must-link constraints (0-indexed data points).

    Returns:
        y_values (float64[n_supernodes, d]): Per-super-node coordinate sums.
        y_values2 (float64[n_supernodes]): Per-super-node sum of squared norms.
        adj (dict[int, set]): Cannot-link adjacency list over super-nodes.
        ml_map (int[n]): Maps each data-point index to its super-node index.
        ml_count (int[n_supernodes]): Number of data points in each super-node.
        n_v_ml (int): Total number of super-nodes.
        d (int): Feature dimensionality.
    """
    def adj_from_list(clml, len_data):
        adj = {i: [] for i in range(len_data)}
        for u, v in clml:
            # For undirected graph
            adj[u].append(v)
            adj[v].append(u)
        return adj
    def connected_components(adj):
        """
        adj: dict {u: iterable of neighbors}, undirected
        returns: list of sets, each set is a component
        """
        visited = set()
        comps = List()
        for s in adj:
            if s in visited:
                continue
            comp = List()
            q = deque([s])
            visited.add(s)
            while q:
                u = q.popleft()
                comp.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            comps.append(comp)
        return comps
    def adj_from_cl_ml(cl, ml, len_data):
        ml_adj = adj_from_list(ml, len_data) 
        ml_list_numba = connected_components(ml_adj)
        n_v_ml = len(ml_list_numba)
        ml_count = np.zeros(n_v_ml, dtype=int)
        ml_map = np.zeros(len_data, dtype=int)
        for i, v_ml in enumerate(ml_list_numba):
            ml_count[i] = len(v_ml)
            for v in v_ml:
                ml_map[v] = i
        adj = {i: set() for i in range(n_v_ml)}
        for u, v in cl:
            uu = ml_map[u]
            vv = ml_map[v]
            if uu != vv:
                if not uu in adj[vv]:
                    adj[vv].add(uu)
                if not vv in adj[uu]:
                    adj[uu].add(vv)
        return adj, ml_count, ml_list_numba ,n_v_ml, ml_map
    len_data, d = data.shape
    adj, ml_count, ml_list_numba, n_v_ml, ml_map = adj_from_cl_ml(cl, ml, len_data)
    data2 = f2(data)
    y_values, y_values2 = compute_sums_numba(data, data2, ml_list_numba, n_v_ml, d)
    return y_values, y_values2, adj, ml_map, ml_count, n_v_ml, d

@njit(cache=True)
def compute_sums_numba(data, data2, ml_list_numba, n_v_ml, d):
    """
    Numba-JIT aggregation: compute coordinate sums and squared-norm sums per super-node.

    For each super-node i, sums the raw data vectors and their squared norms over all
    member data points.  These sufficient statistics are then used by distance_matrix
    and CentroidUpdate without iterating over individual points again.

    Args:
        data (float64[n, d]): Raw data matrix.
        data2 (float64[n]): Squared L2 norm of each data point.
        ml_list_numba (List[List[int]]): Member data-point indices per super-node.
        n_v_ml (int): Number of super-nodes.
        d (int): Feature dimensionality.

    Returns:
        y_values (float64[n_supernodes, d]): Coordinate sums per super-node.
        y_values2 (float64[n_supernodes]): Squared-norm sums per super-node.
    """
    y_values  = np.empty((n_v_ml, d), dtype=data.dtype)
    y_values2 = np.empty(n_v_ml, dtype=data2.dtype)

    for i in range(n_v_ml):
        idxs = ml_list_numba[i]
        s  = np.zeros(d, dtype=data.dtype)
        s2 = 0.0
        for j in idxs:
            s  += data[j]
            s2 += data2[j]
        y_values[i]  = s
        y_values2[i] = s2
    return y_values, y_values2

def recover_ml_from_membership(membership, ml_map):
    """
    Expand super-node cluster assignments back to individual data-point assignments.

    Each data point inherits the cluster label of its super-node, so all must-linked
    points automatically receive the same label.

    Args:
        membership (int32[n_supernodes]): Cluster label for each super-node.
        ml_map (int[n]): Maps each data-point index to its super-node index.

    Returns:
        membership_final (int[n]): Cluster label for each of the n data points.
    """
    membership_final = np.zeros_like(ml_map)
    for i in range(len(ml_map)):
        membership_final[i] = membership[ml_map[i]]
    return membership_final

def COPKM_P(random_state, data, ml, cl, k, verbose = False):
    """
    DSATUR-COP-K-Means: constrained K-Means with cost-aware DSATUR initialisation
    (Algorithm 7, Appendix A.4 of the KSKM paper).

    Initialises cluster centroids via k-means++, then iterates cost-aware DSATUR
    assignment (DSATUR_KM) and centroid update until WCSS no longer improves.  The key
    distinction from classic COP-K-Means is the assignment order: rather than processing
    data points in their original index order, each iteration selects the most-constrained
    unassigned super-node first (highest saturation degree), and assigns it to its nearest
    feasible centroid.  This dramatically reduces infeasibility and produces substantially
    lower WCSS starting solutions than data-order assignment.

    No Gurobi or external solver is required; the algorithm runs in polynomial time and
    is significantly faster than KSKM while still outperforming classic COP-K-Means in
    both feasibility rate and solution quality.

    Args:
        random_state (int): NumPy random seed for k-means++ initialisation.
        data (float64[n, d]): Data matrix.
        ml (list[tuple[int, int]]): Pairwise must-link constraints (0-indexed data points).
        cl (list[tuple[int, int]]): Pairwise cannot-link constraints (0-indexed data points).
        k (int): Number of clusters.
        verbose (bool): Unused; reserved for future logging.

    Returns:
        membership_final (int[n]): Cluster label for each data point (0-indexed), or []
            if no feasible k-colouring exists (chromatic number of the CL graph > k).
    """
    y_values, y_values2, adj, ml_map, ml_count, n_v_ml, d = preprocessing(data, cl, ml)
    sub_adjs = sub_adj_classification(adj, y_values, y_values2, ml_count)
    del cl, ml, adj
    C = choose_initial_cluster_centers(data, k, random_state)
    C2 = f2(C)
    C_T = C.T
    del C
    # membership = DSATUR_KM(n_v_ml, sub_adjs, C_T, C2, k)
    membership = COPKM_DSATUR(y_values, y_values2, k, ml_count, n_v_ml, d, sub_adjs, C_T, C2)
    if not len(membership):
        return []
    
    membership_final = recover_ml_from_membership(membership, ml_map)
    
    return membership_final



def COPKM(random_state, data, ml, cl, k, verbose = False, time_limit = 3600):
    """
    Classic COP-K-Means (Wagstaff et al., 2001) with super-node preprocessing.

    Processes must-link groups as super-nodes and iterates sequential greedy assignment
    (COPKM_assignmen) with centroid update until WCSS no longer improves or the time
    limit is reached.  Assignment follows the natural data index order — not DSATUR
    order — which is faster per iteration but more prone to infeasibility and poor local
    optima on dense constraint graphs.  Provided as a baseline for comparison.

    Args:
        random_state (int): NumPy random seed.
        data (float64[n, d]): Data matrix.
        ml (list[tuple[int, int]]): Pairwise must-link constraints (0-indexed data points).
        cl (list[tuple[int, int]]): Pairwise cannot-link constraints (0-indexed data points).
        k (int): Number of clusters.
        verbose (bool): If True, print WCSS whenever a new best is found.
        time_limit (float): Wall-clock time limit in seconds.

    Returns:
        membership_final (int[n]): Cluster label for each data point, or [] if no
            improving assignment was found within the time limit.
    """
    start_time = time.time()
    y_values, y_values2, adj, ml_map, ml_count, n_v_ml, d = preprocessing(data, cl, ml)
    del cl, ml
    C = choose_initial_cluster_centers(data, k, random_state)
    C2 = f2(C)
    C_T = C.T
    del C
    
    best_obj = np.inf
    best_membership = []
    while True:
        D = distance_matrix(y_values, y_values2, C_T, C2, ml_count)
        membership = COPKM_assignmen(adj, D, k, n_v_ml)
        C_T, C2, obj = CentroidUpdate(membership, y_values, y_values2, k, ml_count, n_v_ml, d)
        time_lefted = time_limit - time.time() + start_time
        if (obj < best_obj) and (time_lefted > 0):
            best_obj = obj
            if verbose:
                print('update best obj')
                print(best_obj)
            best_membership = copy.deepcopy(membership)
        else:
            break
    if len(best_membership):
        membership_final = recover_ml_from_membership(best_membership, ml_map)
        return membership_final
    return[]

def COPKM_assignmen(adj, D, k, n_v_ml):
    """
    Sequential greedy assignment for classic COP-K-Means.

    Iterates super-nodes in their natural index order (not saturation order).  Each
    super-node is assigned to the cheapest feasible cluster — the one with the smallest
    D[v, c] not yet used by any cannot-link neighbour — using a pre-sorted cost list.
    Returns [] immediately if any super-node has no feasible cluster, signalling
    infeasibility.

    Args:
        adj (dict[int, set]): Cannot-link adjacency list over super-nodes.
        D (float64[n, k]): WCSS distance matrix from distance_matrix.
        k (int): Number of clusters.
        n_v_ml (int): Number of super-nodes.

    Returns:
        membership (int32[n_supernodes]): Cluster assignments, or [] if infeasible.
    """
    membership = initiate_membership(n_v_ml)
    colors: Dict[int, int] = {}
    vertices = list(adj.keys())
    
    neigh_colors = {i: set() for i in vertices}

    cost = {}
    for i in vertices:
        idx = D[i].argsort()
        cost[i] = [(D[i,j], j) for j in idx] + [(np.inf, k)]

    for v in vertices:
        # pick uncolored vertex with max saturation (|neigh_colors[v]|), then max degree
        used = neigh_colors[v]
        cost_v = cost[v]
        i = 0
        _, c = cost_v[i]
        while c in used:
            i += 1
            _, c = cost_v[i]
        if c == k:
            return []
        colors[v] = c
        # update neighbors' saturation sets
        for w in adj[v]:
            if w not in colors:
                neigh_colors[w].add(c)
    
    for v, c in colors.items():
        membership[vertices[v]] = c

    return membership


def COPKM_DSATUR(y_values, y_values2, k, ml_count, n_v_ml, d, sub_adjs, C_T, C2):
    """
    Inner loop of COPKM_P: iterate DSATUR assignment and centroid update until convergence.

    Repeatedly calls DSATUR_KM to obtain a new feasible assignment and then CentroidUpdate
    to recompute centroids.  Stops as soon as WCSS fails to improve (strictly), which
    indicates a local optimum under the DSATUR assignment rule, or as soon as DSATUR_KM
    returns infeasible.  Always returns the best membership found so far.

    Args:
        y_values (float64[n, d]): Per-super-node coordinate sums.
        y_values2 (float64[n]): Per-super-node sum of squared norms.
        k (int): Number of clusters.
        ml_count (int[n]): Super-node sizes.
        n_v_ml (int): Number of super-nodes.
        d (int): Feature dimensionality.
        sub_adjs (dict): Component classification from sub_adj_classification.
        C_T (float64[d, k]): Transposed centroid matrix (updated each iteration).
        C2 (float64[k]): Squared L2 norms of centroids (updated each iteration).

    Returns:
        best_membership (int32[n_supernodes]): Best feasible assignment found, or [] if
            DSATUR_KM was infeasible on the very first call.
    """
    best_obj = np.inf
    best_membership = []
    while True:
        membership = DSATUR_KM(n_v_ml, sub_adjs, C_T, C2, k)
        if len(membership):
            C_T, C2, obj = CentroidUpdate(membership, y_values, y_values2, k, ml_count, n_v_ml, d)
            if obj < best_obj:
                best_obj = obj
                best_membership = copy.deepcopy(membership)
            else:
                return best_membership
        else:
            return best_membership
        

