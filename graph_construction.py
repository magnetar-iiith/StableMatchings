"""Constructs the rotation digraph"""

from collections import defaultdict, deque
from rotations import eliminate_rotation

def is_elimating_rotation(rho, rotation, women_shortlists):
    """Checks if rotation is an eliminating rotation for pair rho"""
    for rot_pi in rotation:
        if rot_pi == rho:
            return True
    m = rho[0]
    w_i = rho[1]
    for rot_pi in rotation:
        if w_i == rot_pi[1]:
            m_i = rot_pi[0]
            i = rotation.index((m_i, w_i))
            r = len(rotation)
            i += r - 1
            i %= r
            m_j = rotation[i][0]
            m_i_rank = women_shortlists[w_i].index(m_i)
            m_j_rank = women_shortlists[w_i].index(m_j)
            m_rank = women_shortlists[w_i].index(m)
            if  m_i_rank > m_rank and m_rank > m_j_rank:
                return True
    return False

def create_rotation_digraph(rotations, men_shortlists,\
                             women_shortlists):
    """Creates rotation digraph"""
    graph = {i: set() for i in range(len(rotations))}
    for i, _ in enumerate(rotations):  # pi
        for j,__ in enumerate(rotations): # rho
            if i != j:
                for idx, (m_i, w_i) in enumerate(rotations[j]): # iterate over men in rho
                    l = len(rotations[j])
                    id = idx + 1
                    id %= l
                    w_j = rotations[j][id][1] # w_{i+1}
                    for x in men_shortlists[m_i]: # iterate over x better than w_{i+1}
                        if x == w_i:
                            continue
                        x_rank = men_shortlists[m_i].index(x)
                        w_j_rank = men_shortlists[m_i].index(w_j)
                        if x_rank < w_j_rank:
                            if is_elimating_rotation((m_i, x), rotations[i], women_shortlists):
                                graph[i].add(j)

    return graph

def predecessors(graph):
    """Computes predecessors of each node in rotation digraph"""
    pred = {key: set() for key in graph}
    for key in graph:
        for successor in graph[key]:
            pred[successor].add(key)
    return {key: list(pred[key]) for key in pred}

def topological_sort(graph, pred):
    """Computes a topo sort on the rotation digraph"""
    n = len(graph)
    ind = [0] * n  # in-degree of each node
    for key in pred:
        ind[key] = len(pred[key])
    # Queue for nodes with 0 in-degree
    queue = deque([u for u in range(n) if ind[u] == 0])
    topo_order = []

    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in graph[u]:
            ind[v] -= 1
            if ind[v] == 0:
                queue.append(v)

    if len(topo_order) != n:
        raise ValueError("Graph is not a DAG (contains a cycle)")

    return topo_order

def dfs(index, s, S, topo_order, pred):
    """dfs on the rotation digraph"""
    n = len(topo_order)
    if index < n:
        dfs(index + 1, s, S, topo_order, pred)

        r = topo_order[index]
        flag = True
        for p in pred[r]:
            if p not in s:
                flag = False
        if flag:
            if s not in S:
                S.append(s)
            dfs(index + 1, s + [r], S, topo_order, pred)
    elif index == n:
        if s not in S:
            S.append(s)

def closed_subset_finder(topo_order, pred):
    """Finds all closed subsets in the rotation digraph"""
    closed_subsets = []
    s = []
    # print_graph(pred)
    dfs(0, s, closed_subsets, topo_order, pred)
    return closed_subsets

def assign_weights(rotations, preflist):
    """Assigns weights to rotations"""
    weights = defaultdict(float)
    n = len(preflist[0])
    for i, rotation in enumerate(rotations):
        weights[i] = 1
    for i, rotation in enumerate(rotations):
        r = len(rotation)
        for j, (m, w) in enumerate(rotation):
            j_plus = (j + 1) % r
            j_minus = (j + r - 1) % r
            weights[i] *= ((n - preflist[0][m].index(w)) * (n - preflist[1][w].index(m)))
            weights[i] /= \
            ((n - preflist[0][m].index(rotation[j_plus][1])) *\
              (n - preflist[1][w].index(rotation[j_minus][0])))
    return weights

def stable_matching(s, rotations, topo_order,\
                     men_shortlists, women_shortlists):
    """Returns the stable matching
    corresponding to the closed subset rotations"""
    for i in range(len(topo_order)):
        if topo_order[i] not in s:
            continue
        eliminate_rotation(rotations[topo_order[i]],\
                           men_shortlists, women_shortlists)
    matching = []
    for man, prefs in enumerate(men_shortlists):
        matching.append(prefs[0])
    return matching

def print_weights(weights):
    """Prints the weights of rotations"""
    for w in weights:
        print(f"{w}: {weights[w]:.2f}")
        # rounded to two decimal places

def print_graph(graph):
    """Prints the rotation digraph"""
    for key in graph:
        print(f"{key}: {graph[key]}")
