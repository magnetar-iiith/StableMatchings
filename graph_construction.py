from collections import defaultdict, deque
from rotations import eliminate_rotation
from shortlists import print_shortlists

def is_elimating_rotation(pair, rotation, women_shortlists):
    for p in rotation:
        if p == pair:
            return True
    m = pair[0]
    w_i = pair[1]
    for p in rotation:
        if w_i == p[1]:
            m_i = p[0]
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
def create_rotation_digraph(rotations, men_shortlists, women_shortlists):
    r = len(rotations)
    graph = {i: set() for i in range(r)}
    for i in range(r):  # pi
        for j in range(r): # rho
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



    trimmed_graph = {node: set(successors) for node, successors in graph.items()}

    for n1 in graph:
        for n2 in list(graph[n1]):
            visited = set()
            stack = [n2]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                if current != n2 and current in trimmed_graph[n1]:
                    trimmed_graph[n1].remove(current)

                stack.extend(graph.get(current, []))

    # Convert sets back to lists (optional, for consistency)
    return {node: list(successors) for node, successors in trimmed_graph.items()}

def predecessors(graph):
    pred = {key: set() for key in graph}
    for key in graph:
        for successor in graph[key]:
            pred[successor].add(key)
    return {key: list(pred[key]) for key in pred}

def topological_sort(graph, pred):
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

def recur(index, s, S, topo_order, pred):
    n = len(topo_order)
    if index < n:
        recur(index + 1, s, S, topo_order, pred)

        r = topo_order[index]
        flag = True
        for p in pred[r]:
            if p not in s:
                flag = False
        if flag:
            if s not in S:
                S.append(s)
            recur(index + 1, s + [r], S, topo_order, pred)
    elif index == n:
        if s not in S:
            S.append(s)

def closed_subset_finder(topo_order, pred):
    closed_subsets = []
    s = []
    # print_graph(pred)
    recur(0, s, closed_subsets, topo_order, pred)
    return closed_subsets

def assign_weights(rotations, preflist):
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
            weights[i] /= ((n - preflist[0][m].index(rotation[j_plus][1])) * (n - preflist[1][w].index(rotation[j_minus][0])))
    return weights
    # weights_1, weights_2 = defaultdict(float), defaultdict(float)
    # n = len(preflist[0])
    # c = 1
    # for i, rotation in enumerate(rotations):
    #     weights_1[i] = 0
    #     weights_2[i] = 1
    # for i, rotation in enumerate(rotations):
    #     r = len(rotation)
    #     for j, (m, w) in enumerate(rotation):
    #         j_plus = (j + 1) % r
    #         j_minus = (j + r - 1) % r
    #         weights_1[i] += (preflist[0][m].index(w) + preflist[1][w].index(m))
    #         # print(f"{i}: {weights_1[i]}")
    #         weights_1[i] -= (preflist[0][m].index(rotation[j_plus][1]) + preflist[1][w].index(rotation[j_minus][0]))
    #         # print(f"{i}: {weights_1[i]}")
    #         weights_2[i] *= ((preflist[0][m].index(w) + c) * (preflist[1][w].index(m) + c))
    #         weights_2[i] /= ((preflist[0][m].index(rotation[j_plus][1]) + c) * (preflist[1][w].index(rotation[j_minus][0]) + c))
    #     weights_2[i] = weights_2[i]
    # return (weights_1, weights_2)

def max_weight_subset_1(closed_subsets, weights):
    max_weight = float('-inf')
    optimal_subset = None
    for subset in closed_subsets:
        weight = sum(weights[i] for i in subset)
        if weight > max_weight:
            max_weight = weight
            optimal_subset = subset
    return optimal_subset

def max_weight_subset_2(closed_subsets, weights):
    max_weight = float('-inf')
    optimal_subset = None
    for subset in closed_subsets:
        product = 1
        for i in subset:
            product *= weights[i]
        if product > max_weight:
            max_weight = product
            optimal_subset = subset
    return optimal_subset

def stable_matching(s, rotations, topo_order, men_shortlists, women_shortlists):
    for i in range(len(topo_order)):
        if topo_order[i] not in s:
            continue
        eliminate_rotation(rotations[topo_order[i]], men_shortlists, women_shortlists)
        # print("Iteration:", i)
        # print("Men's Shortlists:")
        # print_shortlists(men_shortlists)
        # print("Womens's Shortlists:")
        # print_shortlists(women_shortlists)
    matching = []
    for man, prefs in enumerate(men_shortlists):
        matching.append(prefs[0])
    return matching

def print_weights(weights):
    for w in weights:
        print(f"{w}: {weights[w]:.2f}")
        # rounded to two decimal places

def print_graph(graph):
    for key in graph:
        print(f"{key}: {graph[key]}")