"""This module generates synthetic data of preferences"""

from itertools import permutations, product
import random
import numpy as np
import copy

random.seed(42)

def print_ranklist(ranklist):
    """Generates a ranklist
    ranklist means preference lists for one side"""
    num_agents = len(ranklist)
    # number of agents on each side
    for i in range(num_agents):
        # iterating over each agent
        print(i, ranklist[i])
        # prints the ranklist of agent i (1-indexed)

def print_preflist(preflist):
    """Generates preference list of both agents"""
    print("Preference list:")
    print("Men:")
    print_ranklist(preflist[0])
    # first ranklist is preference
    # list of men
    print("Women:")
    print_ranklist(preflist[1])
    # second ranklist is prefernce
    # list of women

def create_ranklist(num_agents):
    ranklist = [[0] * num_agents for _ in range(num_agents)]

    for i in range(num_agents):
        perm = list(range(num_agents))
        random.shuffle(perm)
        ranklist[i] = perm.copy()

    return ranklist

# def create_ranklist(num_agents):
#     """Generates a preference list for one agent"""
#     perm = np.array(list(range(num_agents)))
#     # creates a list of agents
#     #  from 0 to num_agents-1
#     ranklist = np.zeros((num_agents, num_agents), dtype=int)
#     # ranklist[i] is preferences of agent i
#     for i in range(num_agents):
#         # iterating over each agent
#         ranklist[i] = np.random.permutation(perm)
#         # creates a random permutation of agents
#         # and assigns it to the preferences of agent i (1-indexed)
#     return ranklist

def create_preflist(num_agents):
    """Generates the complete instance"""
    preflist = [create_ranklist(num_agents), create_ranklist(num_agents)]
    return preflist
# returns a preference list of agents

def create_weight_matrix(num_agents, lower, upper):
    """creates a weight matrix of size num_agents x num_agents
    with weights between 
    lower bound lower and upper bound upper (reals)"""
    weight_matrix = np.random.uniform(lower, upper,\
                                     size=(num_agents, num_agents))
    # creates a random matrix of size n x n
    # with weights between a and b
    return weight_matrix.tolist()

def create_preference_list(weight_matrix):
    """Generates an instance based on the cardinal weight matrix"""
    num_aqents = len(weight_matrix)
    # Men's preferences:
    # sort each row in descending order of weights
    men_prefs = [sorted(range(num_aqents),
            key=lambda j, i=i: -weight_matrix[i][j]
        )
        for i in range(num_aqents)
    ]
    # Women's preferences:
    # sort each column in descending order of weights
    weight_matrix_transpose = np.transpose(weight_matrix)
    # Transpose to iterate over columns
    women_prefs = [sorted(range(num_aqents), \
                    key=lambda i, j = j: -weight_matrix_transpose[j][i])\
                    for j in range(num_aqents)]
    return [men_prefs, women_prefs]

def print_weight_matrix(weight_matrix):
    """prints the weight matrix"""
    print("Weight Matrix:")
    for row in weight_matrix:
        # print upto 2 decimals accuracy
        print([f"{x:.2f}" for x in row])

def popularity_dist_uniform(popularity):
    """Generates an instances based on popularity (uniform)"""
    for i, _ in enumerate(popularity):
        popularity[i] = np.random.uniform(0, 1)
    return popularity

def popularity_dist_triangular(popularity):
    """Generates an instances based on popularity (triangular)"""
    for i, _ in enumerate(popularity):
        popularity[i] = np.random.triangular(0, 0.5, 1)
    return popularity

def popularity_dist_half_normal(popularity):
    """Generates an instances based on popularity (normal)"""
    for i, _ in enumerate(popularity):
        popularity[i] = np.abs(np.random.normal(0, 1))
    return popularity

def ranklist_generator(popularity, available, ranklist):
    """Generates a ranklist based on popularity profile"""
    if len(available) == 0:
        return
    probabilites = [popularity[idx] for idx in available]
    selected_person = random.choices(available, \
                    weights=probabilites/np.sum(probabilites), \
                    k = 1)
    ranklist.append(selected_person[0])
    idx = np.where(available == selected_person[0])[0][0]
    available = np.delete(available, idx)
    ranklist_generator(popularity, available, ranklist)
    return

def uniform_instance_generator(num_aqents):
    """Generate a random instance with num_aqents men and women"""
    preflist = [[], []]
    popularity_men = np.zeros(num_aqents)
    popularity_women = np.zeros(num_aqents)
    popularity_dist_uniform(popularity_men)
    popularity_dist_uniform(popularity_women)
    available_men, available_women \
        = np.arange(num_aqents), np.arange(num_aqents)
    for _ in range(num_aqents):
        # decide preflist[0][i] and preflist[1][i]
        ranklist_1, ranklist_2 = [], []
        ranklist_generator(popularity_men, available_women,\
                            ranklist_1)
        ranklist_generator(popularity_women, available_men,\
                            ranklist_2)
        preflist[0].append(ranklist_1)
        preflist[1].append(ranklist_2)
    return preflist

def triangular_instance_generator(num_aqents):
    """Generate a random instance with num_aqents men and women"""
    preflist = [[], []]
    popularity_men = np.zeros(num_aqents)
    popularity_women = np.zeros(num_aqents)
    popularity_dist_triangular(popularity_men)
    popularity_dist_triangular(popularity_women)
    available_men, available_women \
        = np.arange(num_aqents), np.arange(num_aqents)
    for _ in range(num_aqents):
        # decide preflist[0][i] and preflist[1][i]
        ranklist_1, ranklist_2 = [], []
        ranklist_generator(popularity_men, available_women,\
                            ranklist_1)
        ranklist_generator(popularity_women, available_men,\
                            ranklist_2)
        preflist[0].append(ranklist_1)
        preflist[1].append(ranklist_2)
    return preflist

def normal_instance_generator(num_aqents):
    """Generate a random instance with num_aqents men and women"""
    preflist = [[], []]
    popularity_men = np.zeros(num_aqents)
    popularity_women = np.zeros(num_aqents)
    popularity_dist_half_normal(popularity_men)
    popularity_dist_half_normal(popularity_women)
    available_men, available_women \
        = np.arange(num_aqents), np.arange(num_aqents)
    for _ in range(num_aqents):
        # decide preflist[0][i] and preflist[1][i]
        ranklist_1, ranklist_2 = [], []
        ranklist_generator(popularity_men, available_women,\
                            ranklist_1)
        ranklist_generator(popularity_women, available_men,\
                            ranklist_2)
        preflist[0].append(ranklist_1)
        preflist[1].append(ranklist_2)
    return preflist

def generate_matchings(num_aqents):
    """Generates all possible matchings
     with size num_agents men and women"""
    yield from permutations(range(num_aqents))

def generate_instances(num_agents):
    """
    Generate all unique preference profiles with the first man's
    preference list fixed to [0, 1, ..., num_agents-1].
    """
    prefs = list(permutations(range(num_agents)))
    fixed_pref = tuple(range(num_agents))

    # Generate preferences for the remaining agents:
    # men[1:], women[:]
    for profile in product(prefs, repeat=2 * num_agents - 1):
        men = [list(fixed_pref)] + [
            list(profile[i]) for i in range(num_agents - 1)
        ]
        women = [
            list(profile[i]) for i in range(num_agents - 1, 2 * num_agents - 1)
        ]
        yield [men, women]
        
def generate_instances_n_6():
    """
    Yields (men_prefs, women_prefs) for all 24^4 instances.
    """
    # Fixed men's preference lists
    men_prefs = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 2, 3, 4, 5, 1],
        [0, 3, 4, 5, 1, 2],
        [0, 4, 5, 1, 2, 3],
        [0, 5, 1, 2, 3, 4]
    ]
    # All permutations of [1,2,3,4]
    all_perms = list(permutations(range(1, 6)))
    # num_agents = 5
    for perm0, perm1, perm2, perm3, perm4 in product(all_perms, repeat=5):
        women_prefs = [
            [0, 1, 2, 3, 4, 5],
            [perm0[0], 0, perm0[1], perm0[2], perm0[3], perm0[4]],
            [perm1[0], 0, perm1[1], perm1[2], perm1[3], perm1[4]],
            [perm2[0], 0, perm2[1], perm2[2], perm2[3], perm2[4]],
            [perm3[0], 0, perm3[1], perm3[2], perm3[3], perm3[4]],
            [perm4[0], 0, perm4[1], perm4[2], perm4[3], perm4[4]],
        ]
        yield men_prefs, women_prefs
    
def create_pattern(num_agents):
    preflist = [[list(range(num_agents)) for _ in range(num_agents)], \
                [list(range(num_agents)) for _ in range(num_agents)]]
    preflist[0][0] = list(range(num_agents))
    preflist[1][0] = list(range(num_agents))
    perm = list(range(1, num_agents))
    for i in range(1, num_agents):
        for j in range(1, num_agents):
            preflist[0][i][j] = perm[(i + j - 2) % (num_agents - 1)]
    for i in range(1, num_agents):
        preflist[1][i][1] = 0
        preflist[1][i][0] = perm[(i + j) % (num_agents - 1)]
    for j in range(2, num_agents):
        for i in range(1, num_agents):
            preflist[1][i][j] = perm[(i + j - 1) % (num_agents - 1)]
    return preflist

def generate_worst_ratio_prefs(n, k):

    men = []
    women = []

    # ---------------- Men's preferences ----------------
    for m in range(n):
        if m < k:
            men.append(list(range(n)))
        else:
            t = m - k
            men.append(
                list(range(k))
                + list(range(k + t, n))
                + list(range(k, k + t))
            )

    # ---------------- Women's preferences ----------------
    # First k women are identical
    for _ in range(k):
        women.append(list(range(n)))

    # Remaining women are cyclic
    for w in range(k, n):

        # Find the man who ranks w last
        first_man = None
        for m in range(n):
            if men[m][-1] == w:
                first_man = m

        pref = [first_man]

        # Identical men
        for m in range(k):
            pref.append(m)

        # Cyclic men in cyclic order after first_man
        cyclic = list(range(k, n))
        if first_man >= k:
            idx = cyclic.index(first_man)
            order = cyclic[idx + 1:] + cyclic[:idx]
        else:
            order = cyclic

        for m in order:
            pref.append(m)

        women.append(pref)

    return [men, women]

def generate_man0_preferences(n, k, r):
    """
    Generate all preference lists for man 0.

    Parameters
    ----------
    n : int
        Total number of women (0, ..., n-1).
    k : int
        Women 0,...,k-1 are identical.
        Women k,...,n-1 are cyclic.
    r : int
        Number of identical women to be taken before
        Woman 0 in the preference list

    Returns
    -------
    list_of_preference_lists : list[list[int]]
    """
    identical = list(range(k))
    cyclic = list(range(k, n))

    list_of_preference_lists = []

    # Number of cyclic agents placed before the identical block
    # for r in range(0, len(cyclic) + 1):
    for chosen in permutations(cyclic, r):
        chosen = list(chosen)

        # Remaining cyclic agents in increasing order
        remaining_identical = [w for w in identical if w not in chosen]
        remaining_cyclic = [w for w in cyclic if w not in chosen]

        preference = (
            chosen +
            remaining_identical +
            remaining_cyclic
        )

        list_of_preference_lists.append(preference)

    return list_of_preference_lists

def generate_instances(num_aqents):
    """Generates all possible instances
     with size num_agents men and women"""
    prefs = list(permutations(range(num_aqents)))
    for profile in product(prefs, repeat=num_aqents):
        women = [list(profile[i]) for i in range(num_aqents)]
        # women = [list(profile[i]) for \
        #          i in range(num_aqents, 2*num_aqents)]
        yield women
