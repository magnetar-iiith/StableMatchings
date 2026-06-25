"""This module generates synthetic data of preferences"""

from itertools import permutations, product
import random
import numpy as np

random.seed(42)

def print_ranklist(ranklist):
    """Generates a ranklist
    ranklist means preference lists for one side"""
    num_agents = len(ranklist)
    # number of agents on each side
    for i in range(num_agents):
        # iterating over each agent
        print(i + 1, ranklist[i])
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
        
# def generate_instances(num_aqents):
#     """Generates all possible instances
#      with size num_agents men and women"""
#     prefs = list(permutations(range(num_aqents)))
#     for profile in product(prefs, repeat=2*num_aqents):
#         men = [list(profile[i]) for i in range(num_aqents)]
#         women = [list(profile[i]) for \
#                  i in range(num_aqents, 2*num_aqents)]
#         yield [men, women]
