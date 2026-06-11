import numpy as np
from itertools import permutations, product
import random

random.seed(42)

def print_ranklist(ranklist):
    # ranklist means preference 
    # lists for one side
    num_agents = len(ranklist)
    # number of agents on each side
    for i in range(num_agents):
        # iterating over each agent
        print(i + 1, ranklist[i])
        # prints the ranklist of agent i (1-indexed)

def print_preflist(preflist):
    # preflist means preference list
    # of both agents
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
    # creates a preference list for one agent
    perm = np.array(list(range(num_agents)))
    # creates a list of agents
    #  from 0 to num_agents-1
    ranklist = np.zeros((num_agents, num_agents), dtype=int)
    # ranklist[i] is preferences of agent i
    for i in range(num_agents):
        # iterating over each agent
        ranklist[i] = np.random.permutation(perm)
        # creates a random permutation of agents
        # and assigns it to the preferences of agent i (1-indexed)
    return ranklist
# returns a ranklist of agents

def create_preflist(num_agents):
    # preflist is a complete instance of the SMP
    preflist = np.zeros((2, num_agents, num_agents), dtype=int)
    # preflist[0] is preference list of men
    # preflist[1] is preference list of women
    preflist[0] = create_ranklist(num_agents)
    # invoking create_ranklist 
    # to create mens preferences
    preflist[1] = create_ranklist(num_agents)
    # invoking create_ranklist
    # to create womens preferences
    return preflist.tolist()
# returns a preference list of agents

def create_weight_matrix(n, a, b):
    # creates a weight matrix of size n x n
    # with weights between a and b (reals)
    weight_matrix = np.random.uniform(a, b, size=(n, n))
    # creates a random matrix of size n x n
    # with weights between a and b
    return weight_matrix.tolist()

def create_preference_list(weight_matrix):
    n = len(weight_matrix)
    
    # Men's preferences: sort each row in descending order of weights
    men_prefs = [sorted(range(n), key=lambda j: -weight_matrix[i][j]) for i in range(n)]
    
    # Women's preferences: sort each column in descending order of weights
    weight_matrix_T = np.transpose(weight_matrix)  # Transpose to iterate over columns
    women_prefs = [sorted(range(n), key=lambda i: -weight_matrix_T[j][i]) for j in range(n)]
    
    return [men_prefs, women_prefs]


def print_weight_matrix(weight_matrix):
    # prints the weight matrix
    print("Weight Matrix:")
    for row in weight_matrix:
        # print upto 2 decimals accuracy
        print(["{:.2f}".format(x) for x in row])

def popularity_dist_uniform(popularity):
    for i in range(len(popularity)):
        popularity[i] = np.random.uniform(0, 1)
    return popularity

def popularity_dist_triangular(popularity):
    for i in range(len(popularity)):
        popularity[i] = np.random.triangular(0, 0.5, 1)
    return popularity

def popularity_dist_half_normal(popularity):
    for i in range(len(popularity)):
        popularity[i] = np.abs(np.random.normal(0, 1))
    return popularity

def ranklist_generator(popularity, available, ranklist):
    if len(available) == 0:
        return
    probabilites = [popularity[i] for i in available]
    selected_person = random.choices(available, weights=probabilites/np.sum(probabilites), k = 1)
    ranklist.append(selected_person[0])
    i = np.where(available == selected_person[0])[0][0]
    available = np.delete(available, i)
    ranklist_generator(popularity, available, ranklist)
    return

def uniform_instance_generator(n):
    # Generate a random instance with n agents
    preflist = [[], []]
    popularity_men = np.zeros(n)
    popularity_women = np.zeros(n)
    popularity_dist_uniform(popularity_men)
    popularity_dist_uniform(popularity_women)
    available_men, available_women = np.arange(n), np.arange(n)
    for i in range(n):
        # decide preflist[0][i] and preflist[1][i]
        ranklist_1, ranklist_2 = [], []
        ranklist_generator(popularity_men, available_women, ranklist_1)
        ranklist_generator(popularity_women, available_men, ranklist_2)
        preflist[0].append(ranklist_1)
        preflist[1].append(ranklist_2)
    return preflist

def triangular_instance_generator(n):
    # Generate a random instance with n agents
    preflist = [[], []]
    popularity_men = np.zeros(n)
    popularity_women = np.zeros(n)
    popularity_dist_triangular(popularity_men)
    popularity_dist_triangular(popularity_women)
    available_men, available_women = np.arange(n), np.arange(n)
    for i in range(n):
        # decide preflist[0][i] and preflist[1][i]
        ranklist_1, ranklist_2 = [], []
        ranklist_generator(popularity_men, available_women, ranklist_1)
        ranklist_generator(popularity_women, available_men, ranklist_2)
        preflist[0].append(ranklist_1)
        preflist[1].append(ranklist_2)
    return preflist

def normal_instance_generator(n):
    # Generate a random instance with n agents
    preflist = [[], []]
    popularity_men = np.zeros(n)
    popularity_women = np.zeros(n)
    popularity_dist_half_normal(popularity_men)
    popularity_dist_half_normal(popularity_women)
    available_men, available_women = np.arange(n), np.arange(n)
    for i in range(n):
        # decide preflist[0][i] and preflist[1][i]
        ranklist_1, ranklist_2 = [], []
        ranklist_generator(popularity_men, available_women, ranklist_1)
        ranklist_generator(popularity_women, available_men, ranklist_2)
        preflist[0].append(ranklist_1)
        preflist[1].append(ranklist_2)
    return preflist

def generate_matchings(n):
    yield from permutations(range(n))    

def generate_instances(n):
    prefs = list(permutations(range(n)))

    for profile in product(prefs, repeat=2*n):
        men = [list(profile[i]) for i in range(n)]
        women = [list(profile[i]) for i in range(n, 2*n)]
        yield [men, women]