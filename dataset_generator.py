import numpy as np

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
    perm = np.array(list(range(1, num_agents + 1)))
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