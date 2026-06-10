import numpy as np
import json
import math
from scipy.optimize import linear_sum_assignment
import copy
from matplotlib import pyplot as plt
from dataset_generator import create_preflist, print_preflist, create_weight_matrix, create_preference_list, print_weight_matrix
from gale_shapley_alg import gale_shapley, print_matching, blocking_pairs
from shortlists import create_shortlists, print_shortlists
from rotations import find_a_rotation, eliminate_rotation, print_rotations
from graph_construction import closed_subset_finder, create_rotation_digraph, print_graph, predecessors, topological_sort, assign_weights, print_weights, max_weight_subset_1, max_weight_subset_2, stable_matching
from measures import reg, eg, disp, nsw, statistics
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations, product
import time
import os
np.random.seed(69)

def convert_to_builtin(obj):
    if isinstance(obj, dict):
        return {convert_to_builtin(k): convert_to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_builtin(x) for x in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# alg_1 is for min regret optimal algorithm
# alg_2 is for max satisfaction / egalitarian algorithm
# alg_3 is for min disparity / sex-equal matching algorithm
# alg_4 is for max nsw / nash social welfare algorithm
# c is for min regret measure
# d is for summation of ranks measure
# e is for disparity measure
# nsw is for nash social welfare measure
def generate_matchings(n):
    yield from permutations(range(n))    

def generate_instances(n):
    prefs = list(permutations(range(n)))

    for profile in product(prefs, repeat=2*n):
        men = [list(profile[i]) for i in range(n)]
        women = [list(profile[i]) for i in range(n, 2*n)]
        yield [men, women]

def execute(num_agents, num_iters):
    output_folder = "./matchings"
    os.makedirs(output_folder, exist_ok=True)
    filename = f"matchings_n={num_agents}_iters={num_iters}_create_preflist.json"
    filepath = os.path.join(output_folder, filename)
# def execute(num_agents):
    # reg_1_list, reg_2_list, reg_3_list, reg_4_list = [], [], [], []
    # eg_1_list, eg_2_list, eg_3_list, eg_4_list = [], [], [], []
    # disp_1_list, disp_2_list, disp_3_list, disp_4_list = [], [], [], []
    # nsw_1_list, nsw_2_list, nsw_3_list, nsw_4_list = [], [], [], []
    with open(filepath, 'w') as results_file:
        for iter in range(num_iters):
            if iter % 20000 == 0:
                print(iter)
            preflist = create_preflist(num_agents)
            # preflist = [[[10, 13, 9, 12, 3, 6, 17, 5, 0, 16, 2, 14, 1, 7, 4, 11, 15, 8], [8, 9, 4, 13, 12, 6, 15, 17, 3, 2, 0, 7, 11, 16, 5, 10, 14, 1], [14, 4, 3, 11, 2, 15, 1, 10, 0, 9, 7, 17, 5, 12, 13, 16, 8, 6], [12, 13, 3, 17, 16, 4, 11, 9, 8, 7, 1, 5, 10, 2, 6, 15, 14, 0], [17, 11, 8, 15, 14, 9, 3, 0, 13, 10, 2, 5, 7, 16, 6, 4, 1, 12], [2, 4, 16, 12, 5, 13, 14, 6, 7, 3, 0, 9, 15, 1, 10, 11, 17, 8], [9, 2, 0, 13, 10, 15, 3, 1, 12, 8, 11, 6, 4, 14, 17, 16, 5, 7], [7, 8, 2, 15, 5, 6, 4, 1, 14, 17, 12, 9, 16, 0, 11, 3, 13, 10], [2, 17, 8, 13, 15, 12, 9, 0, 7, 1, 10, 5, 6, 3, 4, 11, 16, 14], [4, 10, 16, 5, 9, 0, 13, 6, 7, 14, 8, 11, 3, 15, 2, 12, 1, 17], [17, 0, 1, 7, 4, 14, 10, 9, 3, 8, 11, 5, 12, 16, 2, 13, 6, 15], [3, 14, 17, 12, 5, 16, 13, 4, 6, 8, 9, 1, 0, 11, 2, 10, 7, 15], [11, 16, 6, 15, 13, 5, 10, 1, 0, 7, 12, 2, 17, 8, 4, 9, 14, 3], [1, 5, 7, 17, 12, 16, 10, 6, 0, 14, 9, 13, 3, 2, 8, 4, 11, 15], [11, 9, 1, 5, 3, 7, 6, 12, 4, 17, 2, 0, 16, 8, 15, 14, 10, 13], [0, 11, 13, 4, 12, 14, 9, 15, 8, 3, 2, 10, 16, 6, 5, 1, 7, 17], [13, 7, 14, 6, 16, 3, 0, 11, 5, 9, 2, 12, 8, 1, 10, 15, 17, 4], [17, 10, 3, 9, 4, 7, 12, 6, 16, 2, 8, 0, 13, 14, 1, 15, 11, 5]], [[1, 16, 2, 8, 3, 14, 7, 11, 4, 10, 13, 17, 0, 15, 6, 12, 5, 9], [8, 12, 1, 0, 4, 14, 2, 9, 16, 11, 6, 3, 5, 15, 13, 17, 10, 7], [17, 7, 8, 5, 2, 15, 3, 9, 16, 10, 14, 12, 13, 11, 4, 0, 6, 1], [0, 2, 4, 9, 11, 7, 1, 13, 17, 14, 5, 16, 15, 12, 10, 8, 3, 6], [10, 14, 16, 2, 8, 0, 17, 6, 12, 9, 7, 13, 5, 4, 1, 3, 11, 15], [8, 9, 0, 12, 1, 16, 13, 14, 10, 4, 11, 5, 7, 15, 3, 2, 17, 6], [15, 2, 11, 13, 12, 4, 16, 6, 17, 5, 9, 14, 0, 1, 7, 8, 3, 10], [0, 12, 17, 10, 4, 7, 9, 5, 2, 1, 14, 6, 8, 11, 13, 16, 3, 15], [4, 6, 11, 12, 9, 3, 17, 0, 10, 8, 2, 15, 7, 13, 1, 14, 5, 16], [1, 14, 8, 9, 16, 6, 13, 3, 0, 2, 10, 11, 4, 12, 7, 15, 17, 5], [1, 12, 15, 7, 4, 9, 10, 16, 11, 5, 13, 8, 6, 0, 14, 3, 17, 2], [5, 8, 16, 3, 11, 7, 13, 6, 9, 10, 1, 12, 2, 4, 0, 17, 15, 14], [11, 7, 9, 10, 15, 13, 14, 5, 3, 16, 17, 1, 6, 4, 12, 2, 0, 8], [2, 11, 3, 9, 15, 14, 7, 1, 10, 12, 0, 6, 8, 5, 13, 17, 16, 4], [15, 10, 1, 12, 14, 16, 8, 6, 11, 5, 3, 4, 2, 13, 0, 9, 17, 7], [4, 0, 6, 12, 8, 13, 7, 5, 14, 15, 2, 3, 1, 16, 9, 10, 17, 11], [0, 1, 12, 14, 7, 16, 2, 6, 17, 13, 5, 8, 11, 4, 3, 10, 15, 9], [4, 7, 6, 3, 12, 2, 8, 11, 16, 17, 15, 9, 14, 5, 13, 10, 1, 0]]]
        # instances = generate_instances(num_agents)
        # matchings = generate_matchings(num_agents)
        # min_nsw_stable, max_nsw_unstable = float('inf'), 0
        # for preflist in instances:
        #     for match in matchings:
        #         if is_stable(match, preflist):
        #             min_nsw_stable = min(min_nsw_stable, nsw(match, preflist))
        #         else:
        #             max_nsw_unstable = max(max_nsw_unstable, nsw(match, preflist))

        # print(min_nsw_stable)
        # print(max_nsw_unstable)
        # print(min_nsw_stable / max_nsw_unstable)
            male_optimal_matching = gale_shapley(preflist)
            copy_preflist = copy.deepcopy(preflist)
            men_shortlists, women_shortlists = create_shortlists(copy_preflist, male_optimal_matching)

            copy_men_shortlists = copy.deepcopy(men_shortlists)
            # create an copy by value of mens shortlists
            copy_women_shortlists = copy.deepcopy(women_shortlists)
            # create an copy by value of womens shortlists

            rotations = []
            while True:
                new_rotation = find_a_rotation(copy_men_shortlists)
                if new_rotation is None:
                    break
                rotations.append(new_rotation)

                eliminate_rotation(new_rotation, copy_men_shortlists, copy_women_shortlists)

            copy_men_shortlists = copy.deepcopy(men_shortlists)
            # create an copy by value of mens shortlists

            copy_women_shortlists = copy.deepcopy(women_shortlists)

            graph = create_rotation_digraph(rotations, copy_men_shortlists, women_shortlists)

            pred = predecessors(graph)
            topo_order = topological_sort(graph, pred)
            closed_subsets = closed_subset_finder(topo_order, pred)
            min_reg  = float('inf')
            min_eg  = float('inf')
            min_disp  = float('inf')
            max_nsw  = float('-inf')
            # Reg_closed_subset, Eg_closed_subset, Disp_closed_subset, Snsw_closed_subset = None, None, None, None
            min_reg_matching, min_eg_matching, min_disp_matching, max_nsw_matching = None, None, None, None
            for subset in closed_subsets:
                copy_men_shortlists = copy.deepcopy(men_shortlists)
                copy_women_shortlists = copy.deepcopy(women_shortlists)
                matching_1 = stable_matching(subset, rotations, topo_order, copy_men_shortlists, copy_women_shortlists)
                regret = reg(matching_1, preflist)
                egalitarian = eg(matching_1, preflist)
                disparity = disp(matching_1, preflist)
                nash_social_welfare = nsw(matching_1, preflist)
                if regret < min_reg:
                    min_reg = regret
                    min_reg_matching = matching_1
                    # Reg_closed_subset = subset
                if egalitarian < min_eg:
                    min_eg = egalitarian
                    min_eg_matching = matching_1
                    # Eg_closed_subset = subset
                if disparity < min_disp:
                    min_disp = disparity
                    min_disp_matching = matching_1
                    # Disp_closed_subset = subset
                if nash_social_welfare > max_nsw:
                    max_nsw = nash_social_welfare
                    max_nsw_matching = matching_1
                    # Snsw_closed_subset = subset
            reg_1 = reg(min_reg_matching, preflist)
            # reg_1_list.append(reg_1)
            eg_1 = eg(min_reg_matching, preflist)
            # eg_1_list.append(eg_1)
            disp_1 = disp(min_reg_matching, preflist)
            # disp_1_list.append(disp_1)
            nsw_1 = nsw(min_reg_matching, preflist)
            # nsw_1_list.append(nsw_1)

            reg_2 = reg(min_eg_matching, preflist)
            # reg_2_list.append(reg_2)
            eg_2 = eg(min_eg_matching, preflist)
            # eg_2_list.append(eg_2)
            disp_2 = disp(min_eg_matching, preflist)
            # disp_2_list.append(disp_2)
            nsw_2 = nsw(min_eg_matching, preflist)
            # nsw_2_list.append(nsw_2)
            
            reg_3 = reg(min_disp_matching, preflist)
            # reg_3_list.append(reg_3)
            eg_3 = eg(min_disp_matching, preflist)
            # eg_3_list.append(eg_3)
            disp_3 = disp(min_disp_matching, preflist)
            # disp_3_list.append(disp_3)
            nsw_3 = nsw(min_disp_matching, preflist)
            # nsw_3_list.append(nsw_3)

            reg_4 = reg(max_nsw_matching, preflist)
            # reg_4_list.append(reg_4)
            eg_4 = eg(max_nsw_matching, preflist)
            # eg_4_list.append(eg_4)
            disp_4 = disp(max_nsw_matching, preflist)
            # disp_4_list.append(disp_4)
            nsw_4 = nsw(max_nsw_matching, preflist)
            # nsw_4_list.append(nsw_4)
            data = {
                "preflist": convert_to_builtin(preflist),
                "min_regret": min_reg_matching,
                "egalitarian": min_eg_matching,
                "sex_equal": min_disp_matching,
                "nsw": max_nsw_matching,
                "scores": {
                    "reg": [float(reg_1), float(reg_2), float(reg_3), float(reg_4)],
                    "eg": [float(eg_1), float(eg_2), float(eg_3), float(eg_4)],
                    "disp": [float(disp_1), float(disp_2), float(disp_3), float(disp_4)],
                    "nsw": [float(nsw_1), float(nsw_2), float(nsw_3), float(nsw_4)]
                }
            }
            results_file.write(json.dumps(convert_to_builtin(data)) + "\n")
        

def run(n):
    print(f"Number of Agents = {n}") 
    execute(n, 100000)
# execute(4)
if __name__ == "__main__":
    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run, range(5, 6)))
    end_time = time.time()
    hrs = (end_time - start_time)/3600
    mins = ((end_time - start_time)%3600)/60
    secs = (end_time - start_time) % 3600
    print(f"Time taken = {hrs} hours {mins} minutes {secs} seconds")


# preflist = [[[2, 1, 4, 0, 3],
#              [0, 4, 1, 3, 2],
#              [3, 2, 0, 1, 4],
#              [1, 2, 3, 0, 4],
#              [1, 2, 4, 0, 3]], 
#             [[2, 0, 4, 1, 3],
#              [0, 2, 3, 1, 4],
#              [3, 0, 1, 4, 2],
#              [3, 0, 1, 4, 2],
#              [2, 3, 0, 1, 4]]]

# preflist =  [[[3, 2, 1, 0, 4],
#               [1, 3, 0, 4, 2],
#               [0, 1, 2, 3, 4],
#               [4, 0, 2, 3, 1], 
#               [1, 2, 0, 3, 4]], 
#              [[3, 0, 1, 2, 4], 
#               [0, 3, 2, 1, 4], 
#               [2, 4, 0, 3, 1], 
#               [3, 1, 0, 2, 4],
#               [4, 1, 2, 3, 0]]]

# preflist = [[[4, 6, 0, 1, 5, 7, 3, 2], 
#             [1, 2, 6, 4, 3, 0, 7, 5], 
#             [7, 4, 0, 3, 5, 1, 2, 6], 
#             [2, 1, 6, 3, 0, 5, 7, 4], 
#             [6, 1, 4, 0, 2, 5, 7, 3], 
#             [0, 5, 6, 4, 7, 3, 1, 2], 
#             [1, 4, 6, 5, 2, 3, 7, 0], 
#             [2, 7, 3, 4, 6, 1, 5, 0]],
#             [[4, 2, 6, 5, 0, 1, 7, 3], 
#             [7, 5, 2, 4, 6, 1, 0, 3], 
#             [0, 4, 5, 1, 3, 7, 6, 2], 
#             [7, 6, 2, 1, 3, 0, 4, 5], 
#             [5, 3, 6, 2, 7, 0, 1, 4], 
#             [1, 7, 4, 2, 3, 5, 6, 0], 
#             [6, 4, 1, 0, 7, 5, 3, 2], 
#             [6, 3, 0, 4, 1, 2, 5, 7]]]

# preflist = [[[2, 0, 4, 6, 3, 1, 7, 5],
#             [5, 0, 2, 3, 7, 6, 4, 1], 
#             [6, 3, 2, 5, 4, 0, 1, 7],
#             [4, 2, 7, 1, 5, 0, 3, 6],
#             [3, 0, 1, 7, 6, 2, 5, 4],
#             [5, 1, 4, 6, 7, 3, 2, 0],
#             [6, 7, 0, 5, 1, 2, 3, 4],
#             [1, 5, 6, 0, 7, 2, 3, 4]], 
#             [[3, 2, 7, 0, 1, 4, 6, 5],
#             [2, 6, 4, 7, 5, 3, 0, 1],
#             [6, 4, 7, 2, 5, 1, 0, 3],
#             [5, 3, 1, 6, 2, 0, 4, 7],
#             [7, 6, 0, 4, 5, 3, 2, 1],
#             [4, 3, 6, 5, 1, 7, 2, 0],
#             [0, 3, 4, 5, 1, 7, 2, 6],
#             [1, 4, 3, 2, 6, 7, 0, 5]]]

# preflist = [[[0, 3, 4, 1, 2], 
#              [2, 0, 4, 1, 3], 
#              [0, 3, 4, 2, 1], 
#              [0, 1, 3, 2, 4], 
#              [3, 1, 2, 4, 0]], 
#              [[0, 2, 3, 1, 4], 
#               [2, 4, 1, 0, 3], 
#               [1, 2, 4, 3, 0], 
#               [2, 0, 3, 1, 4], 
#               [2, 0, 4, 1, 3]]]

# preflist = [[[4, 2, 0, 1, 3], 
#           [0, 3, 1, 4, 2], 
#           [4, 1, 2, 3, 0], 
#           [2, 1, 4, 0, 3], 
#           [1, 0, 2, 4, 3]], 
#          [[0, 2, 3, 1, 4], 
#           [0, 1, 3, 2, 4],
#           [0, 1, 2, 3, 4], 
#           [1, 3, 0, 2, 4], 
#           [3, 1, 0, 4, 2]]]

# preflist = [[[1, 3, 2, 4, 6, 0, 7, 5], 
#              [2, 7, 3, 4, 5, 0, 1, 6], 
#              [7, 5, 3, 0, 4, 6, 1, 2], 
#              [1, 7, 2, 6, 5, 0, 3, 4], 
#              [0, 5, 6, 4, 1, 3, 7, 2], 
#              [5, 0, 3, 2, 6, 4, 7, 1], 
#              [0, 1, 2, 4, 5, 3, 6, 7], 
#              [2, 1, 4, 7, 6, 5, 0, 3]], 
#              [[5, 1, 2, 7, 3, 4, 0, 6], 
#               [4, 2, 6, 5, 3, 1, 7, 0], 
#               [6, 2, 3, 7, 0, 1, 4, 5], 
#               [5, 3, 2, 6, 7, 4, 1, 0], 
#               [5, 6, 1, 2, 0, 7, 3, 4], 
#               [0, 4, 7, 1, 2, 6, 3, 5], 
#               [2, 0, 4, 1, 7, 3, 6, 5], 
#               [5, 0, 2, 7, 4, 3, 6, 1]]]

# Hypothesis of n!^1/n is wrong for following example:
# preflist = [[[1, 3, 2, 0],
#              [2, 0, 1, 3], 
#              [0, 3, 2, 1],
#              [2, 1, 0, 3]],
#             [[3, 0, 2, 1],
#              [3, 2, 0, 1],
#              [0, 2, 3, 1], 
#              [2, 3, 0, 1]]]

# Me and Msnsw closed subsets are not just eliminating or adding rotations example
# preflist = [[[10, 13, 9, 12, 3, 6, 17, 5, 0, 16, 2, 14, 1, 7, 4, 11, 15, 8], [8, 9, 4, 13, 12, 6, 15, 17, 3, 2, 0, 7, 11, 16, 5, 10, 14, 1], [14, 4, 3, 11, 2, 15, 1, 10, 0, 9, 7, 17, 5, 12, 13, 16, 8, 6], [12, 13, 3, 17, 16, 4, 11, 9, 8, 7, 1, 5, 10, 2, 6, 15, 14, 0], [17, 11, 8, 15, 14, 9, 3, 0, 13, 10, 2, 5, 7, 16, 6, 4, 1, 12], [2, 4, 16, 12, 5, 13, 14, 6, 7, 3, 0, 9, 15, 1, 10, 11, 17, 8], [9, 2, 0, 13, 10, 15, 3, 1, 12, 8, 11, 6, 4, 14, 17, 16, 5, 7], [7, 8, 2, 15, 5, 6, 4, 1, 14, 17, 12, 9, 16, 0, 11, 3, 13, 10], [2, 17, 8, 13, 15, 12, 9, 0, 7, 1, 10, 5, 6, 3, 4, 11, 16, 14], [4, 10, 16, 5, 9, 0, 13, 6, 7, 14, 8, 11, 3, 15, 2, 12, 1, 17], [17, 0, 1, 7, 4, 14, 10, 9, 3, 8, 11, 5, 12, 16, 2, 13, 6, 15], [3, 14, 17, 12, 5, 16, 13, 4, 6, 8, 9, 1, 0, 11, 2, 10, 7, 15], [11, 16, 6, 15, 13, 5, 10, 1, 0, 7, 12, 2, 17, 8, 4, 9, 14, 3], [1, 5, 7, 17, 12, 16, 10, 6, 0, 14, 9, 13, 3, 2, 8, 4, 11, 15], [11, 9, 1, 5, 3, 7, 6, 12, 4, 17, 2, 0, 16, 8, 15, 14, 10, 13], [0, 11, 13, 4, 12, 14, 9, 15, 8, 3, 2, 10, 16, 6, 5, 1, 7, 17], [13, 7, 14, 6, 16, 3, 0, 11, 5, 9, 2, 12, 8, 1, 10, 15, 17, 4], [17, 10, 3, 9, 4, 7, 12, 6, 16, 2, 8, 0, 13, 14, 1, 15, 11, 5]], [[1, 16, 2, 8, 3, 14, 7, 11, 4, 10, 13, 17, 0, 15, 6, 12, 5, 9], [8, 12, 1, 0, 4, 14, 2, 9, 16, 11, 6, 3, 5, 15, 13, 17, 10, 7], [17, 7, 8, 5, 2, 15, 3, 9, 16, 10, 14, 12, 13, 11, 4, 0, 6, 1], [0, 2, 4, 9, 11, 7, 1, 13, 17, 14, 5, 16, 15, 12, 10, 8, 3, 6], [10, 14, 16, 2, 8, 0, 17, 6, 12, 9, 7, 13, 5, 4, 1, 3, 11, 15], [8, 9, 0, 12, 1, 16, 13, 14, 10, 4, 11, 5, 7, 15, 3, 2, 17, 6], [15, 2, 11, 13, 12, 4, 16, 6, 17, 5, 9, 14, 0, 1, 7, 8, 3, 10], [0, 12, 17, 10, 4, 7, 9, 5, 2, 1, 14, 6, 8, 11, 13, 16, 3, 15], [4, 6, 11, 12, 9, 3, 17, 0, 10, 8, 2, 15, 7, 13, 1, 14, 5, 16], [1, 14, 8, 9, 16, 6, 13, 3, 0, 2, 10, 11, 4, 12, 7, 15, 17, 5], [1, 12, 15, 7, 4, 9, 10, 16, 11, 5, 13, 8, 6, 0, 14, 3, 17, 2], [5, 8, 16, 3, 11, 7, 13, 6, 9, 10, 1, 12, 2, 4, 0, 17, 15, 14], [11, 7, 9, 10, 15, 13, 14, 5, 3, 16, 17, 1, 6, 4, 12, 2, 0, 8], [2, 11, 3, 9, 15, 14, 7, 1, 10, 12, 0, 6, 8, 5, 13, 17, 16, 4], [15, 10, 1, 12, 14, 16, 8, 6, 11, 5, 3, 4, 2, 13, 0, 9, 17, 7], [4, 0, 6, 12, 8, 13, 7, 5, 14, 15, 2, 3, 1, 16, 9, 10, 17, 11], [0, 1, 12, 14, 7, 16, 2, 6, 17, 13, 5, 8, 11, 4, 3, 10, 15, 9], [4, 7, 6, 3, 12, 2, 8, 11, 16, 17, 15, 9, 14, 5, 13, 10, 1, 0]]]