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
from experiments import plot_3d, plot_circle, plot_bps, plot_regret, plot_egalitarian, plot_disparity, plot_nsw, uniform_instance_generator, triangular_instance_generator, normal_instance_generator, plot_pairs
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

def execute(num_agents, num_iters):
    reg_1_list, reg_2_list, reg_3_list, reg_4_list = [], [], [], []
    eg_1_list, eg_2_list, eg_3_list, eg_4_list = [], [], [], []
    disp_1_list, disp_2_list, disp_3_list, disp_4_list = [], [], [], []
    nsw_1_list, nsw_2_list, nsw_3_list, nsw_4_list = [], [], [], []
    with open(f'test_matchings_n={num_agents}_iters={num_iters}_create_preflist.json', 'w') as results_file:
        for iter in range(num_iters):
            if iter % 100 == 0:
                print(iter)
            preflist = create_preflist(num_agents)
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
            min_nsw  = float('inf')
            # Reg_closed_subset, Eg_closed_subset, Disp_closed_subset, Snsw_closed_subset = None, None, None, None
            min_reg_matching, min_eg_matching, min_disp_matching, min_nsw_matching = None, None, None, None
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
                if nash_social_welfare < min_nsw:
                    min_nsw = nash_social_welfare
                    min_nsw_matching = matching_1
                    # Snsw_closed_subset = subset
            reg_1 = reg(min_reg_matching, preflist)
            reg_1_list.append(reg_1)
            eg_1 = eg(min_reg_matching, preflist)
            eg_1_list.append(eg_1)
            disp_1 = disp(min_reg_matching, preflist)
            disp_1_list.append(disp_1)
            nsw_1 = nsw(min_reg_matching, preflist)
            nsw_1_list.append(nsw_1)

            reg_2 = reg(min_eg_matching, preflist)
            reg_2_list.append(reg_2)
            eg_2 = eg(min_eg_matching, preflist)
            eg_2_list.append(eg_2)
            disp_2 = disp(min_eg_matching, preflist)
            disp_2_list.append(disp_2)
            nsw_2 = nsw(min_eg_matching, preflist)
            nsw_2_list.append(nsw_2)
            
            reg_3 = reg(min_disp_matching, preflist)
            reg_3_list.append(reg_3)
            eg_3 = eg(min_disp_matching, preflist)
            eg_3_list.append(eg_3)
            disp_3 = disp(min_disp_matching, preflist)
            disp_3_list.append(disp_3)
            nsw_3 = nsw(min_disp_matching, preflist)
            nsw_3_list.append(nsw_3)

            reg_4 = reg(min_nsw_matching, preflist)
            reg_4_list.append(reg_4)
            eg_4 = eg(min_nsw_matching, preflist)
            eg_4_list.append(eg_4)
            disp_4 = disp(min_nsw_matching, preflist)
            disp_4_list.append(disp_4)
            nsw_4 = nsw(min_nsw_matching, preflist)
            nsw_4_list.append(nsw_4)

            data = {
                "preflist": convert_to_builtin(preflist),
                "min_regret": min_reg_matching,
                "egalitarian": min_eg_matching,
                "sex_equal": min_disp_matching,
                "nsw": min_nsw_matching,
                "scores": {
                    "reg": [float(reg_1), float(reg_2), float(reg_3), float(reg_4)],
                    "eg": [float(eg_1), float(eg_2), float(eg_3), float(eg_4)],
                    "disp": [float(disp_1), float(disp_2), float(disp_3), float(disp_4)],
                    "nsw": [float(nsw_1), float(nsw_2), float(nsw_3), float(nsw_4)]
                }
            }
            results_file.write(json.dumps(convert_to_builtin(data)) + "\n")


num_agents = []
for n in range(5, 21):
    print("Number of Agents = ", n)
    execute(n, 10000000)
    num_agents.append(n)

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