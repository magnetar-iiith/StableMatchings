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
from experiments import c, d, e, nsw, statistics
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
# def execute(c_alg_1_avg, c_alg_2_avg, c_alg_3_avg, c_alg_4_avg, \
#             c_alg_1_var, c_alg_2_var, c_alg_3_var, c_alg_4_var, \
#             d_alg_1_avg, d_alg_2_avg, d_alg_3_avg, d_alg_4_avg, \
#             d_alg_1_var, d_alg_2_var, d_alg_3_var, d_alg_4_var, \
#             e_alg_1_avg, e_alg_2_avg, e_alg_3_avg, e_alg_4_avg, \
#             e_alg_1_var, e_alg_2_var, e_alg_3_var, e_alg_4_var, \
#             nsw_alg_1_avg, nsw_alg_2_avg, nsw_alg_3_avg, nsw_alg_4_avg, \
#             nsw_alg_1_var, nsw_alg_2_var, nsw_alg_3_var, nsw_alg_4_var, num_agents):
def compare(Mr1, Me1, Md1, Msnsw1, Mr2, Me2, Md2, Msnsw2, flag1, flag2):
    if (Msnsw1 - Mr1) * flag1 < 0:
        if(Msnsw2 - Mr2) * flag2 < 0:
            return [1, 0, 0] # regret
    if (Msnsw1 - Me1) * flag1 < 0:
        if(Msnsw2 - Me2) * flag2 < 0:
            return [0, 1, 0] # egalitarian
    if (Msnsw1 - Md1) * flag1 < 0:
        if(Msnsw2 - Md2) * flag2 < 0:
            return [0, 0, 1] # disparity

def find(line):
    data = json.loads(line)
    preflist = data["preflist"]
    n = len(preflist[0])
    Mr = data["min_regret"]
    Me = data["egalitarian"]
    Md = data["sex_equal"]
    Msnsw = data["nsw"]

    mu_r_Mr = data["scores"]["c"][0]
    mu_r_Me = data["scores"]["c"][1]
    mu_r_Md = data["scores"]["c"][2]
    mu_r_Msnsw = data["scores"]["c"][3]

    mu_e_Mr = data["scores"]["d"][0]
    mu_e_Me = n - data["scores"]["d"][1]
    mu_e_Md = data["scores"]["d"][2]
    mu_e_Msnsw = n - data["scores"]["d"][3]
    mu_d_Mr = data["scores"]["e"][0]
    mu_d_Me = data["scores"]["e"][1]
    mu_d_Md = data["scores"]["e"][2]
    mu_d_Msnsw = data["scores"]["e"][3]

    mu_nsw_Mr = data["scores"]["nsw"][0]
    mu_nsw_Me = data["scores"]["nsw"][1]
    mu_nsw_Md = data["scores"]["nsw"][2]
    mu_nsw_Msnsw = data["scores"]["nsw"][3]
    # return ((mu_e_Msnsw / mu_e_Me), preflist, Me, Msnsw)

    # check2 = True
    # check4 = True
    # check1 = True
    check1 = compare(mu_r_Mr, mu_r_Me, mu_r_Md, mu_r_Msnsw, \
            mu_e_Mr, mu_e_Me, mu_e_Md, mu_e_Msnsw, -1, 1)
    check2 = compare(mu_r_Mr, mu_r_Me, mu_r_Md, mu_r_Msnsw, \
            mu_d_Mr, mu_d_Me, mu_d_Md, mu_d_Msnsw, -1, -1)
    check3 = compare(mu_r_Mr, mu_r_Me, mu_r_Md, mu_r_Msnsw, \
            mu_nsw_Mr, mu_nsw_Me, mu_nsw_Md, mu_nsw_Msnsw, -1, 1)
    check4 = compare(mu_e_Mr, mu_e_Me, mu_e_Md, mu_e_Msnsw, \
            mu_d_Mr, mu_d_Me, mu_d_Md, mu_d_Msnsw, 1, -1)
    check5 = compare(mu_e_Mr, mu_e_Me, mu_e_Md, mu_e_Msnsw, \
            mu_nsw_Mr, mu_nsw_Me, mu_nsw_Md, mu_nsw_Msnsw, 1, 1)
    check6 = compare(mu_d_Mr, mu_d_Me, mu_d_Md, mu_d_Msnsw, \
            mu_nsw_Mr, mu_nsw_Me, mu_nsw_Md, mu_nsw_Msnsw, -1, 1)
    final_histogram = check1 + check2 + check3 + check4 + check5 + check6
    return final_histogram
    # print(check1, check2, check3, check4, check5, check6)
    # if not check1:
        # print("SNSW pareto dominated in mu_r, mu_e axes")
        # print("===== MU_r Values =====")
        # print("mu_r_Mr     =", mu_r_Mr)
        # print("mu_r_Me     =", mu_r_Me)
        # print("mu_r_Md     =", mu_r_Md)
        # print("mu_r_Msnsw  =", mu_r_Msnsw)

        # print("\n===== MU_e Values =====")
        # print("mu_e_Mr     =", mu_e_Mr)
        # print("mu_e_Me     =", mu_e_Me)
        # print("mu_e_Md     =", mu_e_Md)
        # print("mu_e_Msnsw  =", mu_e_Msnsw)

        # print("\n===== MU_d Values =====")
        # print("mu_d_Mr     =", mu_d_Mr)
        # print("mu_d_Me     =", mu_d_Me)
        # print("mu_d_Md     =", mu_d_Md)
        # print("mu_d_Msnsw  =", mu_d_Msnsw)

        # print("\n===== MU_nsw Values =====")
        # print("mu_nsw_Mr     =", mu_nsw_Mr)
        # print("mu_nsw_Me     =", mu_nsw_Me)
        # print("mu_nsw_Md     =", mu_nsw_Md)
        # print("mu_nsw_Msnsw  =", mu_nsw_Msnsw)
        # print(Mr, Me, Md, Msnsw)
    # if not check2:
        # print("SNSW pareto dominated in mu_r, mu_d axes")
        # print("===== MU_r Values =====")
        # print("mu_r_Mr     =", mu_r_Mr)
        # print("mu_r_Me     =", mu_r_Me)
        # print("mu_r_Md     =", mu_r_Md)
        # print("mu_r_Msnsw  =", mu_r_Msnsw)

        # print("\n===== MU_e Values =====")
        # print("mu_e_Mr     =", mu_e_Mr)
        # print("mu_e_Me     =", mu_e_Me)
        # print("mu_e_Md     =", mu_e_Md)
        # print("mu_e_Msnsw  =", mu_e_Msnsw)

        # print("\n===== MU_d Values =====")
        # print("mu_d_Mr     =", mu_d_Mr)
        # print("mu_d_Me     =", mu_d_Me)
        # print("mu_d_Md     =", mu_d_Md)
        # print("mu_d_Msnsw  =", mu_d_Msnsw)

        # print("\n===== MU_nsw Values =====")
        # print("mu_nsw_Mr     =", mu_nsw_Mr)
        # print("mu_nsw_Me     =", mu_nsw_Me)
        # print("mu_nsw_Md     =", mu_nsw_Md)
        # print("mu_nsw_Msnsw  =", mu_nsw_Msnsw)
        # print(Mr, Me, Md, Msnsw)
    # if not check3:
        # print("SNSW pareto dominated in mu_r, mu_nsw axes")
        # print("===== MU_r Values =====")
        # print("mu_r_Mr     =", mu_r_Mr)
        # print("mu_r_Me     =", mu_r_Me)
        # print("mu_r_Md     =", mu_r_Md)
        # print("mu_r_Msnsw  =", mu_r_Msnsw)

        # print("\n===== MU_e Values =====")
        # print("mu_e_Mr     =", mu_e_Mr)
        # print("mu_e_Me     =", mu_e_Me)
        # print("mu_e_Md     =", mu_e_Md)
        # print("mu_e_Msnsw  =", mu_e_Msnsw)

        # print("\n===== MU_d Values =====")
        # print("mu_d_Mr     =", mu_d_Mr)
        # print("mu_d_Me     =", mu_d_Me)
        # print("mu_d_Md     =", mu_d_Md)
        # print("mu_d_Msnsw  =", mu_d_Msnsw)

        # print("\n===== MU_nsw Values =====")
        # print("mu_nsw_Mr     =", mu_nsw_Mr)
        # print("mu_nsw_Me     =", mu_nsw_Me)
        # print("mu_nsw_Md     =", mu_nsw_Md)
        # print("mu_nsw_Msnsw  =", mu_nsw_Msnsw)
        # print(Mr, Me, Md, Msnsw)
    # if not check4:
        # print("SNSW pareto dominated in mu_e, mu_d axes")
        # print("===== MU_r Values =====")
        # print("mu_r_Mr     =", mu_r_Mr)
        # print("mu_r_Me     =", mu_r_Me)
        # print("mu_r_Md     =", mu_r_Md)
        # print("mu_r_Msnsw  =", mu_r_Msnsw)

        # print("\n===== MU_e Values =====")
        # print("mu_e_Mr     =", mu_e_Mr)
        # print("mu_e_Me     =", mu_e_Me)
        # print("mu_e_Md     =", mu_e_Md)
        # print("mu_e_Msnsw  =", mu_e_Msnsw)

        # print("\n===== MU_d Values =====")
        # print("mu_d_Mr     =", mu_d_Mr)
        # print("mu_d_Me     =", mu_d_Me)
        # print("mu_d_Md     =", mu_d_Md)
        # print("mu_d_Msnsw  =", mu_d_Msnsw)

        # print("\n===== MU_nsw Values =====")
        # print("mu_nsw_Mr     =", mu_nsw_Mr)
        # print("mu_nsw_Me     =", mu_nsw_Me)
        # print("mu_nsw_Md     =", mu_nsw_Md)
        # print("mu_nsw_Msnsw  =", mu_nsw_Msnsw)
        # print(Mr, Me, Md, Msnsw)
    # if not check5:
        # print("SNSW pareto dominated in mu_e, mu_nsw axes")
        # print("===== MU_r Values =====")
        # print("mu_r_Mr     =", mu_r_Mr)
        # print("mu_r_Me     =", mu_r_Me)
        # print("mu_r_Md     =", mu_r_Md)
        # print("mu_r_Msnsw  =", mu_r_Msnsw)

        # print("\n===== MU_e Values =====")
        # print("mu_e_Mr     =", mu_e_Mr)
        # print("mu_e_Me     =", mu_e_Me)
        # print("mu_e_Md     =", mu_e_Md)
        # print("mu_e_Msnsw  =", mu_e_Msnsw)

        # print("\n===== MU_d Values =====")
        # print("mu_d_Mr     =", mu_d_Mr)
        # print("mu_d_Me     =", mu_d_Me)
        # print("mu_d_Md     =", mu_d_Md)
        # print("mu_d_Msnsw  =", mu_d_Msnsw)

        # print("\n===== MU_nsw Values =====")
        # print("mu_nsw_Mr     =", mu_nsw_Mr)
        # print("mu_nsw_Me     =", mu_nsw_Me)
        # print("mu_nsw_Md     =", mu_nsw_Md)
        # print("mu_nsw_Msnsw  =", mu_nsw_Msnsw)
        # print(Mr, Me, Md, Msnsw)
    # if not check6:
        # print("SNSW pareto dominated in mu_d, mu_nsw axes")
        # print("===== MU_r Values =====")
        # print("mu_r_Mr     =", mu_r_Mr)
        # print("mu_r_Me     =", mu_r_Me)
        # print("mu_r_Md     =", mu_r_Md)
        # print("mu_r_Msnsw  =", mu_r_Msnsw)

        # print("\n===== MU_e Values =====")
        # print("mu_e_Mr     =", mu_e_Mr)
        # print("mu_e_Me     =", mu_e_Me)
        # print("mu_e_Md     =", mu_e_Md)
        # print("mu_e_Msnsw  =", mu_e_Msnsw)

        # print("\n===== MU_d Values =====")
        # print("mu_d_Mr     =", mu_d_Mr)
        # print("mu_d_Me     =", mu_d_Me)
        # print("mu_d_Md     =", mu_d_Md)
        # print("mu_d_Msnsw  =", mu_d_Msnsw)

        # print("\n===== MU_nsw Values =====")
        # print("mu_nsw_Mr     =", mu_nsw_Mr)
        # print("mu_nsw_Me     =", mu_nsw_Me)
        # print("mu_nsw_Md     =", mu_nsw_Md)
        # print("mu_nsw_Msnsw  =", mu_nsw_Msnsw)
        # print(Mr, Me, Md, Msnsw)
    # if not check1 or not check2 or not check3 or not check4 or not check5 or not check6:
        # print_preflist(preflist)
        # return False
    # return True
    
def analyse(num_agents):
    # c_1_list, c_2_list, c_3_list, c_4_list = [], [], [], []
    # d_1_list, d_2_list, d_3_list, d_4_list = [], [], [], []
    # e_1_list, e_2_list, e_3_list, e_4_list = [], [], [], []
    # nsw_1_list, nsw_2_list, nsw_3_list, nsw_4_list = [], [], [], []
    # num_iters = 1
    dist_1 = "\\mathcal{U}"
    dist_2 = "\\mathrm{P}_{\\mathcal{N}}"
    dist_3 = "\\mathrm{P}_{\\mathcal{T}}"
    dist_4 = "\\mathrm{P}_{\\mathcal{U}}"
    with open(f"all_matchings/matchings_{num_agents}_{dist_1}.jsonl", "r") as f_1, \
         open(f"all_matchings/matchings_{num_agents}_{dist_2}.jsonl", "r") as f_2, \
         open(f"all_matchings/matchings_{num_agents}_{dist_3}.jsonl", "r") as f_3, \
         open(f"all_matchings/matchings_{num_agents}_{dist_4}.jsonl", "r") as f_4:
        # ratio_min, preflist, Me, Msnsw = 1.1, None, None, None
        h = [0, 0, 0, 0]
        for line_1, line_2, line_3, line_4 in zip(f_1, f_2, f_3, f_4):
            h1 = find(line_1)
            h2 = find(line_2)
            h3 = find(line_3)
            h4 = find(line_4)
            h += h1 + h2 + h3 + h4
        # print("percentage of instances snsw undominated")
        # print((100.0 * h[3]) / (h[0] + h[1] + h[2] + h[3]))
        plt.plot(['r', 'e', 'd'], [h[0], h[1], h[2]])
        plt.show()
            # if find(line_1) and find(line_2) and find(line_3) and find(line_4):
            #     continue
            # else:
            #     break

            # ratio1, preflist1, Me1, Msnsw1 = (find(line_1))
            # ratio2, preflist2, Me2, Msnsw2 = (find(line_2))
            # ratio3, preflist3, Me3, Msnsw3 = (find(line_3))
            # ratio4, preflist4, Me4, Msnsw4 = (find(line_4))
            # r_min_curr = min(ratio_min, ratio1, ratio2, ratio3, ratio4)
            # if r_min_curr >= ratio_min:
            #     continue
            # ratio_min = r_min_curr
            # if r_min_curr == ratio1:
            #     preflist = preflist1
            #     Me = Me1
            #     Msnsw = Msnsw1
            # if r_min_curr == ratio2:
            #     preflist = preflist2
            #     Me = Me2
            #     Msnsw = Msnsw2
            # if r_min_curr == ratio3:
            #     preflist = preflist3
            #     Me = Me3
            #     Msnsw = Msnsw3
            # if r_min_curr == ratio4:
            #     preflist = preflist4
            #     Me = Me4
            #     Msnsw = Msnsw4
            # p1 = (find(line_1))
            # p2 = (find(line_2))
            # p3 = (find(line_3))
            # p4 = (find(line_4))
            # if p1:
            #     print(p1)
            # if p2:
            #     print(p2)
            # if p3:
            #     print(p3)
            # if p4:
            #     print(p4)
        # print("Minimum Ratio = ", ratio_min)
        # print("Corresponding Preference Lists = ")
        # # print_preflist(preflist)
        # print(preflist)
        # print("Egalitarian Matching")
        # print_matching(Me)
        # print("M_snsw Matching")
        # print_matching(Msnsw)
        # return ratio_min

# def execute(num_agents, num_iters):
#     # max_ratio, min_ratio = 0, float('inf')
#     for iter in range(num_iters):
#         if iter % 100 == 0:
#             print(iter)
#         # print(iter)
#         # ans = []
#         # preflist = [[[26, 2, 32, 28, 39, 49, 42, 43, 27, 16, 21, 36, 41, 6, 47, 45, 5, 37, 35, 29, 22, 7, 20, 30, 0, 12, 1, 17, 14, 34, 23, 3, 19, 24, 33, 9, 46, 8, 15, 48, 31, 11, 4, 13, 10, 40, 18, 38, 44, 25], [44, 6, 5, 16, 30, 14, 42, 21, 1, 15, 49, 35, 13, 7, 46, 37, 43, 33, 40, 34, 38, 9, 23, 48, 39, 27, 3, 31, 17, 4, 41, 22, 18, 12, 11, 26, 10, 19, 8, 29, 2, 45, 47, 0, 32, 36, 28, 24, 25, 20], [21, 26, 37, 12, 2, 22, 0, 41, 23, 7, 42, 47, 4, 32, 44, 34, 30, 35, 6, 45, 33, 46, 8, 9, 40, 3, 29, 43, 20, 18, 15, 25, 49, 39, 13, 16, 5, 17, 28, 24, 36, 48, 14, 27, 11, 19, 10, 31, 1, 38], [26, 16, 45, 30, 6, 28, 23, 0, 46, 47, 39, 42, 29, 21, 27, 40, 7, 3, 49, 17, 13, 32, 35, 14, 48, 38, 44, 22, 11, 20, 34, 12, 5, 8, 25, 19, 41, 18, 9, 10, 4, 15, 37, 1, 31, 43, 24, 36, 2, 33], [17, 42, 38, 15, 20, 49, 22, 27, 21, 37, 1, 28, 40, 8, 45, 30, 35, 18, 39, 11, 5, 12, 24, 34, 41, 25, 32, 26, 29, 14, 16, 44, 6, 0, 7, 48, 9, 3, 47, 33, 31, 2, 46, 43, 23, 4, 36, 13, 10, 19], [16, 18, 49, 19, 17, 27, 28, 4, 32, 0, 37, 42, 5, 46, 40, 29, 14, 2, 10, 21, 41, 38, 22, 30, 36, 44, 8, 47, 23, 15, 9, 6, 45, 12, 3, 7, 48, 1, 43, 34, 26, 11, 25, 39, 20, 35, 13, 33, 24, 31], [5, 6, 18, 39, 16, 26, 34, 22, 40, 48, 21, 12, 37, 14, 27, 46, 17, 23, 1, 8, 0, 20, 45, 38, 29, 35, 15, 28, 4, 47, 9, 25, 3, 2, 42, 30, 33, 43, 10, 36, 49, 19, 13, 31, 41, 11, 7, 24, 32, 44], [18, 40, 20, 47, 43, 0, 31, 14, 11, 38, 37, 48, 8, 41, 15, 32, 16, 39, 42, 34, 30, 19, 22, 45, 1, 29, 3, 9, 26, 44, 5, 28, 46, 23, 12, 33, 6, 2, 27, 21, 17, 25, 24, 49, 36, 13, 4, 10, 7, 35], [14, 30, 31, 5, 13, 9, 2, 0, 20, 21, 26, 1, 17, 22, 40, 42, 29, 41, 49, 32, 47, 16, 15, 44, 33, 6, 39, 45, 8, 23, 48, 24, 46, 37, 18, 25, 38, 12, 19, 43, 28, 11, 4, 27, 36, 3, 7, 34, 10, 35], [28, 21, 42, 15, 18, 2, 16, 27, 44, 32, 19, 41, 49, 40, 23, 0, 39, 34, 20, 12, 37, 13, 30, 5, 45, 6, 46, 29, 11, 17, 1, 10, 38, 47, 8, 22, 4, 48, 24, 26, 33, 36, 25, 43, 3, 14, 31, 9, 7, 35], [46, 1, 9, 42, 45, 30, 29, 37, 5, 14, 49, 36, 34, 43, 39, 11, 25, 15, 7, 12, 6, 21, 8, 40, 22, 27, 26, 4, 16, 41, 2, 38, 44, 19, 17, 0, 18, 32, 3, 28, 48, 47, 13, 20, 23, 24, 10, 35, 33, 31], [23, 40, 14, 8, 37, 30, 43, 9, 22, 18, 46, 25, 19, 15, 26, 3, 49, 16, 20, 47, 11, 12, 0, 17, 45, 44, 32, 4, 27, 29, 48, 2, 42, 24, 35, 13, 28, 34, 38, 1, 41, 21, 10, 36, 6, 31, 7, 5, 39, 33], [39, 25, 30, 1, 21, 15, 0, 40, 19, 47, 36, 33, 22, 17, 3, 13, 2, 26, 42, 7, 23, 20, 11, 38, 9, 24, 45, 28, 41, 46, 29, 12, 27, 32, 35, 8, 49, 37, 5, 16, 43, 48, 34, 6, 4, 14, 44, 18, 31, 10], [17, 11, 47, 37, 22, 25, 18, 38, 9, 34, 14, 39, 0, 30, 20, 35, 21, 8, 45, 16, 33, 40, 29, 15, 3, 19, 23, 10, 43, 32, 27, 26, 6, 7, 48, 28, 2, 49, 42, 46, 31, 44, 12, 5, 36, 41, 4, 24, 1, 13], [9, 37, 49, 15, 24, 42, 25, 48, 23, 19, 34, 45, 8, 20, 6, 2, 1, 38, 28, 16, 22, 17, 31, 26, 40, 32, 18, 27, 47, 41, 5, 29, 4, 0, 12, 35, 11, 39, 43, 14, 30, 33, 46, 44, 7, 36, 13, 21, 10, 3], [45, 7, 28, 14, 46, 22, 29, 17, 21, 47, 9, 11, 6, 44, 3, 40, 42, 37, 43, 15, 13, 39, 4, 5, 0, 32, 26, 20, 41, 34, 30, 35, 16, 25, 38, 2, 19, 27, 48, 8, 24, 12, 49, 33, 31, 18, 36, 23, 1, 10], [26, 40, 44, 18, 42, 36, 45, 22, 10, 21, 30, 6, 5, 39, 11, 8, 31, 7, 33, 9, 32, 37, 16, 48, 15, 34, 35, 14, 23, 19, 28, 12, 13, 46, 0, 17, 2, 20, 1, 47, 25, 27, 29, 3, 49, 24, 4, 43, 41, 38], [44, 13, 20, 39, 29, 0, 18, 12, 40, 7, 14, 37, 10, 32, 1, 35, 2, 46, 41, 28, 3, 23, 47, 11, 15, 4, 26, 19, 33, 25, 21, 49, 38, 45, 31, 42, 22, 5, 16, 8, 27, 6, 9, 30, 17, 48, 34, 36, 24, 43], [28, 40, 47, 23, 38, 48, 16, 21, 15, 35, 8, 45, 2, 4, 27, 6, 30, 29, 14, 25, 41, 11, 44, 0, 18, 32, 26, 9, 1, 46, 19, 20, 37, 3, 24, 34, 5, 39, 12, 33, 43, 42, 22, 13, 17, 49, 7, 36, 10, 31], [48, 12, 11, 1, 36, 41, 28, 37, 46, 0, 31, 14, 40, 29, 9, 21, 22, 47, 5, 44, 18, 15, 8, 6, 34, 49, 30, 38, 7, 3, 26, 16, 13, 19, 10, 17, 23, 32, 27, 42, 45, 2, 43, 20, 35, 25, 39, 4, 33, 24], [18, 20, 34, 9, 5, 22, 26, 40, 32, 14, 37, 0, 4, 27, 16, 11, 3, 21, 8, 31, 44, 1, 47, 25, 45, 43, 48, 42, 30, 33, 36, 23, 12, 49, 2, 15, 28, 6, 29, 19, 24, 46, 39, 13, 41, 35, 10, 17, 38, 7], [12, 37, 4, 48, 34, 43, 9, 42, 16, 6, 14, 32, 39, 26, 2, 19, 0, 29, 15, 49, 46, 27, 45, 44, 18, 25, 8, 41, 21, 3, 13, 10, 11, 5, 24, 23, 22, 7, 20, 40, 35, 1, 17, 33, 47, 30, 38, 28, 36, 31], [43, 21, 29, 5, 15, 45, 48, 18, 33, 34, 27, 40, 14, 22, 32, 23, 49, 0, 8, 31, 19, 20, 44, 36, 37, 17, 46, 11, 28, 2, 9, 25, 41, 4, 42, 3, 1, 30, 13, 12, 16, 26, 39, 47, 10, 24, 7, 6, 38, 35], [17, 11, 20, 4, 23, 26, 6, 0, 2, 37, 42, 21, 30, 22, 29, 46, 32, 13, 40, 12, 5, 24, 34, 9, 14, 15, 16, 48, 45, 43, 19, 49, 38, 8, 31, 25, 7, 39, 33, 44, 47, 3, 36, 41, 28, 27, 1, 35, 18, 10], [16, 43, 21, 48, 30, 37, 7, 29, 22, 0, 28, 40, 12, 6, 45, 20, 3, 34, 19, 15, 32, 41, 23, 47, 35, 2, 31, 11, 18, 44, 42, 9, 27, 26, 10, 14, 8, 33, 4, 46, 39, 17, 24, 5, 49, 38, 36, 25, 1, 13], [8, 14, 40, 29, 0, 25, 28, 16, 42, 18, 9, 27, 45, 23, 21, 4, 46, 12, 3, 43, 10, 26, 6, 32, 37, 41, 30, 38, 22, 11, 47, 31, 20, 44, 35, 1, 24, 39, 15, 5, 2, 36, 7, 19, 49, 48, 34, 33, 17, 13], [13, 35, 24, 0, 30, 12, 5, 36, 6, 45, 48, 37, 18, 19, 4, 11, 15, 22, 26, 23, 28, 42, 33, 29, 32, 17, 41, 21, 46, 31, 7, 44, 27, 49, 8, 34, 40, 25, 39, 3, 20, 43, 16, 1, 14, 38, 9, 47, 2, 10], [19, 27, 37, 45, 32, 4, 17, 46, 12, 5, 9, 16, 29, 26, 21, 25, 43, 44, 48, 18, 8, 40, 49, 30, 0, 24, 42, 28, 1, 23, 15, 20, 10, 39, 31, 35, 6, 41, 22, 36, 11, 33, 47, 34, 14, 38, 2, 3, 7, 13], [16, 8, 10, 12, 21, 40, 43, 36, 5, 30, 27, 6, 29, 18, 39, 23, 17, 38, 4, 3, 13, 48, 25, 0, 42, 7, 19, 1, 34, 44, 46, 37, 11, 49, 20, 32, 26, 15, 33, 45, 47, 35, 22, 2, 9, 14, 24, 41, 28, 31], [21, 46, 20, 14, 37, 41, 26, 35, 43, 5, 45, 42, 29, 22, 15, 9, 12, 36, 2, 49, 32, 18, 30, 40, 13, 48, 23, 8, 17, 6, 47, 27, 0, 11, 25, 7, 28, 3, 39, 4, 19, 34, 24, 16, 10, 44, 38, 1, 33, 31], [43, 16, 38, 23, 12, 0, 28, 21, 39, 37, 48, 8, 32, 9, 15, 25, 42, 41, 19, 40, 46, 29, 45, 20, 30, 17, 6, 24, 2, 3, 14, 11, 27, 47, 4, 44, 5, 7, 49, 26, 18, 34, 22, 1, 31, 33, 10, 13, 36, 35], [1, 43, 28, 20, 25, 27, 6, 45, 36, 40, 14, 39, 8, 29, 15, 47, 37, 3, 23, 0, 26, 33, 42, 7, 10, 41, 2, 5, 35, 11, 21, 9, 19, 32, 12, 44, 48, 22, 31, 49, 30, 34, 17, 46, 18, 4, 16, 24, 13, 38], [29, 42, 15, 23, 12, 0, 31, 43, 44, 22, 19, 26, 30, 1, 18, 5, 41, 6, 33, 36, 48, 16, 37, 11, 34, 32, 45, 14, 25, 9, 46, 49, 40, 8, 21, 17, 47, 24, 13, 27, 10, 20, 7, 39, 4, 3, 35, 2, 28, 38], [3, 4, 21, 48, 42, 44, 0, 47, 45, 10, 40, 23, 41, 7, 20, 16, 5, 29, 30, 17, 1, 6, 34, 37, 22, 8, 18, 19, 15, 32, 33, 14, 31, 43, 11, 46, 39, 49, 9, 28, 35, 25, 12, 38, 26, 2, 27, 13, 36, 24], [49, 28, 46, 3, 45, 2, 16, 14, 40, 11, 17, 19, 10, 1, 15, 44, 26, 0, 21, 18, 22, 6, 27, 42, 30, 32, 7, 25, 9, 4, 39, 12, 23, 36, 29, 47, 41, 37, 31, 13, 48, 8, 5, 33, 43, 20, 38, 24, 34, 35], [22, 43, 29, 20, 46, 32, 23, 1, 6, 3, 2, 21, 35, 11, 8, 12, 49, 27, 15, 40, 9, 19, 26, 41, 36, 16, 17, 5, 24, 34, 37, 31, 42, 28, 4, 18, 33, 48, 7, 47, 44, 0, 14, 45, 30, 13, 39, 25, 10, 38], [44, 49, 46, 3, 14, 11, 15, 41, 9, 35, 26, 42, 8, 28, 47, 23, 30, 22, 39, 6, 32, 27, 0, 40, 12, 21, 37, 29, 43, 4, 2, 48, 5, 38, 45, 34, 16, 18, 17, 19, 10, 25, 24, 1, 36, 7, 33, 20, 13, 31], [22, 32, 0, 34, 37, 18, 12, 8, 49, 21, 42, 47, 10, 5, 44, 16, 20, 40, 9, 26, 35, 36, 2, 14, 38, 1, 43, 39, 31, 11, 4, 23, 29, 17, 15, 41, 3, 33, 13, 30, 27, 25, 19, 45, 6, 28, 48, 24, 7, 46], [24, 27, 37, 38, 3, 16, 32, 11, 30, 2, 42, 15, 40, 23, 44, 26, 49, 1, 41, 0, 47, 13, 18, 31, 5, 28, 9, 6, 45, 25, 8, 10, 21, 4, 22, 34, 33, 46, 29, 36, 43, 14, 39, 19, 12, 48, 17, 20, 7, 35], [41, 8, 2, 43, 5, 27, 40, 15, 11, 29, 45, 24, 12, 14, 38, 39, 18, 23, 22, 17, 7, 6, 13, 32, 42, 1, 30, 0, 48, 9, 33, 19, 28, 16, 47, 3, 21, 10, 26, 4, 49, 20, 34, 25, 46, 44, 35, 37, 36, 31], [26, 42, 31, 23, 48, 37, 49, 40, 20, 27, 17, 8, 16, 3, 29, 22, 21, 0, 34, 4, 2, 15, 18, 12, 45, 41, 25, 46, 24, 36, 14, 30, 47, 44, 6, 19, 39, 1, 7, 43, 32, 9, 38, 10, 5, 11, 13, 28, 35, 33], [16, 22, 1, 49, 23, 2, 41, 42, 46, 40, 0, 10, 34, 12, 17, 15, 37, 13, 14, 27, 21, 18, 26, 32, 6, 33, 20, 29, 30, 19, 9, 8, 45, 39, 36, 48, 4, 47, 25, 3, 44, 11, 7, 28, 5, 43, 31, 24, 38, 35], [0, 27, 16, 30, 29, 46, 23, 45, 21, 19, 26, 18, 13, 17, 48, 40, 41, 37, 6, 8, 39, 9, 22, 32, 49, 24, 4, 10, 14, 5, 15, 2, 33, 3, 25, 11, 7, 44, 1, 28, 12, 43, 34, 42, 47, 35, 20, 36, 31, 38], [0, 17, 42, 34, 40, 19, 30, 28, 48, 5, 7, 38, 25, 39, 21, 41, 46, 8, 12, 45, 10, 14, 32, 20, 4, 11, 43, 37, 23, 31, 27, 35, 16, 9, 26, 18, 49, 36, 24, 6, 13, 47, 44, 1, 15, 2, 22, 29, 33, 3], [20, 18, 22, 44, 7, 28, 37, 40, 25, 41, 9, 38, 15, 6, 5, 42, 17, 29, 32, 23, 19, 39, 21, 30, 2, 12, 11, 49, 16, 10, 31, 4, 35, 27, 47, 8, 3, 34, 13, 26, 0, 46, 45, 1, 48, 36, 14, 24, 33, 43], [14, 25, 12, 28, 37, 43, 42, 5, 1, 29, 39, 22, 26, 33, 27, 0, 9, 10, 3, 44, 23, 15, 8, 17, 34, 11, 49, 45, 30, 32, 31, 7, 20, 41, 46, 16, 40, 6, 18, 47, 48, 21, 19, 4, 2, 35, 24, 13, 36, 38], [16, 12, 13, 27, 42, 4, 15, 34, 0, 39, 38, 8, 49, 26, 40, 30, 48, 23, 32, 43, 24, 1, 6, 22, 11, 29, 17, 10, 21, 18, 46, 9, 37, 41, 7, 20, 28, 3, 47, 45, 35, 5, 14, 25, 19, 36, 2, 44, 31, 33], [42, 15, 9, 12, 13, 39, 5, 37, 23, 14, 4, 29, 21, 35, 2, 49, 48, 10, 18, 46, 44, 11, 30, 40, 7, 19, 41, 32, 34, 16, 27, 3, 22, 28, 0, 26, 25, 17, 8, 45, 1, 31, 20, 43, 47, 6, 38, 36, 24, 33], [5, 6, 37, 45, 30, 41, 2, 39, 26, 14, 10, 11, 15, 23, 42, 25, 34, 33, 44, 18, 47, 12, 21, 36, 29, 40, 35, 0, 28, 8, 1, 27, 46, 43, 20, 16, 22, 48, 19, 24, 3, 38, 49, 13, 4, 32, 17, 9, 31, 7], [11, 40, 45, 29, 20, 8, 21, 0, 2, 16, 7, 27, 3, 23, 28, 14, 32, 26, 48, 13, 30, 9, 39, 6, 1, 37, 41, 22, 43, 4, 12, 35, 18, 15, 44, 42, 25, 38, 19, 46, 10, 17, 31, 5, 24, 49, 47, 36, 33, 34]], [[25, 34, 17, 0, 21, 6, 26, 46, 4, 10, 36, 41, 20, 39, 27, 22, 28, 16, 9, 40, 15, 33, 19, 47, 31, 5, 38, 12, 43, 13, 8, 2, 18, 37, 14, 23, 32, 11, 35, 1, 30, 45, 44, 7, 24, 49, 29, 42, 48, 3], [14, 18, 36, 29, 31, 10, 35, 21, 47, 2, 41, 11, 12, 23, 6, 4, 43, 32, 5, 13, 8, 15, 7, 9, 22, 34, 24, 42, 16, 40, 26, 25, 20, 3, 0, 38, 1, 45, 39, 19, 44, 27, 17, 30, 37, 28, 49, 48, 46, 33], [28, 26, 4, 48, 24, 10, 30, 43, 12, 0, 38, 35, 15, 40, 36, 29, 5, 45, 22, 13, 41, 47, 31, 7, 8, 20, 2, 1, 9, 18, 14, 32, 23, 49, 42, 21, 11, 25, 6, 16, 33, 37, 3, 17, 19, 39, 27, 44, 34, 46], [8, 21, 4, 7, 26, 43, 19, 42, 25, 45, 2, 38, 20, 37, 35, 40, 24, 5, 39, 36, 12, 34, 30, 6, 28, 14, 23, 17, 46, 9, 33, 27, 48, 31, 15, 10, 47, 32, 49, 11, 18, 0, 13, 3, 16, 29, 22, 41, 44, 1], [24, 28, 31, 21, 11, 35, 26, 48, 45, 10, 25, 46, 5, 8, 22, 30, 3, 23, 17, 16, 20, 41, 14, 47, 40, 42, 15, 34, 43, 13, 38, 44, 0, 32, 37, 1, 36, 12, 19, 6, 4, 7, 29, 39, 2, 49, 27, 9, 18, 33], [2, 46, 14, 22, 35, 38, 4, 21, 16, 23, 36, 29, 12, 11, 10, 34, 30, 31, 25, 5, 26, 20, 42, 24, 15, 7, 9, 6, 41, 39, 40, 0, 33, 45, 48, 28, 47, 19, 44, 13, 37, 43, 8, 3, 18, 32, 17, 49, 1, 27], [1, 18, 35, 4, 25, 7, 14, 3, 41, 6, 28, 10, 42, 49, 16, 12, 31, 5, 15, 11, 22, 2, 21, 39, 0, 36, 24, 44, 43, 29, 32, 34, 47, 9, 33, 8, 26, 17, 13, 19, 30, 40, 37, 23, 45, 27, 46, 38, 48, 20], [24, 26, 13, 21, 5, 43, 48, 44, 1, 49, 30, 12, 20, 4, 36, 31, 40, 0, 45, 32, 35, 17, 39, 8, 10, 41, 47, 6, 28, 29, 11, 16, 34, 38, 46, 25, 14, 3, 7, 18, 33, 9, 42, 15, 37, 19, 22, 27, 2, 23], [21, 26, 40, 38, 47, 9, 11, 30, 42, 36, 0, 15, 31, 23, 19, 16, 41, 2, 12, 34, 32, 35, 6, 29, 20, 28, 14, 39, 25, 24, 43, 5, 7, 13, 45, 18, 1, 8, 49, 22, 48, 4, 17, 10, 33, 46, 3, 37, 44, 27], [26, 9, 43, 12, 6, 17, 10, 34, 2, 21, 16, 28, 14, 36, 11, 39, 24, 25, 47, 40, 30, 45, 18, 37, 19, 4, 42, 31, 5, 15, 0, 32, 22, 29, 23, 35, 8, 27, 44, 3, 48, 38, 41, 13, 33, 20, 7, 46, 1, 49], [43, 15, 10, 35, 41, 18, 2, 34, 14, 23, 1, 37, 31, 4, 11, 36, 25, 47, 26, 12, 49, 16, 40, 42, 38, 22, 9, 20, 39, 6, 0, 8, 13, 48, 32, 46, 44, 30, 7, 27, 19, 5, 21, 33, 17, 29, 3, 45, 24, 28], [30, 41, 24, 26, 4, 10, 33, 22, 46, 37, 11, 1, 9, 29, 3, 35, 49, 34, 47, 28, 44, 15, 40, 43, 48, 39, 23, 36, 7, 13, 38, 16, 0, 18, 21, 12, 5, 32, 14, 8, 31, 6, 2, 20, 17, 42, 27, 45, 25, 19], [14, 28, 13, 35, 2, 40, 10, 43, 25, 21, 34, 42, 33, 17, 26, 44, 4, 12, 15, 22, 30, 41, 9, 47, 11, 36, 24, 32, 48, 0, 5, 49, 20, 31, 16, 18, 6, 1, 38, 19, 37, 29, 39, 8, 45, 3, 23, 7, 46, 27], [17, 49, 30, 4, 18, 16, 24, 11, 32, 43, 29, 26, 34, 35, 40, 20, 10, 14, 36, 41, 23, 47, 42, 39, 45, 37, 6, 21, 46, 7, 9, 2, 19, 12, 27, 15, 25, 0, 33, 31, 28, 5, 3, 1, 8, 22, 13, 44, 38, 48], [4, 6, 8, 11, 47, 5, 18, 23, 2, 13, 49, 0, 36, 10, 26, 40, 32, 30, 17, 24, 7, 15, 35, 28, 12, 19, 9, 16, 21, 45, 41, 1, 3, 31, 37, 25, 20, 42, 29, 46, 22, 44, 39, 43, 14, 48, 34, 33, 27, 38], [37, 26, 23, 12, 19, 15, 36, 30, 29, 40, 21, 25, 31, 22, 45, 43, 42, 1, 24, 8, 9, 33, 34, 18, 39, 32, 10, 4, 28, 0, 14, 41, 16, 27, 5, 17, 38, 6, 35, 47, 11, 48, 3, 49, 46, 7, 20, 44, 2, 13], [12, 39, 34, 8, 10, 30, 18, 35, 16, 36, 43, 38, 40, 24, 28, 47, 9, 7, 6, 4, 26, 31, 21, 2, 19, 42, 14, 33, 11, 1, 45, 3, 32, 0, 41, 49, 13, 15, 44, 23, 48, 22, 5, 20, 17, 37, 29, 46, 25, 27], [6, 40, 22, 0, 18, 13, 34, 4, 23, 41, 47, 26, 42, 28, 36, 10, 38, 49, 43, 17, 12, 31, 16, 33, 19, 15, 21, 44, 20, 25, 35, 2, 14, 39, 32, 29, 9, 46, 11, 5, 24, 8, 30, 37, 3, 27, 45, 48, 7, 1], [32, 26, 12, 46, 5, 19, 18, 23, 47, 14, 43, 39, 4, 2, 6, 45, 10, 38, 33, 41, 24, 30, 13, 20, 3, 9, 35, 8, 28, 15, 16, 27, 11, 7, 31, 29, 1, 21, 40, 36, 25, 49, 17, 0, 22, 37, 34, 42, 48, 44], [35, 40, 15, 26, 25, 41, 28, 4, 11, 13, 49, 32, 12, 7, 0, 39, 47, 21, 34, 33, 17, 5, 2, 16, 38, 20, 22, 31, 30, 37, 27, 10, 24, 45, 42, 6, 43, 19, 48, 23, 29, 18, 9, 36, 1, 46, 14, 3, 8, 44], [35, 43, 29, 22, 44, 32, 48, 38, 6, 21, 4, 45, 42, 26, 19, 25, 40, 30, 41, 0, 14, 23, 46, 28, 11, 17, 47, 36, 37, 33, 7, 20, 15, 12, 2, 10, 27, 16, 9, 8, 34, 31, 13, 39, 5, 3, 18, 24, 49, 1], [14, 35, 4, 22, 9, 27, 21, 26, 19, 11, 48, 18, 15, 16, 34, 36, 30, 5, 10, 37, 23, 1, 20, 33, 24, 41, 31, 42, 25, 6, 8, 12, 39, 43, 17, 38, 3, 40, 28, 29, 0, 32, 49, 44, 7, 13, 46, 45, 47, 2], [31, 10, 24, 15, 26, 2, 32, 23, 6, 43, 12, 17, 28, 0, 40, 20, 47, 1, 11, 42, 18, 45, 16, 4, 7, 37, 49, 5, 21, 19, 14, 3, 8, 36, 33, 46, 9, 30, 35, 25, 41, 22, 27, 34, 13, 48, 39, 29, 38, 44], [28, 12, 25, 41, 18, 49, 4, 8, 42, 2, 29, 22, 35, 16, 23, 20, 15, 7, 21, 33, 26, 17, 14, 30, 47, 40, 9, 31, 0, 44, 46, 39, 34, 43, 45, 36, 24, 48, 13, 32, 10, 37, 5, 11, 27, 1, 19, 6, 3, 38], [11, 10, 16, 21, 17, 15, 0, 43, 9, 4, 41, 31, 28, 40, 34, 25, 36, 46, 24, 49, 35, 45, 27, 23, 14, 30, 26, 18, 13, 32, 2, 39, 47, 42, 38, 5, 1, 20, 33, 22, 44, 12, 3, 6, 8, 19, 37, 7, 48, 29], [48, 34, 9, 0, 42, 14, 21, 46, 25, 17, 28, 15, 36, 10, 44, 29, 4, 3, 31, 16, 7, 47, 22, 41, 8, 39, 5, 37, 24, 30, 6, 1, 35, 12, 20, 27, 49, 33, 40, 45, 11, 2, 26, 38, 23, 18, 43, 13, 32, 19], [30, 11, 7, 36, 0, 21, 25, 41, 38, 29, 9, 4, 15, 5, 44, 39, 10, 8, 26, 14, 31, 17, 28, 6, 3, 33, 32, 16, 49, 42, 35, 12, 45, 20, 2, 43, 34, 37, 22, 47, 40, 24, 27, 19, 13, 23, 1, 18, 48, 46], [1, 11, 18, 46, 25, 22, 48, 0, 43, 21, 6, 40, 13, 32, 10, 37, 26, 17, 27, 24, 3, 23, 41, 9, 30, 29, 31, 28, 20, 36, 15, 8, 19, 14, 38, 39, 34, 16, 2, 12, 35, 47, 49, 42, 45, 4, 5, 7, 33, 44], [26, 40, 17, 7, 19, 35, 31, 39, 5, 25, 18, 9, 12, 6, 2, 0, 24, 8, 3, 28, 29, 37, 4, 21, 10, 43, 32, 45, 13, 23, 11, 16, 49, 36, 46, 42, 41, 47, 14, 38, 15, 30, 22, 48, 33, 20, 27, 44, 1, 34], [44, 40, 1, 33, 31, 27, 38, 34, 30, 32, 0, 36, 6, 28, 10, 5, 41, 18, 20, 42, 15, 24, 17, 26, 4, 2, 25, 35, 7, 37, 48, 43, 21, 14, 13, 16, 11, 29, 47, 19, 8, 9, 12, 39, 3, 23, 45, 46, 49, 22], [41, 14, 20, 18, 28, 44, 29, 6, 35, 13, 42, 10, 22, 39, 9, 31, 32, 11, 7, 17, 24, 43, 26, 5, 38, 27, 47, 21, 3, 8, 33, 15, 36, 16, 30, 0, 23, 45, 1, 40, 2, 4, 48, 12, 37, 34, 46, 49, 19, 25], [18, 35, 2, 32, 14, 9, 43, 15, 48, 42, 41, 12, 24, 44, 31, 30, 38, 36, 22, 21, 28, 6, 19, 40, 23, 46, 0, 47, 20, 4, 16, 5, 34, 11, 3, 45, 39, 25, 29, 10, 13, 17, 7, 8, 26, 33, 37, 49, 1, 27], [42, 6, 9, 46, 30, 23, 11, 4, 35, 22, 10, 14, 26, 13, 32, 15, 3, 47, 41, 39, 27, 36, 7, 16, 29, 34, 20, 2, 31, 33, 43, 5, 28, 19, 25, 24, 1, 18, 8, 0, 37, 21, 40, 17, 44, 12, 49, 48, 38, 45], [4, 32, 30, 26, 10, 42, 6, 31, 12, 0, 46, 16, 49, 19, 17, 41, 3, 36, 28, 35, 13, 39, 9, 8, 43, 48, 21, 47, 15, 38, 2, 14, 29, 23, 18, 24, 22, 20, 45, 34, 11, 7, 27, 40, 25, 1, 33, 5, 44, 37], [36, 47, 3, 14, 18, 32, 4, 44, 0, 34, 26, 8, 41, 31, 7, 9, 1, 17, 13, 38, 16, 10, 5, 27, 33, 22, 2, 28, 48, 6, 39, 40, 15, 37, 30, 21, 35, 20, 23, 24, 43, 11, 45, 42, 19, 49, 12, 46, 29, 25], [21, 32, 44, 0, 39, 7, 15, 19, 36, 41, 45, 4, 35, 8, 16, 38, 1, 40, 13, 26, 5, 18, 47, 14, 24, 20, 28, 48, 25, 43, 31, 10, 6, 17, 30, 2, 33, 23, 46, 37, 22, 12, 11, 42, 49, 34, 9, 3, 29, 27], [45, 8, 4, 17, 35, 36, 41, 21, 11, 28, 23, 14, 30, 0, 10, 5, 26, 40, 3, 32, 34, 9, 6, 7, 38, 22, 43, 31, 24, 33, 39, 49, 15, 46, 13, 25, 20, 42, 1, 44, 27, 2, 19, 37, 48, 18, 47, 12, 16, 29], [48, 14, 17, 40, 23, 44, 11, 31, 8, 4, 43, 0, 1, 39, 3, 19, 25, 15, 2, 21, 42, 20, 33, 32, 46, 37, 30, 45, 28, 10, 18, 35, 6, 26, 49, 24, 36, 38, 7, 27, 16, 29, 34, 5, 13, 47, 41, 22, 9, 12], [21, 22, 26, 41, 37, 3, 5, 19, 13, 2, 45, 40, 38, 35, 12, 0, 20, 23, 15, 8, 48, 17, 25, 33, 10, 30, 42, 47, 49, 28, 43, 24, 39, 4, 6, 32, 16, 11, 31, 44, 7, 29, 36, 18, 9, 46, 14, 34, 27, 1], [42, 10, 47, 34, 15, 6, 0, 38, 40, 25, 32, 43, 2, 21, 5, 30, 22, 37, 14, 20, 44, 36, 31, 4, 11, 18, 33, 9, 46, 24, 48, 17, 1, 13, 39, 49, 16, 35, 23, 12, 8, 7, 28, 19, 41, 26, 29, 3, 27, 45], [20, 11, 34, 22, 10, 47, 27, 35, 38, 39, 46, 26, 42, 36, 28, 43, 21, 31, 2, 6, 32, 9, 16, 49, 15, 3, 18, 12, 25, 29, 13, 44, 14, 30, 41, 8, 40, 7, 23, 5, 4, 17, 45, 48, 37, 24, 19, 0, 1, 33], [21, 26, 22, 1, 12, 13, 42, 36, 3, 43, 28, 38, 33, 10, 23, 35, 34, 32, 30, 18, 8, 47, 15, 40, 37, 31, 4, 46, 17, 29, 6, 25, 49, 39, 48, 0, 19, 27, 41, 2, 24, 20, 11, 14, 45, 16, 9, 7, 5, 44], [37, 6, 17, 1, 43, 20, 5, 0, 36, 2, 47, 8, 38, 25, 10, 35, 28, 27, 22, 32, 11, 41, 21, 31, 14, 40, 39, 12, 34, 9, 30, 16, 42, 26, 13, 33, 3, 24, 15, 48, 29, 18, 4, 23, 49, 44, 45, 46, 7, 19], [36, 37, 5, 33, 26, 14, 38, 11, 39, 17, 47, 16, 9, 25, 6, 48, 49, 21, 31, 43, 30, 8, 41, 12, 40, 13, 24, 15, 4, 42, 32, 19, 46, 28, 34, 0, 10, 27, 20, 23, 29, 2, 44, 35, 22, 18, 45, 3, 1, 7], [21, 34, 47, 42, 23, 36, 33, 49, 9, 13, 28, 26, 31, 11, 14, 2, 8, 18, 17, 35, 41, 0, 12, 16, 39, 40, 27, 43, 37, 10, 3, 20, 24, 19, 22, 7, 4, 6, 25, 30, 5, 46, 15, 1, 32, 29, 48, 38, 44, 45], [43, 6, 41, 32, 13, 42, 7, 22, 36, 40, 26, 9, 44, 39, 35, 28, 5, 12, 11, 33, 16, 4, 25, 15, 30, 31, 10, 47, 24, 2, 20, 0, 19, 23, 38, 49, 21, 1, 17, 27, 45, 37, 3, 29, 34, 14, 8, 18, 48, 46], [4, 23, 41, 37, 39, 31, 6, 24, 36, 11, 43, 35, 30, 8, 13, 49, 16, 19, 22, 10, 7, 12, 48, 33, 45, 21, 40, 14, 47, 38, 26, 20, 32, 17, 15, 42, 46, 28, 18, 9, 3, 25, 29, 0, 5, 34, 2, 1, 27, 44], [6, 26, 25, 0, 27, 21, 19, 34, 9, 7, 23, 22, 5, 14, 40, 12, 43, 2, 47, 28, 15, 41, 36, 24, 30, 35, 38, 8, 32, 18, 11, 39, 31, 4, 20, 42, 16, 33, 29, 49, 1, 48, 37, 13, 10, 3, 45, 17, 44, 46], [35, 29, 12, 20, 1, 13, 15, 39, 23, 17, 36, 4, 19, 40, 11, 47, 27, 21, 14, 2, 46, 16, 9, 26, 32, 18, 30, 10, 5, 28, 49, 41, 22, 45, 37, 31, 0, 38, 3, 24, 42, 43, 8, 33, 6, 48, 7, 44, 25, 34], [26, 30, 46, 17, 29, 43, 10, 4, 9, 7, 33, 36, 40, 6, 31, 14, 37, 47, 24, 48, 16, 15, 20, 39, 34, 22, 12, 38, 28, 42, 35, 2, 41, 5, 13, 21, 1, 45, 8, 3, 27, 25, 11, 19, 23, 44, 32, 0, 49, 18]]]
#         preflist = [[[9, 4, 17, 8, 7, 5, 6, 1, 3, 14, 11, 15, 0, 2, 13, 12, 16, 10], [14, 15, 17, 7, 2, 12, 3, 11, 13, 8, 5, 1, 0, 9, 4, 6, 10, 16], [17, 7, 14, 5, 6, 15, 0, 13, 12, 8, 2, 1, 3, 11, 9, 10, 4, 16], [7, 15, 6, 17, 2, 9, 11, 13, 0, 5, 4, 3, 12, 1, 14, 8, 10, 16], [0, 15, 7, 14, 5, 13, 6, 2, 12, 11, 8, 17, 1, 9, 3, 4, 10, 16], [8, 0, 5, 9, 15, 14, 7, 13, 2, 16, 3, 11, 12, 1, 6, 10, 17, 4], [9, 15, 13, 11, 8, 12, 5, 3, 6, 0, 7, 2, 17, 14, 1, 4, 16, 10], [15, 2, 9, 12, 11, 14, 6, 3, 13, 4, 5, 1, 17, 8, 7, 0, 10, 16], [7, 11, 15, 5, 9, 6, 12, 8, 3, 17, 0, 14, 2, 1, 13, 10, 4, 16], [8, 7, 5, 0, 6, 3, 4, 11, 2, 17, 14, 15, 16, 9, 12, 1, 13, 10], [2, 8, 5, 11, 3, 13, 15, 6, 1, 17, 7, 0, 14, 9, 12, 4, 16, 10], [15, 9, 3, 0, 13, 5, 11, 17, 12, 2, 8, 14, 6, 10, 7, 1, 4, 16], [17, 12, 8, 11, 7, 15, 13, 0, 14, 9, 2, 6, 5, 3, 1, 4, 10, 16], [15, 0, 9, 3, 11, 5, 6, 13, 7, 2, 14, 17, 1, 12, 4, 8, 10, 16], [2, 15, 13, 5, 8, 17, 0, 9, 11, 14, 3, 7, 6, 1, 12, 4, 10, 16], [15, 11, 2, 3, 13, 7, 8, 12, 9, 17, 5, 1, 6, 16, 0, 14, 10, 4], [13, 11, 7, 15, 9, 12, 5, 3, 8, 6, 2, 17, 14, 0, 1, 4, 16, 10], [15, 11, 2, 17, 10, 5, 13, 8, 12, 14, 9, 7, 1, 0, 3, 6, 4, 16]], [[9, 13, 3, 14, 7, 10, 12, 2, 5, 11, 4, 0, 15, 1, 17, 16, 6, 8], [14, 12, 10, 0, 9, 2, 15, 7, 3, 16, 4, 11, 5, 1, 13, 17, 6, 8], [4, 2, 7, 6, 9, 3, 5, 13, 12, 11, 0, 16, 15, 17, 10, 14, 1, 8], [4, 7, 13, 9, 6, 10, 2, 12, 0, 8, 3, 16, 5, 11, 1, 14, 15, 17], [4, 9, 3, 7, 13, 5, 11, 16, 17, 6, 15, 2, 1, 12, 10, 0, 8, 14], [9, 13, 5, 3, 2, 7, 14, 4, 11, 0, 12, 10, 17, 6, 1, 16, 15, 8], [4, 2, 12, 7, 6, 5, 17, 3, 11, 9, 15, 0, 14, 10, 8, 13, 1, 16], [7, 9, 3, 12, 11, 5, 2, 17, 0, 10, 1, 15, 4, 13, 8, 6, 14, 16], [0, 3, 2, 7, 15, 11, 5, 14, 9, 13, 4, 17, 12, 8, 1, 16, 10, 6], [11, 7, 0, 5, 16, 15, 9, 2, 14, 17, 3, 13, 10, 1, 12, 4, 6, 8], [12, 9, 16, 3, 5, 7, 17, 2, 4, 13, 0, 11, 6, 14, 10, 15, 1, 8], [2, 4, 3, 12, 17, 9, 6, 15, 5, 1, 13, 0, 7, 11, 14, 10, 16, 8], [12, 17, 2, 7, 4, 0, 14, 3, 9, 5, 15, 13, 11, 1, 6, 16, 10, 8], [12, 2, 9, 5, 3, 7, 16, 13, 6, 0, 14, 11, 15, 4, 1, 10, 8, 17], [12, 10, 5, 9, 7, 3, 0, 17, 14, 13, 15, 4, 2, 11, 16, 1, 6, 8], [2, 0, 9, 3, 17, 16, 6, 8, 7, 5, 12, 10, 4, 15, 13, 1, 14, 11], [3, 7, 17, 2, 12, 5, 13, 1, 9, 8, 15, 4, 16, 0, 11, 10, 14, 6], [12, 9, 2, 17, 7, 4, 0, 5, 15, 8, 10, 13, 3, 6, 11, 14, 16, 1]]]
#         # preflist = create_preflist(num_agents)
#         # print_preflist(preflist)
#         # preflist = uniform_instance_generator(num_agents)
#         # preflist = triangular_instance_generator(num_agents)
#         # preflist = normal_instance_generator(num_agents)
#         # if iter == 2:
#         #     print(preflist)
#         # print("Male optimal matching (Men propose to Women):")
#         male_optimal_matching = gale_shapley(preflist)
#         print_matching(male_optimal_matching)
#         # print(preflist)
#         copy_preflist = copy.deepcopy(preflist)
#         men_shortlists, women_shortlists = create_shortlists(copy_preflist, male_optimal_matching)

#         print("Male Shortlists:")
#         print_shortlists(men_shortlists)
        
#         print("Female Shortlists:")
#         print_shortlists(women_shortlists)

#         copy_men_shortlists = copy.deepcopy(men_shortlists)
#         # create an copy by value of mens shortlists
#         copy_women_shortlists = copy.deepcopy(women_shortlists)
#         # create an copy by value of womens shortlists

#         rotations = []
#         # first_rotation = None
#         while True:
#             new_rotation = find_a_rotation(copy_men_shortlists)
#             if new_rotation is None:
#                 break
#             rotations.append(new_rotation)

#             eliminate_rotation(new_rotation, copy_men_shortlists, copy_women_shortlists)
#             # print("Found new rotation:", new_rotation)
#             # print("Men Shortlists after eliminating rotation:", new_rotation)
#             # print_shax.set_aspect('equal')ortlists(copy_men_shortlists)
#             # print("Women Shortlists after eliminating rotation:", new_rotation)
#             # print_shortlists(copy_women_shortlists)
#         weights = assign_weights(rotations, copy_preflist)
#         print_rotations(rotations, weights)

#         copy_men_shortlists = copy.deepcopy(men_shortlists)
#         # create an copy by value of mens shortlists

#         copy_women_shortlists = copy.deepcopy(women_shortlists)

#         graph = create_rotation_digraph(rotations, copy_men_shortlists, women_shortlists)
#         print_graph(graph)

#         pred = predecessors(graph)
#         # print("Predecessors of each node:")
#         # print_graph(pred)
#         topo_order = topological_sort(graph, pred)
#         print("Topological Order:", topo_order)
#         closed_subsets = closed_subset_finder(topo_order, pred)
#         # print("Number of Closed subsets", len(closed_subsets))
#         # print(closed_subsets)
#         # (weights_1, weights_2) = assign_weights(rotations, preflist)
#         # print("Weights of summation of ranks criterion:")
#         # print_weights(weights_1)
#         # print("Weights of Nash Social Welfare")
#         # print_weights(weights_2)
#         # optimal_closed_subset_1 = max_weight_subset_1(closed_subsets, weights_1)
#         # optimal_closed_subset_2 = max_weight_subset_2(closed_subsets, weights_2)

#         # # print(optimal_closed_subset_1)
#         # # print(optimal_closed_subset_2)

#         # print("Men Shortlists")
#         # print_shortlists(copy_men_shortlists)
#         # print("Women Shortlists")
#         # print_shortlists(copy_women_shortlists)
#         # min_regret, 
#         min_sum_val, Eg_closed_subset = float('inf'), None
#         # min_disparity, 
#         max_nsw_val, Snsw_closed_subset = float('-inf'), None
#         # float('inf'), float('-inf')
#         # min_nsw_val = float('inf')
#         # min_regret_matching, egalitarian_matching, sex_equal_matching, nsw_matching = None, None, None, None
#         # print(len(closed_subsets), "closed subsets found.")
#         for subset in closed_subsets:
#             copy_men_shortlists = copy.deepcopy(men_shortlists)
#             copy_women_shortlists = copy.deepcopy(women_shortlists)
#             matching_1 = stable_matching(subset, rotations, topo_order, copy_men_shortlists, copy_women_shortlists)
#             # print(subset)
#             # print_matching(matching_1)
#             # regret = c(matching_1, preflist)
#             summation = d(matching_1, preflist)
#             # disparity = e(matching_1, preflist)
#             nash_social_welfare = nsw(matching_1, preflist)
#             # print(nash_social_welfare)
#             # ans.append(nash_social_welfare)
#             # if regret < min_regret:
#             #     min_regret = regret
#             #     min_regret_matching = matching_1
#             if summation < min_sum_val:
#                 min_sum_val = summation
#                 egalitarian_matching = matching_1
#                 Eg_closed_subset = subset
#             # if disparity < min_disparity:
#             #     min_disparity = disparity
#             #     sex_equal_matching = matching_1
#                 # print(subset)
#             if nash_social_welfare > max_nsw_val:
#                 max_nsw_val = nash_social_welfare
#                 nsw_matching = matching_1
#                 Snsw_closed_subset = subset
#         print("Egalitarian Matching:")
#         print_matching(egalitarian_matching)
#         print(Eg_closed_subset)
#         print("NSW Matching:")
#         print_matching(nsw_matching)
#         print(Snsw_closed_subset)
        # intersection = (set(Snsw_closed_subset) & set(Eg_closed_subset))
        # if intersection == set(Snsw_closed_subset):
        #     continue
        # elif intersection == set(Eg_closed_subset):
        #     continue
        # else:
        #     print(preflist)
#     #     if nash_social_welfare < min_nsw_val:
#     #         min_nsw_val = nash_social_welfare
#     #         # anti_nsw_matching = matching_1
#     # ratio = max_nsw_val / min_nsw_val
#     # max_ratio = max(max_ratio, ratio)
#     # min_ratio = min(min_ratio, ratio)
#     # c_1 = c(min_regret_matching, preflist)
#     # c_1_list.append(c_1)
#     # d_1 = d(min_regret_matching, preflist)
#     # d_1_list.append(d_1)
#     # e_1 = e(min_regret_matching, preflist)
#     # e_1_list.append(e_1)
#     # nsw_1 = nsw(min_regret_matching, preflist)
#     # nsw_1_list.append(nsw_1)

#     # c_2 = c(egalitarian_matching, preflist)
#     # c_2_list.append(c_2)
#     # d_2 = d(egalitarian_matching, preflist)
#     # d_2_list.append(d_2)
#     # e_2 = e(egalitarian_matching, preflist)
#     # e_2_list.append(e_2)
#     # nsw_2 = nsw(egalitarian_matching, preflist)
#     # print(nsw_2)
#     # nsw_2_list.append(nsw_2)
#     # # print("Matching for summation of ranks criterion:")
#     # # print_matching(matching_1, False)

#     # # copy_men_shortlists = copy.deepcopy(male_shortlists)
#     # # copy_women_shortlists = copy.deepcopy(female_shortlists)

#     # # matching_2 = stable_matching(optimal_closed_subset_2, rotations, topo_order, copy_men_shortlists, copy_women_shortlists)
#     # # c_2_m, c_2_w = worst_rank(nsw_matching, preflist)

#     # c_3 = c(sex_equal_matching, preflist)
#     # c_3_list.append(c_3)
#     # d_3 = d(sex_equal_matching, preflist)
#     # d_3_list.append(d_3)
#     # e_3 = e(sex_equal_matching, preflist)
#     # e_3_list.append(e_3)
#     # nsw_3 = nsw(sex_equal_matching, preflist)
#     # nsw_3_list.append(nsw_3)

#     # c_4 = c(nsw_matching, preflist)
#     # c_4_list.append(c_4)
#     # d_4 = d(nsw_matching, preflist)
#     # d_4_list.append(d_4)
#     # e_4 = e(nsw_matching, preflist)
#     # e_4_list.append(e_4)
#     # nsw_4 = nsw(nsw_matching, preflist)
#     # print(nsw_4)
#     # nsw_4_list.append(nsw_4)
#     # print("Average = ", math.prod(ans) ** (1/len(ans)))
#     # data = {
#     #     "preflist": convert_to_builtin(preflist),
#     #     "min_regret": min_regret_matching,
#     #     "egalitarian": egalitarian_matching,
#     #     "sex_equal": sex_equal_matching,
#     #     "nsw": nsw_matching,
#     #     "scores": {
#     #         "c": [float(c_1), float(c_2), float(c_3), float(c_4)],
#     #         "d": [float(d_1), float(d_2), float(d_3), float(d_4)],
#     #         "e": [float(e_1), float(e_2), float(e_3), float(e_4)],
#     #         "nsw": [float(nsw_1), float(nsw_2), float(nsw_3), float(nsw_4)]
#     #     }
#     # }
#     # results_file.write(json.dumps(convert_to_builtin(data)) + "\n")

#     # print(d_1-d_2)
#     # if c_2 - c_1 >= 2 and d_1 - d_2 >=0.2:
#     #     print_preflist(preflist)
#     #     print("Min Regret Matching:")
#     #     print_matching(min_regret_matching)
#     #     print("Egalitarian Matching:")
#     #     print_matching(egalitarian_matching)
#     #     print("min-regret=", c_1)
#     #     print("suboptimal regret=", c_2)
#     #     print("min-sum=", d_2)
#     #     print("suboptimal sum=", d_1)
#                 # print("Sex-equal Matching:")
#                 # print_matching(sex_equal_matching)
#                 # print("NSW Matching:")
#                 # print_matching(nsw_matching)
# #     nsw_1 = nsw(nsw_matching, preflist)
# #     nsw_gs = nsw(male_optimal_matching, preflist)
# #     nsw_anti = nsw(anti_nsw_matching, preflist)
# #     nsw_1_list.append(nsw_1)
# #     nsw_2_list.append(nsw_gs)
# #     nsw_3_list.append(nsw_anti)
#     # nsw_alg_1_avg.append(np.mean(nsw_1_list))
#     # nsw_alg_2_avg.append(np.mean(nsw_2_list))
#     # nsw_alg_3_avg.append(np.mean(nsw_3_list))
#     # nsw_alg_1_var.append(np.var(nsw_1_list))
#     # nsw_alg_2_var.append(np.var(nsw_2_list))
#     # nsw_alg_3_var.append(np.var(nsw_3_list))
#     # print(f"Number of agents: {num_agents} done.")
#     # print(max_ratio, min_ratio)
#     # results_file.close()


#     # print(c_1_list, c_2_list, c_1_list_m, c_1_list_w, c_2_list_m, c_2_list_w)
#     # stats = statistics(c_1_list, c_2_list, c_3_list, c_4_list, d_1_list, d_2_list, d_3_list, d_4_list, e_1_list, e_2_list, e_3_list, e_4_list, nsw_1_list, nsw_2_list, nsw_3_list, nsw_4_list)
#     # c_alg_1_avg.append(stats[0][0]) 
#     # c_alg_2_avg.append(stats[0][1])
#     # c_alg_3_avg.append(stats[0][2])
#     # c_alg_4_avg.append(stats[0][3])
#     # c_alg_1_var.append(stats[1][0]) 
#     # c_alg_2_var.append(stats[1][1])
#     # c_alg_3_var.append(stats[1][2])
#     # c_alg_4_var.append(stats[1][3])
#     # d_alg_1_avg.append(stats[0][4])
#     # d_alg_2_avg.append(stats[0][5])
#     # d_alg_3_avg.append(stats[0][6])
#     # d_alg_4_avg.append(stats[0][7])
#     # d_alg_1_var.append(stats[1][4])
#     # d_alg_2_var.append(stats[1][5])
#     # d_alg_3_var.append(stats[1][6])
#     # d_alg_4_var.append(stats[1][7])
#     # e_alg_1_avg.append(stats[0][8])
#     # e_alg_2_avg.append(stats[0][9])
#     # e_alg_3_avg.append(stats[0][10])
#     # e_alg_4_avg.append(stats[0][11])
#     # e_alg_1_var.append(stats[1][8])
#     # e_alg_2_var.append(stats[1][9])
#     # e_alg_3_var.append(stats[1][10])
#     # e_alg_4_var.append(stats[1][11])
#     # nsw_alg_1_avg.append(stats[0][12])
#     # nsw_alg_2_avg.append(stats[0][13])
#     # nsw_alg_3_avg.append(stats[0][14])
#     # nsw_alg_4_avg.append(stats[0][15])
#     # nsw_alg_1_var.append(stats[1][12])
#     # nsw_alg_2_var.append(stats[1][13])
#     # nsw_alg_3_var.append(stats[1][14])
#     # nsw_alg_4_var.append(stats[1][15])

#     # plot_pairs(num_agents, c_alg_1_avg, c_alg_2_avg, c_alg_3_avg, c_alg_4_avg, \
#     #            d_alg_1_avg, d_alg_2_avg, d_alg_3_avg, d_alg_4_avg, \
#     #            e_alg_1_avg, e_alg_2_avg, e_alg_3_avg, e_alg_4_avg, \
#     #            nsw_alg_1_avg, nsw_alg_2_avg, nsw_alg_3_avg, nsw_alg_4_avg)
#     # plot_3d(num_agents, c_alg_1_avg, c_alg_2_avg, c_alg_3_avg, c_alg_4_avg, \
#     #         d_alg_1_avg, d_alg_2_avg, d_alg_3_avg, d_alg_4_avg, \
#     #         e_alg_1_avg, e_alg_2_avg, e_alg_3_avg, e_alg_4_avg, \
#     #         nsw_alg_1_avg, nsw_alg_2_avg, nsw_alg_3_avg, nsw_alg_4_avg)
#     # plot_circle(num_agents, c_alg_1_avg, c_alg_2_avg, c_alg_3_avg, c_alg_4_avg, \
#     #             d_alg_1_avg, d_alg_2_avg, d_alg_3_avg, d_alg_4_avg, \
#     #             e_alg_1_avg, e_alg_2_avg, e_alg_3_avg, e_alg_4_avg, \
#     #             nsw_alg_1_avg, nsw_alg_2_avg, nsw_alg_3_avg, nsw_alg_4_avg)

# ratios = []
# num_agents = []
analyse(5)
# for n in range(16, 21):
    # print("Number of Agents = ", n)
    # ratios.append(analyse(n))
    # analyse(n)
    # execute(n, 1000)
    # num_agents.append(n)
# execute(18, 1)
# plt.plot(num_agents, ratios)
# plt.show()
# execute(16, 1)
# # agents = []
# # c_alg_1_avg, c_alg_2_avg, c_alg_3_avg, c_alg_4_avg = [], [], [], []
# # c_alg_1_var, c_alg_2_var, c_alg_3_var, c_alg_4_var = [], [], [], []
# # d_alg_1_avg, d_alg_2_avg, d_alg_3_avg, d_alg_4_avg = [], [], [], []
# # d_alg_1_var, d_alg_2_var, d_alg_3_var, d_alg_4_var = [], [], [], []
# # e_alg_1_avg, e_alg_2_avg, e_alg_3_avg, e_alg_4_avg = [], [], [], []
# # e_alg_1_var, e_alg_2_var, e_alg_3_var, e_alg_4_var = [], [], [], []
# # nsw_alg_1_avg, nsw_alg_2_avg, nsw_alg_3_avg, nsw_alg_4_avg = [], [], [], []
# # nsw_alg_1_var, nsw_alg_2_var, nsw_alg_3_var, nsw_alg_4_var = [], [], [], []
# # for num_agents in range(5, 6):
# #     # if num_agents != 5:
# #     #     continue
# #     agents.append(num_agents)
# #     execute(c_alg_1_avg, c_alg_2_avg, c_alg_3_avg, c_alg_4_avg, \
# #             c_alg_1_var, c_alg_2_var, c_alg_3_var, c_alg_4_var, \
# #             d_alg_1_avg, d_alg_2_avg, d_alg_3_avg, d_alg_4_avg, \
# #             d_alg_1_var, d_alg_2_var, d_alg_3_var, d_alg_4_var, \
# #             e_alg_1_avg, e_alg_2_avg, e_alg_3_avg, e_alg_4_avg, \
# #             e_alg_1_var, e_alg_2_var, e_alg_3_var, e_alg_4_var, \
# #             nsw_alg_1_avg, nsw_alg_2_avg, nsw_alg_3_avg, nsw_alg_4_avg, \
# #             nsw_alg_1_var, nsw_alg_2_var, nsw_alg_3_var, nsw_alg_4_var, num_agents)
    
#     # def execute_2(c_alg_1_avg, d_alg_1_avg, e_alg_1_avg, nsw_alg_1_avg, c_alg_2_avg, d_alg_2_avg, e_alg_2_avg, nsw_alg_2_avg, c_alg_1_var, d_alg_1_var, e_alg_1_var, nsw_alg_1_var, \
#     #           c_alg_2_var, d_alg_2_var, e_alg_2_var, nsw_alg_2_var, num_agents):
#     # iters, count = 10000, 0
#     # c_alg_1_list, d_alg_1_list, e_alg_1_list, nsw_alg_1_list = [], [], [], []
#     # c_alg_2_list, d_alg_2_list, e_alg_2_list, nsw_alg_2_list = [], [], [], []
#     # bp_agents_list, bp_nsw_list = [], []
#     # for iter in range(iters):
#     #     if iter % 100 == 0:
#     #         print(iter)
#     #     a = 40
#     #     b = 50
#     #     weight_matrix = create_weight_matrix(num_agents, a, b)
#     #     preflist = create_preference_list(weight_matrix)
#         # male_optimal_matching = gale_shapley(preflist)
#         # copy_preflist = copy.deepcopy(preflist)
#         # men_shortlists, women_shortlists = create_shortlists(copy_preflist, male_optimal_matching)
#         # copy_men_shortlists = copy.deepcopy(men_shortlists)
#         # copy_women_shortlists = copy.deepcopy(women_shortlists)
#         # rotations = []
#         # while True:
#         #     new_rotation = find_a_rotation(copy_men_shortlists)
#         #     if new_rotation is None:
#         #         break
#         #     rotations.append(new_rotation)
#         #     eliminate_rotation(new_rotation, copy_men_shortlists, copy_women_shortlists)
#         # copy_men_shortlists = copy.deepcopy(men_shortlists)
#         # copy_women_shortlists = copy.deepcopy(women_shortlists)
#         # graph = create_rotation_digraph(rotations, copy_men_shortlists, women_shortlists)
#         # pred = predecessors(graph)
#         # topo_order = topological_sort(graph, pred)
#         # closed_subsets = closed_subset_finder(topo_order, pred)
#         # max_nsw__val = float('-inf')
#         # snsw_matching = None
#         # for subset in closed_subsets:
#         #     copy_men_shortlists = copy.deepcopy(men_shortlists)
#         #     copy_women_shortlists = copy.deepcopy(women_shortlists)
#         #     matching_1 = stable_matching(subset, rotations, topo_order, copy_men_shortlists, copy_women_shortlists)
#         #     nash_social_welfare = nsw(matching_1, preflist)
#         #     if nash_social_welfare > max_nsw__val:
#         #         max_nsw__val = nash_social_welfare
#         #         snsw_matching = matching_1
#         # c_1 = c(snsw_matching, preflist)
#         # c_alg_1_list.append(c_1)
#         # d_1 = d(snsw_matching, preflist)
#         # d_alg_1_list.append(d_1)
#         # e_1 = e(snsw_matching, preflist)
#         # e_alg_1_list.append(e_1)
#         # nsw_1 = nsw(snsw_matching, preflist)
#         # nsw_alg_1_list.append(nsw_1)

#         # cost_matrix = -np.array(weight_matrix)
#         # row_ind, col_ind = linear_sum_assignment(cost_matrix)

#         # nsw_matching = list(col_ind)
#         # c_2 = c(nsw_matching, preflist)
#         # c_alg_2_list.append(c_2)
#         # d_2 = d(nsw_matching, preflist)
#         # d_alg_2_list.append(d_2)
#         # e_2 = e(nsw_matching, preflist)
#         # e_alg_2_list.append(e_2)
#         # nsw_2 = nsw(nsw_matching, preflist)
#         # nsw_alg_2_list.append(nsw_2)

#         # copy_preflist = copy.deepcopy(preflist)
#         # bp_nsw = blocking_pairs(nsw_matching, copy_preflist)
#         # bp_nsw_list.append(len(bp_nsw))
#         # bp_men = set()
#         # bp_women = set()
#         # for (m_i, w_i) in bp_nsw:
#         #     bp_men.add(m_i)
#         #     bp_women.add(w_i)
#         # bp_agents_list.append(len(bp_men)+len(bp_women))
        
#         # if nsw_matching == snsw_matching:
#         #     count += 1
#     # c_alg_1_avg.append(np.mean(c_alg_1_list))
#     # d_alg_1_avg.append(np.mean(d_alg_1_list))
#     # e_alg_1_avg.append(np.mean(e_alg_1_list))
#     # nsw_alg_1_avg.append(np.mean(nsw_alg_1_list))
#     # c_alg_2_avg.append(np.mean(c_alg_2_list))
#     # d_alg_2_avg.append(np.mean(d_alg_2_list))
#     # e_alg_2_avg.append(np.mean(e_alg_2_list))
#     # nsw_alg_2_avg.append(np.mean(nsw_alg_2_list))
#     # c_alg_1_var.append(np.var(c_alg_1_list))
#     # d_alg_1_var.append(np.var(d_alg_1_list))
#     # e_alg_1_var.append(np.var(e_alg_1_list))
#     # nsw_alg_1_var.append(np.var(nsw_alg_1_list))
#     # c_alg_2_var.append(np.var(c_alg_2_list))
#     # d_alg_2_var.append(np.var(d_alg_2_list))
#     # e_alg_2_var.append(np.var(e_alg_2_list))
#     # nsw_alg_2_var.append(np.var(nsw_alg_2_list))

#     # print(f"Number of agents: {num_agents} done.")
#     # avg_bp_agents = np.mean(bp_agents_list)
#     # avg_bp_nsw = np.mean(bp_nsw_list)
#     # min_bp_nsw = min(bp_nsw_list)
#     # min_bp_agents = min(bp_agents_list)
#     # max_bp_nsw = max(bp_nsw_list)
#     # max_bp_agents = max(bp_agents_list)
#     # return (avg_bp_nsw, avg_bp_agents, min_bp_nsw, min_bp_agents, max_bp_nsw, max_bp_agents)
#     # return count / iters
#     # plot_circle_2(num_agents, c_alg_1_avg, d_alg_1_avg, e_alg_1_avg, nsw_alg_1_avg, \
#     #         c_alg_2_avg, d_alg_2_avg, e_alg_2_avg, nsw_alg_2_avg)

# agents = []
# c_alg_1_avg, d_alg_1_avg, e_alg_1_avg, nsw_alg_1_avg = [], [], [], []
# c_alg_2_avg, d_alg_2_avg, e_alg_2_avg, nsw_alg_2_avg = [], [], [], []
# c_alg_1_var, d_alg_1_var, e_alg_1_var, nsw_alg_1_var = [], [], [], []
# c_alg_2_var, d_alg_2_var, e_alg_2_var, nsw_alg_2_var = [], [], [], []
# ratio_c, ratio_d, ratio_e, ratio_nsw = [], [], [], []
# avg_bp_len_nsw, avg_bp_len_agents = [], []
# min_bp_len_nsw, min_bp_len_agents = [], []
# max_bp_len_nsw, max_bp_len_agents = [], []
# for num_agents in range(5, 51):
#     agents.append(num_agents)
#     # if num_agents != 50:
#     #   continue
#     (avg_bp_nsw, avg_bp_agents, min_bp_nsw, min_bp_agents, max_bp_nsw, max_bp_agents) = execute_2(c_alg_1_avg, d_alg_1_avg, e_alg_1_avg, nsw_alg_1_avg, \
#               c_alg_2_avg, d_alg_2_avg, e_alg_2_avg, nsw_alg_2_avg, \
#               c_alg_1_var, d_alg_1_var, e_alg_1_var, nsw_alg_1_var, \
#               c_alg_2_var, d_alg_2_var, e_alg_2_var, nsw_alg_2_var, num_agents)
#     # fraction = execute_2(c_alg_1_avg, d_alg_1_avg, e_alg_1_avg, nsw_alg_1_avg, \
#     #           c_alg_2_avg, d_alg_2_avg, e_alg_2_avg, nsw_alg_2_avg, \
#     #           c_alg_1_var, d_alg_1_var, e_alg_1_var, nsw_alg_1_var, \
#     #           c_alg_2_var, d_alg_2_var, e_alg_2_var, nsw_alg_2_var, num_agents)
#     # print(fraction)
#     avg_bp_len_nsw.append(avg_bp_nsw)
#     avg_bp_len_agents.append(avg_bp_agents)
#     min_bp_len_nsw.append(min_bp_nsw)
#     min_bp_len_agents.append(min_bp_agents)    
#     max_bp_len_nsw.append(max_bp_nsw)
#     max_bp_len_agents.append(max_bp_agents)
# plot_bps(agents, avg_bp_len_nsw, avg_bp_len_agents, max_bp_len_nsw, max_bp_len_agents, min_bp_len_nsw, min_bp_len_agents)
# ratio_c = [c_alg_2_avg[i] / c_alg_1_avg[i] for i in range(len(c_alg_1_avg))]
# ratio_d = [d_alg_2_avg[i] / d_alg_1_avg[i] for i in range(len(d_alg_1_avg))]
# ratio_e = [e_alg_2_avg[i] / e_alg_1_avg[i] for i in range(len(e_alg_1_avg))]
# ratio_nsw = [nsw_alg_2_avg[i] / nsw_alg_1_avg[i] for i in range(len(nsw_alg_1_avg))]

# print(np.mean(ratio_c), np.mean(ratio_d), np.mean(ratio_e), np.mean(ratio_nsw))
# plot_regret(agents, c_alg_1_avg, c_alg_2_avg, c_alg_1_var, c_alg_2_var)
# plot_egalitarian(agents, d_alg_1_avg, d_alg_2_avg, d_alg_1_var, d_alg_2_var)
# plot_disparity(agents, e_alg_1_avg, e_alg_2_avg, e_alg_1_var, e_alg_2_var)
# plot_nsw(agents, nsw_alg_1_avg, nsw_alg_2_avg, nsw_alg_1_var, nsw_alg_2_var)
# plot_nsw(nsw_alg_1_avg, nsw_alg_2_avg, nsw_alg_3_avg, \
#     nsw_alg_1_var, nsw_alg_2_var, nsw_alg_3_var, agents)
# plot_worst_ranks(agents, c_alg_1_avg, c_alg_2_avg, c_alg_3_avg, c_alg_1_var, c_alg_2_var, c_alg_3_var)
# plot_summation_ranks(agents, d_alg_1_avg, d_alg_2_avg, d_alg_3_avg, d_alg_1_var, d_alg_2_var, d_alg_3_var)
# plot_max_disparity(agents, e_alg_1_avg, e_alg_2_avg, e_alg_3_avg, e_alg_1_var, e_alg_2_var, e_alg_3_var)

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