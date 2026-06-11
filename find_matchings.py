import numpy as np
import json
import copy
from gale_shapley_alg import gale_shapley
from shortlists import create_shortlists
from rotations import find_a_rotation, eliminate_rotation
from graph_construction import closed_subset_finder, create_rotation_digraph, predecessors, topological_sort, stable_matching
from measures import reg, eg, disp, nsw

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
    
def routine(preflist, results_file):    
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