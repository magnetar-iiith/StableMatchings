import numpy as np
import json
import copy
from gale_shapley_alg import gale_shapley
from shortlists import create_shortlists
from rotations import find_a_rotation, eliminate_rotation
from graph_construction import closed_subset_finder, create_rotation_digraph, predecessors, topological_sort, stable_matching
from measures import regret, egalitarian, disparity, nash_welfare, egalitarian_welfare

def convert_to_builtin(obj):
    """Converts objects into lists"""
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
    """Finds min-regret, egalitarian, sex-equal,
     and snsw stable matchings"""   
    male_optimal_matching = gale_shapley(preflist)
    # copy_preflist = copy.deepcopy(preflist)
    # men_shortlists, women_shortlists = create_shortlists(copy_preflist, male_optimal_matching)
    men_shortlists, women_shortlists = create_shortlists(preflist, male_optimal_matching)
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

    graph = create_rotation_digraph(rotations, men_shortlists, women_shortlists)

    pred = predecessors(graph)
    topo_order = topological_sort(graph, pred)
    closed_subsets = closed_subset_finder(topo_order, pred)
    # min_regret  = float('inf')
    min_egalitarian  = float('inf')
    # min_disparity  = float('inf')
    max_nash_welfare  = float('-inf')
    min_egalitarian_matching, max_nash_welfare_matching = None, None
    # min_regret_matching, min_egalitarian_matching, \
    # min_disparity_matching, max_nash_welfare_matching\
    #  = None, None, None, None
    for subset in closed_subsets:
        copy_men_shortlists = copy.deepcopy(men_shortlists)
        copy_women_shortlists = copy.deepcopy(women_shortlists)
        matching_1 = stable_matching(subset, rotations, topo_order, copy_men_shortlists, copy_women_shortlists)
        
        # regret_val = regret(matching_1, preflist)
        egalitarian_val = egalitarian(matching_1, preflist)
        # disparity_val = disparity(matching_1, preflist)
        nash_welfare_val = nash_welfare(matching_1, preflist)
        # if regret_val < min_regret:
        #     min_regret = regret_val
        #     min_regret_matching = matching_1
        if egalitarian_val < min_egalitarian:
            min_egalitarian = egalitarian_val
            min_egalitarian_matching = matching_1
        # if disparity_val < min_disparity:
        #     min_disparity = disparity_val
        #     min_disparity_matching = matching_1
        if nash_welfare_val > max_nash_welfare:
            max_nash_welfare = nash_welfare_val
            max_nash_welfare_matching = matching_1
    # regret_1 = regret(min_regret_matching, preflist)
    # egalitarian_1 = egalitarian(min_regret_matching, preflist)
    # disparity_1 = disparity(min_regret_matching, preflist)
    # nash_welfare_1 = nash_welfare(min_regret_matching, preflist)

    # regret_2 = regret(min_egalitarian_matching, preflist)
    egalitarian_2 = egalitarian_welfare(min_egalitarian_matching, preflist)
    # disparity_2 = disparity(min_egalitarian_matching, preflist)
    # nash_welfare_2 = nash_welfare(min_egalitarian_matching, preflist)
    
    # regret_3 = regret(min_disparity_matching, preflist)
    # egalitarian_3 = egalitarian(min_disparity_matching, preflist)
    # disparity_3 = disparity(min_disparity_matching, preflist)
    # nash_welfare_3 = nash_welfare(min_disparity_matching, preflist)

    # regret_4 = regret(max_nash_welfare_matching, preflist)
    egalitarian_4 = egalitarian_welfare(max_nash_welfare_matching, preflist)
    # disparity_4 = disparity(max_nash_welfare_matching, preflist)
    # nash_welfare_4 = nash_welfare(max_nash_welfare_matching, preflist)
    data = {
        "preflist": convert_to_builtin(preflist),
        # "min_regret": min_regret_matching,
        "egalitarian": min_egalitarian_matching,
        # "sex_equal": min_disparity_matching,
        "nsw": max_nash_welfare_matching,
        "scores": {
            # "reg": [float(regret_1), float(regret_2), float(regret_3), float(regret_4)],
            # "eg": [float(egalitarian_1), float(egalitarian_2), float(egalitarian_3), float(egalitarian_4)],
            # "disp": [float(disparity_1), float(disparity_2), float(disparity_3), float(disparity_4)],
            # "nsw": [float(nash_welfare_1), float(nash_welfare_2), float(nash_welfare_3), float(nash_welfare_4)]
            "mueemuensw" : [float(egalitarian_2), float(egalitarian_4)]
        }
    }
    # return convert_to_builtin(data), egalitarian_4 / egalitarian_2
    results_file.write(json.dumps(convert_to_builtin(data)) + "\n")