import numpy as np
from dataset_generator import print_preflist, generate_man0_preferences, generate_worst_ratio_prefs, create_preflist, create_preference_list, uniform_instance_generator, triangular_instance_generator, normal_instance_generator, generate_instances,generate_instances_n_6, create_pattern
from graph_analysis import analyze_ratio
from find_matchings import routine, convert_to_builtin
from data_processor_chatgpt import data_processor
from itertools import permutations, product
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from matplotlib import pyplot as plt
import multiprocessing
import time
import copy
import os
from itertools import repeat
import ctypes
import json
import math
np.random.seed(69)

# alg_1 is for min regret optimal algorithm
# alg_2 is for max satisfaction / egalitarian algorithm
# alg_3 is for min disparity / sex-equal matching algorithm
# alg_4 is for max nsw / nash social welfare algorithm
# c is for min regret measure
# d is for summation of ranks measure
# e is for disparity measure
# nsw is for nash social welfare measure

def execute(num_agents, num_iters):
    output_folder_1 = "./matchings-uniform"
    # output_folder_2 = "./matchings-uniform_popularity"
    # output_folder_3 = "./matchings-triangular_popularity"
    # output_folder_4 = "./matchings-normal_popularity"
    os.makedirs(output_folder_1, exist_ok=True)
    # os.makedirs(output_folder_2, exist_ok=True)
    # os.makedirs(output_folder_3, exist_ok=True)
    # os.makedirs(output_folder_4, exist_ok=True)
    filename = f"matchings_n={num_agents}_iters={num_iters}.json"
    filepath_1 = os.path.join(output_folder_1, filename)
    # filepath_2 = os.path.join(output_folder_2, filename)
    # filepath_3 = os.path.join(output_folder_3, filename)
    # filepath_4 = os.path.join(output_folder_4, filename)
    ratio_min = float('inf')
    # with open(filepath_1, 'w') as results_file:
    for iter in range(num_iters):
        if iter % 20000 == 0:
            print(iter)
        preflist = create_preflist(num_agents)
        ratio_min = routine(preflist, filepath_1, ratio_min)
    # with open(filepath_2, 'w') as results_file:
    #     for iter in range(num_iters):
    #         if iter % 20000 == 0:
    #             print(iter)
    #         preflist = uniform_instance_generator(num_agents)
    #         routine(preflist, results_file)
    # with open(filepath_3, 'w') as results_file:
    #     for iter in range(num_iters):
    #         if iter % 20000 == 0:
    #             print(iter)
    #         preflist = triangular_instance_generator(num_agents)
    #         routine(preflist, results_file)
    # with open(filepath_4, 'w') as results_file:
    #     for iter in range(num_iters):
    #         if iter % 20000 == 0:
    #             print(iter)
    #         preflist = normal_instance_generator(num_agents)
    #         routine(preflist, results_file)

def regret_snsw_analysis(folderpath):
    for filename in os.listdir(folderpath):
        filepath = os.path.join(folderpath, filename)
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                mu_r_Mr = data["scores"]["reg"][0]
                mu_r_Msnsw = data["scores"]["reg"][3]
                if mu_r_Mr < mu_r_Msnsw:
                    print(line)
                    break

def egalitarian_n_4_snsw_analysis(folderpath):
    ratio_min = float('inf')
    min_filename = None
    for filename in os.listdir(folderpath):
        filepath = os.path.join(folderpath, filename)
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                ratio_curr = data["scores"]["ratio"]
                if ratio_curr < ratio_min:
                    ratio_min = ratio_curr
                    min_filename = filename
    print(min_filename)

def worker(args):
    worker_id, num_workers, num_agents = args
    foldername = "latest_worst_ratio_n=6/"
    os.makedirs(foldername, exist_ok=True)
    filename = f"matchings_{num_agents}_worst_ratio_{worker_id}.json"
    filepath = os.path.join(foldername, filename)
    ratio_min = float('inf')
    for idx, instance in enumerate(generate_instances_n_6()):
        if idx % num_workers == worker_id:
            ratio_min = routine(instance, filepath, ratio_min)
    

def real_world_dataset():
    output_folder = "./real_world_exps"
    os.makedirs(output_folder, exist_ok=True)
    file_path = "./dataset/2024_10_08_18_00_09_PayoffMatrices.txt"
    markets = data_processor(file_path)
    for market in markets:
        name = market["treatment_name"]
        filename = f"{name}.json"
        filepath = os.path.join(output_folder, filename)
        men_utilities = market["food_weights"]
        women_utilities = market["color_weights"]
        weight_matrix = np.array(men_utilities) + np.array(women_utilities)
        preflist = create_preference_list(weight_matrix)
        with open(filepath, 'w') as results_file:
            routine(preflist, results_file)

def run(n):
    print(f"Number of Agents = {n}") 
    execute(n, 1000000)

def process_modification_1(args):
    preflist, modification = args

    _preflist = copy.deepcopy(preflist)
    _preflist[0][0] = list(modification[0])
    _preflist[0][1] = list(modification[1])

    # Modify routine so that it returns
    # (result_dict, ratio)
    result, ratio = routine(_preflist)
    result["pref_man_0"] = convert_to_builtin(modification[0])
    result["pref_man_1"] = convert_to_builtin(modification[1])
    return result, ratio

def process_modification_2(args):
    preflist, modification = args

    _preflist = copy.deepcopy(preflist)
    _preflist[0][0] = list(modification[0])
    _preflist[1][0] = list(modification[1])

    # Modify routine so that it returns
    # (result_dict, ratio)
    result, ratio = routine(_preflist)
    
    result["pref_man_0"] = convert_to_builtin(modification[0])
    result["pref_woman_0"] = convert_to_builtin(modification[1])
    return result, ratio

if __name__ == "__main__":
    for n in range(4, 8):
        k = max(0, n - 1 - 2*math.floor(math.sqrt(n)))
        preflist = generate_worst_ratio_prefs(n, k)

        modifications = product(permutations(range(n)), repeat=2)

        folderpath = "./AAAI_2027_exps/"
        os.makedirs(folderpath, exist_ok=True)

        filename_1 = f"man_0_man_1_all_modifications_n={n}.json"
        filename_2 = f"man_0_woman_0_all_modifications_n={n}.json"
        filepath_1 = os.path.join(folderpath, filename_1)
        filepath_2 = os.path.join(folderpath, filename_2)


        base_preflist = {
            "preflist_common": convert_to_builtin(preflist)
        }

        min_ratio_so_far = float("inf")

        with open(filepath_1, "w") as results_file:
            results_file.write(json.dumps(base_preflist) + "\n")

            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

                args = ((preflist, modification) for modification in modifications)

                for i, (result, ratio) in enumerate(executor.map(process_modification_1, args, chunksize=100)):
                    if i % 10000 == 0:
                        print(i)

                    min_ratio_so_far = min(min_ratio_so_far, ratio)

                    results_file.write(json.dumps(result) + "\n")

        print("Minimum ratio:", min_ratio_so_far)

        with open(filepath_2, "w") as results_file:
                results_file.write(json.dumps(base_preflist) + "\n")
        
                with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        
                    modifications = product(permutations(range(n)), repeat=2)
                    args = ((preflist, modification) for modification in modifications)
        
                    for i, (result, ratio) in enumerate(executor.map(process_modification_2, args, chunksize=100)):
                        if i % 10000 == 0:
                            print(i)
        
                        min_ratio_so_far = min(min_ratio_so_far, ratio)
        
                        results_file.write(json.dumps(result) + "\n")
        
        print("Minimum ratio:", min_ratio_so_far)
    # for n in range(6, 8):
    #     print(n)
        # preflist = create_pattern(n)
    #     print_preflist(preflist)
        # filename = f"pattern_n={n}.json"
        # folderpath = "./latest_worst_ratio/"
        # filepath = os.path.join(folderpath, filename)
    #     ratio = routine(preflist, filepath, float('inf'))
    #     print(ratio)
    # for n in range(4, 11):
    # men_preflist = []
    # for t in range(n):
    #     men_preflist.append(list(range(t, n)) + list(range(t)))
    #     # men_preflist.append(list(range(n)))
    # list_of_women_prefs = generate_instances(n)
    # ratio_list = []
    # ratio_min = float('inf')
    # preflist_min = None
    # folderpath = "./exps/"
    # os.makedirs(folderpath, exist_ok=True)
    # for i, women_preflist in enumerate(list_of_women_prefs):
    #     if i % 10000 == 0:
    #         print(i)
    #     preflist = [men_preflist, women_preflist]
    #     filename = f"coop_men_all_women_n={n}.json"
    #     filepath = os.path.join(folderpath, filename)
    #     ratio_curr = routine(preflist, filepath, ratio_min)
    #     ratio_list.append(ratio_curr)
    #     if ratio_curr < ratio_min:
    #         ratio_min = ratio_curr
    #         preflist_min = preflist
    # print("Minimum Ratio = ", ratio_min)
    # print("Minimum Ratio Preference List")
    # print_preflist(preflist_min)
    # analyze_ratio(ratio_list)
    # for ratio, indices in dict.items():
    #     print("=============================================================================================================")
    #     print(ratio)
    #     print("=============================================================================================================")
        # for j, i in enumerate(indices):
        #     if j > 1:
        #         break
            # print("mue(Me) = ", mueMe_list[i])
            # print("mue(Msnsw)", mueMsnsw_list[i])
            # print_ranklist(list_of_women_prefs[i])
            # break
    # preflist = [[[0, 1, 2, 3],
    #              [0, 1, 2, 3],
    #              [0, 1, 2, 3],
    #              [0, 1, 2, 3]],
    #             [[1, 2, 3, 0],
    #              [2, 3, 0, 1],
    #              [2, 1, 0, 3],
    #              [3, 2, 1, 0]]]

    # preflist = [[[0, 1, 3, 2, 4, 5, 6, 7],
    #              [0, 1, 3, 2, 4, 5, 6, 7],
    #              [2, 3, 1, 0, 7, 6, 5, 4],
    #              [2, 3, 1, 0, 7, 6, 5, 4],
    #              [4, 5, 6, 7, 3, 2, 1, 0],
    #              [4, 5, 6, 7, 3, 2, 1, 0],
    #              [6, 7, 5, 4, 1, 0, 3, 2],
    #              [6, 7, 5, 4, 1, 0, 3, 2]],
    #             [[]]]
    # preflist = [[[[0, 1, 2, 3, 4, 5, 6, 7],
    #              [0, 1, 2, 3, 4, 5, 6, 7],
    #              [3, 2, 1, 0, 7, 6, 5, 4],
    #              [3, 2, 1, 0, 7, 6, 5, 4],
    #              [4, 5, 6, 7, 2, 3, 0, 1],
    #              [4, 5, 6, 7, 2, 3, 0, 1],
    #              [6, 7, 4, 5, 1, 0, 3, 2],
    #              [6, 7, 4, 5, 1, 0, 3, 2],],
    #             [[5, 4, 6, 7, 2, 3, 1, 0],
    #              [4, 5, 7, 6, 3, 2, 0, 1],
    #              [6, 7, 5, 4, 1, 0, 2, 3],
    #              [7, 6, 4, 5, 0, 1, 3, 2],
    #              [4, 5, 6, 7, 0, 2, 3, 1],
    #              [5, 4, 7, 6, 2, 0, 1, 3],
    #              [6, 7, 4, 5, 3, 1, 0, 2],
    #              [7, 6, 5, 4, 3, 2, 1, 0]]]]
    # 
    #                  [10, 3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 11, 12, 13]]
    # for r in range(1, (n // 2) + 1):
    # r = 3
    # modifications = []
    # modifications = generate_man0_preferences(n, 3, r)
    # modifications.append(list(range(n)))
    # modifications = [[5, 4, 0, 1, 2, 3], [4, 5, 0, 1, 2, 3]]
    # modifications.append([n - 1] + list(range(n - 1)))
    # for x in range(n):
    #     modifications.append([x] + list(range(0, x)) + list(range(x + 1, n)))
    # modifications = list(permutations(range(n)))
    # modifications.append(range(n))
    # identical_agents = list(range(0, (n // 2)))
    # identical_agents.append(n - 1)
    # cyclic_agents = list(range((n // 2), n - 1))


    # list_of_prefs = change_first_agent_pref(preflist, identical_agents, cyclic_agents)
    # preflist = [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 7], 
    #              [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 7, 8], 
    #              [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 7, 8, 9], 
    #              [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 7, 8, 9, 10], 
    #              [0, 1, 2, 3, 4, 5, 6, 12, 13, 7, 8, 9, 10, 11], 
    #              [0, 1, 2, 3, 4, 5, 6, 13, 7, 8, 9, 10, 11, 12]], 
    #             [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
    #              [8, 0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 7], 
    #              [9, 0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 7, 8], 
    #              [10, 0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 7, 8, 9], 
    #              [11, 0, 1, 2, 3, 4, 5, 6, 12, 13, 7, 8, 9, 10], 
    #              [12, 0, 1, 2, 3, 4, 5, 6, 13, 7, 8, 9, 10, 11], 
    #              [13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
    #              [7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]]]
    # print_preflist(preflist)
    # print(preflist)
    # dict = {}
    # for i in range(len(cyclic_agents)):
    #     dict[i] = set()
    # mueMe_list = []
    # mueMsnsw_list = []
    #     # print_preflist(_preflist)
    #     # print(ratio)
        # ratio_list.append(ratio)
    #     mueMe_list.append(mueMe)
    #     mueMsnsw_list.append(mueMsnsw)
    # dict = analyze_ratio(ratio_list)
    # for ratio, indices in dict.items():
    #     print("========================================")
    #     print(ratio)
    #     print("========================================")
    #     for j, i in enumerate(indices):
    #         if j > 100:
    #             break
    #         # print("mue(Me) = ", mueMe_list[i])
    #         # print("mue(Msnsw)", mueMsnsw_list[i])
    #         print(modifications[i])
    #         # break

    #     # print(i)
    #     # if i < 2000:
    #     #     continue
    #     # if i > 2000:
    #     #     break
    #     print(modifications[i - 1], modifications[i])
        
        # x_list = range(len(ratio_list))
        # plt.plot(x_list, ratio_list)
        # plt.show()
        # print(ratio)
    #     pref = preflist[0][0]
    #     idx = pref.index(n - 2)
    #     dict[idx].add(ratio)
    # print(dict)
        # print(ratio)
        # print_preflist(preflist)
    # folderpath = 'latest_worst_ratio_25June2026'
    # egalitarian_n_4_snsw_analysis(folderpath)
    # regret_snsw_analysis(folderpath)
    # real_world_dataset()
    # start_time = time.time()
    # num_agents = 6
    # num_workers = os.cpu_count()
    # with Pool(num_workers) as pool:
    #     pool.map(
    #         worker,
    #         [(i, num_workers, num_agents) for i in range(num_workers)]
    #     )
    # with ProcessPoolExecutor() as executor:
    #     results = list(executor.map(run, range(4, 5)))
    # end_time = time.time()
    # hrs = (end_time - start_time)//3600
    # mins = ((end_time - start_time)%3600)//60
    # secs = (end_time - start_time) % 60
    # prina if rt(f"Time taken = {hrs} hours {mins} minutes {secs} seconds")


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