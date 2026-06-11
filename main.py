import numpy as np
from dataset_generator import create_preflist, create_preference_list, uniform_instance_generator, triangular_instance_generator, normal_instance_generator
from find_matchings import routine
from data_processor_chatgpt import data_processor
from concurrent.futures import ProcessPoolExecutor
import time
import os
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
    output_folder_2 = "./matchings-uniform_popularity"
    output_folder_3 = "./matchings-triangular_popularity"
    output_folder_4 = "./matchings-normal_popularity"
    os.makedirs(output_folder_1, exist_ok=True)
    os.makedirs(output_folder_2, exist_ok=True)
    os.makedirs(output_folder_3, exist_ok=True)
    os.makedirs(output_folder_4, exist_ok=True)
    filename = f"matchings_n={num_agents}_iters={num_iters}.json"
    filepath_1 = os.path.join(output_folder_1, filename)
    filepath_2 = os.path.join(output_folder_2, filename)
    filepath_3 = os.path.join(output_folder_3, filename)
    filepath_4 = os.path.join(output_folder_4, filename)
    with open(filepath_1, 'w') as results_file:
        for iter in range(num_iters):
            if iter % 20000 == 0:
                print(iter)
            preflist = create_preflist(num_agents)
            routine(preflist, results_file)
    with open(filepath_2, 'w') as results_file:
        for iter in range(num_iters):
            if iter % 20000 == 0:
                print(iter)
            preflist = uniform_instance_generator(num_agents)
            routine(preflist, results_file)
    with open(filepath_3, 'w') as results_file:
        for iter in range(num_iters):
            if iter % 20000 == 0:
                print(iter)
            preflist = triangular_instance_generator(num_agents)
            routine(preflist, results_file)
    with open(filepath_4, 'w') as results_file:
        for iter in range(num_iters):
            if iter % 20000 == 0:
                print(iter)
            preflist = normal_instance_generator(num_agents)
            routine(preflist, results_file)
            
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
    execute(n, 100)

if __name__ == "__main__":
    # real_world_dataset()
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