import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
import numpy as np
import random
import json
import os

random.seed(42)

def load_and_average_scores(file_path):
    c_scores = [[] for _ in range(4)]
    d_scores = [[] for _ in range(4)]
    e_scores = [[] for _ in range(4)]
    nsw_scores = [[] for _ in range(4)]

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            for i in range(4):
                c_scores[i].append(data["scores"]["reg"][i])
                d_scores[i].append(data["scores"]["eg"][i])
                e_scores[i].append(data["scores"]["disp"][i])
                nsw_scores[i].append(data["scores"]["nsw"][i])

    c_avgs = [np.mean(c_scores[i]) for i in range(4)]
    d_avgs = [np.mean(d_scores[i]) for i in range(4)]
    e_avgs = [np.mean(e_scores[i]) for i in range(4)]
    nsw_avgs = [np.mean(nsw_scores[i]) for i in range(4)]
    # c_covs  = [np.std(c_scores[i])/np.mean(c_scores[i]) for i in range(4)]
    # d_covs = [np.std(d_scores[i])/np.mean(d_scores[i]) for i in range(4)]
    # e_covs = [np.std(e_scores[i])/np.mean(e_scores[i]) for i in range(4)]
    # nsw_covs = [np.std(nsw_scores[i])/np.mean(nsw_scores[i]) for i in range(4)]
    # return c_avgs + d_avgs + e_avgs + nsw_avgs + c_covs + d_covs + e_covs + nsw_covs  # length 32
    return c_avgs + d_avgs + e_avgs + nsw_avgs

def normalize(alg, maxes, mins):
    regret = [(alg[0] - mins[0]) / (maxes[0] - mins[0])]

    egalitarian = [(alg[1] - mins[1]) / (maxes[1] - mins[1])]

    sex_equality = [(alg[2] - mins[2]) / (maxes[2] - mins[2])]

    nsw = [(maxes[3] - alg[3]) / (maxes[3] - mins[3])]

    return regret + egalitarian + sex_equality + nsw

def plot_areas(agents, area1_list, area2_list, area3_list, area4_list):
    plt.figure(figsize=(10, 6))
    plt.plot(agents, area2_list, marker='o', color='blue', label=r'$\mathcal{B}_1$')
    plt.plot(agents, area1_list, marker='s', color='red', label=r'$\mathcal{B}_2$')
    plt.plot(agents, area3_list, marker='x', color='purple', label=r'$\mathcal{B}_3$')
    plt.plot(agents, area4_list, marker='*', color='green', label='SNSW-Alg')

    plt.xlabel('Number of Agents', fontsize=30)
    plt.ylabel('Area', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(r'Area of Baseline triangles', fontsize=35)
    plt.legend(fontsize=25)
    plt.grid(True)
    plt.show()

def process_all_files(folder_path):
    # ratio_1, ratio_2, ratio_3 = float('inf'), float('inf'), float('inf')
    # area4_list, agents, area3_list, area2_list, area1_list = [], [], [], [], []
    for filename in sorted(os.listdir(folder_path)):
        for n in range(5, 51):
            if filename.startswith(f"matchings_n={n}") and filename.endswith(".json"):
                # if "_\\mathcal{U}" not in filename:
                #     continue  # Skip anything that doesn’t exactly contain "_\mathcal{U"

                # # Skip files that contain undesired prefixes like "\mathrm{P}"
                # if "\\mathrm" in filename:
                #     continue
                # if "_\\mathrm{P}_{\\mathcal{N}}" not in filename:
                #     continue
                num_agents = n
                # agents.append(num_agents)
                file_path = os.path.join(folder_path, filename)
                scores = load_and_average_scores(file_path)
                (c1, c2, c3, c4, d1, d2, d3, d4, e1, e2, e3, e4, nsw1, nsw2, nsw3, nsw4) = scores
                plot_pairs(num_agents, c1, c2, c3, c4, d1, d2, d3, d4, e1, e2, e3, e4, nsw1, nsw2, nsw3, nsw4)
                plot_circle(num_agents, c1, c2, c3, c4, d1, d2, d3, d4, e1, e2, e3, e4, nsw1, nsw2, nsw3, nsw4)

        #         areas = [area1, area2, area3, area4]
        #         area4_list.append(area4)
        #         area3_list.append(area3)
        #         area2_list.append(area2)
        #         area1_list.append(area1)
        #         # ratio_1 = min(ratio_1, area1/area4)
        #         # ratio_2 = min(ratio_2, area2/area4)
        #         # ratio_3 = min(ratio_3, area3/area4)
        # # print(f"Max ratio for area1: {ratio_1}, area2: {ratio_2}, area3: {ratio_3}")
        # sorted_pairs = sorted(zip(agents, area4_list, area3_list, area2_list, area1_list))
        # agents_sorted, area4_sorted, area3_sorted, area2_sorted, area1_sorted = zip(*sorted_pairs)
        # agents_sorted = list(agents_sorted)
        # area4_sorted = list(area4_sorted)
        # area3_sorted = list(area3_sorted)
        # area2_sorted = list(area2_sorted)
        # area1_sorted = list(area1_sorted)

        # plot_areas(agents_sorted, area1_sorted, area2_sorted, area3_sorted, area4_sorted)
                # # Unpack scores to individual variables if needed
                # (c1, c2, c3, c4, d1, d2, d3, d4, e1, e2, e3, e4, nsw1, nsw2, nsw3, nsw4, \
                #  C1, C2, C3, C4, D1, D2, D3, D4, E1, E2, E3, E4, NSW1, NSW2, NSW3, NSW4) = scores
                
                # if num_agents == 50:
                    # # print all values
                    # print(f"{c1}({C1}), {c2}({C2}), {c3}({C3}), {c4}({C4})")
                    # print(f"{d1}({D1}), {d2}({D2}), {d3}({D3}), {d4}({D4})")
                    # print(f"{e1}({E1}), {e2}({E2}), {e3}({E3}), {e4}({E4})")
                    # print(f"{nsw1}({NSW1}), {nsw2}({NSW2}), {nsw3}({NSW3}), {nsw4}({NSW4})")
                    # # Replace this with your plotting function call
                    # print(f"Calling plot for {num_agents} agents:")
                    # plot_pairs(num_agents, c1, c2, c3, c4, d1, d2, d3, d4, e1, e2, e3, e4, nsw1, nsw2, nsw3, nsw4)


# def plot_worst_ranks(agents, c_1_avg, c_2_avg, c_3_avg, c_1_var, c_2_var, c_3_var):
#     plt.figure(figsize=(10, 6))
#     plt.plot(agents, c_1_avg, label='Summation of Ranks Criterion', marker='o', color='blue')
#     plt.plot(agents, c_2_avg, label='Nash Social Welfare', marker='x', color='red')
#     plt.plot(agents, c_3_avg, label='Disparity Criterion', marker='s', color='green')
#     plt.fill_between(agents, np.array(c_1_avg) - np.array(c_1_var), np.array(c_1_avg) + np.array(c_1_var), alpha=0.2)
#     plt.fill_between(agents, np.array(c_2_avg) - np.array(c_2_var), np.array(c_2_avg) + np.array(c_2_var), alpha=0.2)
#     plt.fill_between(agents, np.array(c_3_avg) - np.array(c_3_var), np.array(c_3_avg) + np.array(c_3_var), alpha=0.2)

#     plt.xlabel('Number of Agents', fontsize = 20)
#     plt.ylabel('Average Worst Rank', fontsize = 20)
#     # plt.title('Comparison of Matching Criteria', fontsize = 30)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.legend(fontsize=20)
#     plt.grid(True)
#     plt.show()

# def plot_summation_ranks(agents, d_1_avg, d_2_avg, d_3_avg, d_1_var, d_2_var, d_3_var):
#     plt.figure(figsize=(10, 6))
#     plt.plot(agents, d_1_avg, label='Summation of Ranks Criterion', marker='o', color='blue')
#     plt.plot(agents, d_2_avg, label='Nash Social Welfare', marker='x', color='red')
#     plt.plot(agents, d_3_avg, label='Disparity Criterion', marker='s', color='green')
#     plt.fill_between(agents, np.array(d_1_avg) - np.array(d_1_var), np.array(d_1_avg) + np.array(d_1_var), alpha=0.2)
#     plt.fill_between(agents, np.array(d_2_avg) - np.array(d_2_var), np.array(d_2_avg) + np.array(d_2_var), alpha=0.2)
#     plt.fill_between(agents, np.array(d_3_avg) - np.array(d_3_var), np.array(d_3_avg) + np.array(d_3_var), alpha=0.2)

#     plt.xlabel('Number of Agents', fontsize = 20)
#     plt.ylabel('Average Summation of Ranks', fontsize = 20)
#     # plt.title('Comparison of Matching Criteria', fontsize = 30)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.legend(fontsize=20)
#     plt.grid(True)
#     plt.show()

# def plot_max_disparity(agents, e_1_avg, e_2_avg, e_3_avg, e_1_var, e_2_var, e_3_var):
#     plt.figure(figsize=(10, 6))
#     plt.plot(agents, e_1_avg, label='Summation of Ranks Criterion', marker='o', color='blue')
#     plt.plot(agents, e_2_avg, label='Nash Social Welfare', marker='x', color='red')
#     plt.plot(agents, e_3_avg, label='Disparity Criterion', marker='s', color='green')
#     plt.fill_between(agents, np.array(e_1_avg) - np.array(e_1_var), np.array(e_1_avg) + np.array(e_1_var), alpha=0.2)
#     plt.fill_between(agents, np.array(e_2_avg) - np.array(e_2_var), np.array(e_2_avg) + np.array(e_2_var), alpha=0.2)
#     plt.fill_between(agents, np.array(e_3_avg) - np.array(e_3_var), np.array(e_3_avg) + np.array(e_3_var), alpha=0.2)
    
#     plt.xlabel('Number of Agents', fontsize = 20)
#     plt.ylabel('Average Maximum Disparity', fontsize = 20)
#     # plt.title('', fontsize = 30)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.legend(fontsize=20)
#     plt.grid(True)
#     plt.show()

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

# Helper function to draw directional arrow labeled "better"
def draw_arrow_nsw(ax, direction='horizontal'):
    if direction == 'horizontal':
        # Horizontal arrow: same y-coordinate for start and end
        ax.annotate('',xy=(0.95, 0), xytext=(0.05, 0),  # Same y-coordinate (0.1)
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    ha='center', va='bottom', fontsize=20)
    elif direction == 'vertical':
        # Vertical arrow: same x-coordinate for start and end
        ax.annotate('',xy=(-0.2, 0.95), xytext=(-0.2, 0.65),  # Same x-coordinate (0.1)
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    ha='left', va='center', fontsize=20, rotation=90)

# Helper function to draw directional arrow labeled "better"
def draw_arrow_rest(ax, direction='horizontal'):
    x_coord_left, x_coord_right = 0.45, 0.35
    y_coord_left, y_coord_right = -0.3, -0.6
    if direction == 'horizontal':
        # Horizontal arrow: same y-coordinate for start and end
        ax.annotate('',xy=(0.025, -0.185), xytext=(0.4, -0.185),  # Same y-coordinate (0.1)
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    ha='center', va='bottom', fontsize=20)
    elif direction == 'vertical':
        # Vertical arrow: same x-coordinate for start and end
        ax.annotate('', xy=(-0.15, 0.05), xytext=(-0.15, 0.4), # Same x-coordinate (0.1)
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    ha='left', va='center', fontsize=20, rotation=90)

def mark_pareto_optimality_1(ax, x, y):
    # mark the points pareto dominated by nsw
    # pareto dominated means they are both right and down to nsw
    x_4, y_4 = x[3], y[3]
    for i in range(3):
        if x[i] >= x_4 and y[i] >= y_4:
            circ = Ellipse((x[i], y[i]), height=0.0375, width=0.028125, edgecolor='black',
                  facecolor='none', linewidth=2)
            # ax.text(x[i], y[i], 'Pareto Dominated by NSW', fontsize=10, color='black')
            ax.add_patch(circ)
    # n = 25
    # default uniform height=0.05, width=0.15
    # popularity uniform height=0.0375, width=0.025
    # popularity triangular height=0.0375, width=0.05625
    # popularity normal height=0.0375, width=0.028125
    # n = 50
    # default uniform height=0.05, width=0.25
    # popularity uniform height=0.045, width=0.025
    # popularity triangular height=0.045, width=0.1
    # popularity normal height=0.045, width=0.028125
def mark_pareto_optimality_2(ax, x, y):
    # mark the points pareto dominated by nsw
    # pareto dominated means they are both right and down to nsw
    x_4, y_4 = x[3], y[3]
    for i in range(3):
        if x[i] >= x_4 and y[i] >= y_4:
            circ = Ellipse((x[i], y[i]), height=7.5, width=0.028125, edgecolor='black',
                  facecolor='none', linewidth=2)
            # ax.text(x[i], y[i], 'Pareto Dominated by NSW', fontsize=10, color='black')
            ax.add_patch(circ)

    # n = 25
    # default uniform height=10, width=0.15
    # popularity uniform height=7.5, width=0.025
    # popularity triangular height=7.5, width=0.05625
    # popularity normal height=7.5, width=0.028125
    # n = 50
    # default uniform height=15, width=0.25
    # popularity uniform height=12.5, width=0.025
    # popularity triangular height=12.5, width=0.1
    # popularity normal height=12.5, width=0.028125
def mark_pareto_optimality_3(ax, x, y):
    # mark the points pareto dominated by nsw
    # pareto dominated means they are both right and down to nsw
    x_4, y_4 = x[3], y[3]
    for i in range(3):
        if x[i] >= x_4 and y[i] <= y_4:
            circ = Ellipse((x[i], y[i]), height=0.05, width=0.0375, edgecolor='black',
                  facecolor='none', linewidth=2)
            # ax.text(x[i], y[i], 'Pareto Dominated by NSW', fontsize=10, color='black')
            ax.add_patch(circ)

    # n = 25
    # default uniform height=0.05, width=0.15
    # popularity uniform height=0.05, width=0.025
    # popularity triangular height=0.05, width=0.075
    # popularity normal height=0.05, width=0.0375
    # n = 50
    # default uniform height=0.05, width=0.25
    # popularity uniform height=0.05, width=0.025
    # popularity triangular height=0.05, width=0.075
    # popularity normal height=0.05, width=0.0225
def mark_pareto_optimality_4(ax, x, y):
    # mark the points pareto dominated by nsw
    # pareto dominated means they are both right and down to nsw
    x_4, y_4 = x[3], y[3]
    for i in range(3):
        if x[i] >= x_4 and y[i] >= y_4:
            circ = Ellipse((x[i], y[i]), height=7.5, width=0.025, edgecolor='black',
                  facecolor='none', linewidth=2)
            # ax.text(x[i], y[i], 'Pareto Dominated by NSW', fontsize=10, color='black')
            ax.add_patch(circ)

    # n = 25
    # default uniform height=10, width=0.025
    # popularity uniform height=7.5, width=0.025
    # popularity triangular height=7.5, width=0.025
    # popularity normal height=7.5, width=0.025
    # n = 50
    # default uniform height=16.5, width=0.025
    # popularity uniform height=10, width=0.025
    # popularity triangular height=10, width=0.025
    # popularity normal height=10, width=0.025
def mark_pareto_optimality_5(ax, x, y):
    # mark the points pareto dominated by nsw
    # pareto dominated means they are both right and down to nsw
    x_4, y_4 = x[3], y[3]
    for i in range(3):
        if x[i] >= x_4 and y[i] <= y_4:
            circ = Ellipse((x[i], y[i]), height=0.05, width=0.025, edgecolor='black',
                  facecolor='none', linewidth=2)
            # ax.text(x[i], y[i], 'Pareto Dominated by NSW', fontsize=10, color='black')
            ax.add_patch(circ)

    # n = 25
    # default uniform height=0.05, width=0.025
    # popularity uniform height=0.05, width=0.025
    # popularity triangular height=0.05, width=0.025
    # popularity normal height=0.05, width=0.025
    # n = 50
    # default uniform height=0.05, width=0.025
    # popularity uniform height=0.05, width=0.025
    # popularity triangular height=0.05, width=0.025
    # popularity normal height=0.05, width=0.025

def mark_pareto_optimality_6(ax, x, y):
    # mark the points pareto dominated by nsw
    # pareto dominated means they are both right and down to nsw
    x_4, y_4 = x[3], y[3]
    for i in range(3):
        if x[i] >= x_4 and y[i] <= y_4:
            circ = Ellipse((x[i], y[i]), height=0.03, width=3, edgecolor='black',
                  facecolor='none', linewidth=2)
            # ax.text(x[i], y[i], 'Pareto Dominated by NSW', fontsize=10, color='black')
            ax.add_patch(circ)
    
    # n = 25
    # default uniform height=0.05, width=5
    # popularity uniform height=0.03, width=3
    # popularity triangular height=0.03, width=3
    # popularity normal height=0.03, width=3
    # n = 50
    # default uniform height=0.045, width=8.5
    # popularity uniform height=0.05, width=7.5
    # popularity triangular height=0.05, width=7.5
    # popularity normal height=0.05, width=7.5

def plot_pairs(n, c1, c2, c3, c4, d1, d2, d3, d4, e1, e2, e3, e4, nsw1, nsw2, nsw3, nsw4):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    x_1 = [c2, c1, c3, c4]
    y_1 = [d2, d1, d3, d4]
    x_2 = [c2, c1, c3, c4]
    y_2 = [e2, e1, e3, e4]
    x_3 = [c2, c1, c3, c4]
    y_3 = [nsw2, nsw1, nsw3, nsw4]
    x_4 = [d2, d1, d3, d4]
    y_4 = [e2, e1, e3, e4]
    x_5 = [d2, d1, d3, d4]
    y_5 = [nsw2, nsw1, nsw3, nsw4]
    x_6 = [e2, e1, e3, e4]
    y_6 = [nsw2, nsw1, nsw3, nsw4]
    colors = ['blue', 'red', 'purple', 'green']
    markers = ['o', 'x', 's', '*']
    labels = [r'$\mathcal{B}_1$', r'$\mathcal{B}_2$', r'$\mathcal{B}_3$', r'SNSW-Alg']

    # Set the limits for each axis
    max00x, max00y = 0, 0
    max01x, max01y = 0, 0
    max02x, max02y = 0, 0
    max10x, max10y = 0, 0
    max11x, max11y = 0, 0
    max12x, max12y = 0, 0
    min00x, min00y = float('inf'), float('inf')
    min01x, min01y = float('inf'), float('inf')
    min02x, min02y = float('inf'), float('inf')
    min10x, min10y = float('inf'), float('inf')
    min11x, min11y = float('inf'), float('inf')
    min12x, min12y = float('inf'), float('inf')
    for i in range(len(labels)):
        max00x = max(max00x, x_1[i])
        max01x = max(max01x, x_2[i])
        max02x = max(max02x, x_3[i])
        max10x = max(max10x, x_4[i])
        max11x = max(max11x, x_5[i])
        max12x = max(max12x, x_6[i])
        max00y = max(max00y, y_1[i])
        max01y = max(max01y, y_2[i])
        max02y = max(max02y, y_3[i])
        max10y = max(max10y, y_4[i])
        max11y = max(max11y, y_5[i])
        max12y = max(max12y, y_6[i])
        min00x = min(min00x, x_1[i])
        min01x = min(min01x, x_2[i])
        min02x = min(min02x, x_3[i])
        min10x = min(min10x, x_4[i])
        min11x = min(min11x, x_5[i])
        min12x = min(min12x, x_6[i])
        min00y = min(min00y, y_1[i])
        min01y = min(min01y, y_2[i])
        min02y = min(min02y, y_3[i])
        min10y = min(min10y, y_4[i])
        min11y = min(min11y, y_5[i])
        min12y = min(min12y, y_6[i])
    egalitarian_padding = 0.1
    regret_padding = 0.1
    sex_equality_padding = 20
    nsw_padding = 0.1
    axes[0,0].set_xlim(min00x-regret_padding, max00x+regret_padding)
    axes[0,0].set_ylim(min00y-egalitarian_padding, max00y+egalitarian_padding)
    axes[0,1].set_xlim(min01x-regret_padding, max01x+regret_padding)
    axes[0,1].set_ylim(min01y-sex_equality_padding, max01y+sex_equality_padding)
    axes[0,2].set_xlim(min02x-regret_padding, max02x+regret_padding)
    axes[0,2].set_ylim(min02y-nsw_padding, max02y+nsw_padding)
    axes[1,0].set_xlim(min10x-egalitarian_padding, max10x+egalitarian_padding)
    axes[1,0].set_ylim(min10y-sex_equality_padding, max10y+sex_equality_padding)
    axes[1,1].set_xlim(min11x-egalitarian_padding, max11x+egalitarian_padding)
    axes[1,1].set_ylim(min11y-nsw_padding, max11y+nsw_padding)
    axes[1,2].set_xlim(min12x-sex_equality_padding, max12x+sex_equality_padding)
    # axes[1,2].set_aspect('auto')
    axes[1,2].set_ylim(bottom=min12y-nsw_padding, top=max12y+nsw_padding)
    # axes[1,2].set_ybound(lower=min12y-nsw_padding, upper=max12y+nsw_padding)
    # print(max12y)

    for i, label in enumerate(labels):
        axes[0, 0].scatter(x_1[i], y_1[i], c=colors[i], label=labels[i], marker = markers[i], s=100)
        axes[0, 1].scatter(x_2[i], y_2[i], c=colors[i], label=labels[i], marker = markers[i], s=100)
        axes[0, 2].scatter(x_3[i], y_3[i], c=colors[i], label=labels[i], marker = markers[i], s=100)
        axes[1, 0].scatter(x_4[i], y_4[i], c=colors[i], label=labels[i], marker = markers[i], s=100)
        axes[1, 1].scatter(x_5[i], y_5[i], c=colors[i], label=labels[i], marker = markers[i], s=100)
        axes[1, 2].scatter(x_6[i], y_6[i], c=colors[i], label=labels[i], marker = markers[i], s=100)
        if i == 3:
            # upward
            # ax.plot([x, x], [y, ax.get_ylim()[1]], linestyle='--', color='gray')
            # downward
            # ax.plot([x, x], [y, ax.get_ylim()[0]], linestyle='--', color='gray')
            # leftward
            # ax.plot([x, ax.get_xlim()[0]], [y, y], linestyle='--', color='gray')
            # rightward
            # ax.plot([x, ax.get_xlim()[1]], [y, y], linestyle='--', color='gray')
            num=0
            axes[0,0].plot([x_1[i], x_1[i]], [y_1[i], axes[0,0].get_ylim()[0]+num], linestyle='--', color='gray')
            axes[0,0].plot([x_1[i], axes[0,0].get_xlim()[0]+num], [y_1[i], y_1[i]], linestyle='--', color='gray')
            axes[0,1].plot([x_2[i], x_2[i]], [y_2[i], axes[0,1].get_ylim()[0]+num], linestyle='--', color='gray')
            axes[0,1].plot([x_2[i], axes[0,1].get_xlim()[0]+num], [y_2[i], y_2[i]], linestyle='--', color='gray')
            axes[0,2].plot([x_3[i], x_3[i]], [y_3[i], axes[0,2].get_ylim()[1]+num], linestyle='--', color='gray')
            axes[0,2].plot([x_3[i], axes[0,2].get_xlim()[0]+num], [y_3[i], y_3[i]], linestyle='--', color='gray')
            axes[1,0].plot([x_4[i], x_4[i]], [y_4[i], axes[1,0].get_ylim()[0]+num], linestyle='--', color='gray')
            axes[1,0].plot([x_4[i], axes[1,0].get_xlim()[0]+num], [y_4[i], y_4[i]], linestyle='--', color='gray')
            axes[1,1].plot([x_5[i], x_5[i]], [y_5[i], axes[1,1].get_ylim()[1]+num], linestyle='--', color='gray')
            axes[1,1].plot([x_5[i], axes[1,1].get_xlim()[0]-5], [y_5[i], y_5[i]], linestyle='--', color='gray')
            axes[1,2].plot([x_6[i], x_6[i]], [y_6[i], axes[1,2].get_ylim()[1]+30], linestyle='--', color='gray')
            # print(axes[1,2].get_ylim()[1])
            axes[1,2].plot([axes[1,2].get_xlim()[0], x_6[i]+num], [y_6[i], y_6[i]], linestyle='--', color='gray')
    # print(x_2)
    # print(y_2)
    axes[0, 0].set_title(r'$\tilde{\mu}^e$ vs $\tilde{\mu}^r$', fontsize=20)
    axes[0, 0].set_xlabel(r'$\tilde{\mu}^r$', fontsize=20)
    axes[0, 0].set_ylabel(r'$\tilde{\mu}^e$', fontsize=20)
    axes[0, 1].set_title(r'$\tilde{\mu}^d$ vs $\tilde{\mu}^r$', fontsize=20)
    axes[0, 1].set_xlabel(r'$\tilde{\mu}^r$', fontsize=20)
    axes[0, 1].set_ylabel(r'$\tilde{\mu}^d$', fontsize=20)
    axes[0, 2].set_title(r'$\tilde{\mu}^{nsw}$ vs $\tilde{\mu}^r$', fontsize=20)
    axes[0, 2].set_xlabel(r'$\tilde{\mu}^r$', fontsize=20)
    axes[0, 2].set_ylabel(r'$\tilde{\mu}^{nsw}$', fontsize=20)
    axes[1, 0].set_title(r'$\tilde{\mu}^d$ vs $\tilde{\mu}^e$', fontsize=20)
    axes[1, 0].set_xlabel(r'$\tilde{\mu}^e$', fontsize=20)
    axes[1, 0].set_ylabel(r'$\tilde{\mu}^d$', fontsize=20)
    axes[1, 1].set_title(r'$\tilde{\mu}^{nsw}$ vs $\tilde{\mu}^e$', fontsize=20)
    axes[1, 1].set_xlabel(r'$\tilde{\mu}^e$', fontsize=20)
    axes[1, 1].set_ylabel(r'$\tilde{\mu}^{nsw}$', fontsize=20)
    axes[1, 2].set_title(r'$\tilde{\mu}^{nsw}$ vs $\tilde{\mu}^d$', fontsize=20)
    axes[1, 2].set_xlabel(r'$\tilde{\mu}^d$', fontsize=20)
    axes[1, 2].set_ylabel(r'$\tilde{\mu}^{nsw}$', fontsize=20)
    axes[0,0].tick_params(axis='x', labelsize=15)
    axes[0,0].tick_params(axis='y', labelsize=15)
    axes[0,1].tick_params(axis='x', labelsize=15)
    axes[0,1].tick_params(axis='y', labelsize=15)
    axes[0,2].tick_params(axis='x', labelsize=15)
    axes[0,2].tick_params(axis='y', labelsize=15)
    axes[1,0].tick_params(axis='x', labelsize=15)
    axes[1,0].tick_params(axis='y', labelsize=15)
    axes[1,1].tick_params(axis='x', labelsize=15)
    axes[1,1].tick_params(axis='y', labelsize=15)
    axes[1,2].tick_params(axis='x', labelsize=15)
    axes[1,2].tick_params(axis='y', labelsize=15)

    # After plotting all points:
    for i in range(2):
        for j in range(3):
            if i == 1 and j == 1:
                draw_arrow_nsw(axes[i, j], direction='vertical')
                draw_arrow_rest(axes[i, j], direction='horizontal')
                mark_pareto_optimality_5(axes[i,j], x_5, y_5) # fifth
            elif j == 2:
                draw_arrow_nsw(axes[i, j], direction='vertical')
                draw_arrow_rest(axes[i, j], direction='horizontal')
                if i == 0:
                    mark_pareto_optimality_3(axes[i, j], x_3, y_3) # third
                else:
                    mark_pareto_optimality_6(axes[i, j], x_6, y_6) # sixth
            else:
                draw_arrow_rest(axes[i, j], direction='horizontal')
                draw_arrow_rest(axes[i, j], direction='vertical')
                if i == 0 and j == 0:
                    mark_pareto_optimality_1(axes[i, j], x_1, y_1) # first
                elif i == 0 and j == 1:
                    mark_pareto_optimality_2(axes[i, j], x_2, y_2) # second
                else:
                    mark_pareto_optimality_4(axes[i, j], x_4, y_4) # fourth


    plt.tight_layout()
    plt.suptitle(f'Pairwise Comparison of baseline matchings for n = {n}', fontsize=25)
    plt.subplots_adjust(top=0.9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', framealpha=0, bbox_to_anchor=(0, 0.5), fontsize=20)
    output_folder = "./pareto_plots"
    os.makedirs(output_folder, exist_ok=True)
    filename = f"n={n}_create_preflist_pareto_plot.pdf"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight', format='pdf')

def padded_limits(data, pad_ratio=0.05):
    dmin, dmax = min(data), max(data)
    pad = (dmax - dmin) * pad_ratio
    return dmin - pad, dmax + pad

def plot_3d(n, c1, c2, c3, c4, d1, d2, d3, d4, e1, e2, e3, e4, nsw1, nsw2, nsw3, nsw4):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax_2 = fig.add_subplot(1, 2, 2, projection='3d')
    x = [c1, c2, c3, c4]
    y = [d1, d2, d3, d4]
    z = [e1, e2, e3, e4]
    w = [nsw1, nsw2, nsw3, nsw4]
    l2_disp = [np.sqrt(a**2 + b**2 + c**2) for a, b, c in zip(x, y, z)]
    l2_nsw = [np.sqrt(a**2+b**2+c**2) for a, b, c in zip(x, y, w)]
    colors = ['red', 'purple', 'blue', 'green']
    labels = ['Min-Regret Matching', 'Egalitarian Matching', 'Sex-Equal Matching', 'NSW Matching']
    markers = ['o', 'x', 's', '*']
    for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], marker=markers[i], c=colors[i], label=labels[i], s=200)
        # label points with their l2
        ax.text(x[i], y[i], z[i], f'{l2_disp[i]:.2f}', fontsize=15, color='black')
    # ax.quiver(0, 0, 0, arrow_length, 0, 0, color='blue', arrow_length_ratio=0.1)
    # ax.quiver(0, 0, 0, 0, arrow_length, 0, color='green', arrow_length_ratio=0.1)
    # ax.quiver(0, 0, 0, 0, 0, arrow_length, color='purple', arrow_length_ratio=0.1)
    # for i in range(len(x)):
    #     ax_2.scatter(x[i], y[i], w[i], marker=markers[i], c=colors[i], label=labels[i], s=200)
    #     # label points with their l2
    #     ax_2.text(x[i], y[i], w[i], f'{l2_nsw[i]:.2f}', fontsize=10, color='black')
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(min(z), max(z))
    ax.set_xlim(padded_limits(x))
    ax.set_ylim(padded_limits(y))
    ax.set_zlim(padded_limits(z))
    ax.legend()
    ax.set_xlabel('Regret')
    ax.set_ylabel('Egalitarian')
    ax.set_zlabel('Disparity')
    # ax_2.set_xlim(min(x), max(x))
    # ax_2.set_ylim(min(y), max(y))
    # ax_2.set_zlim(min(w), max(w))
    # ax_2.set_xlim(padded_limits(x))
    # ax_2.set_ylim(padded_limits(y))
    # ax_2.set_zlim(padded_limits(w))
    # ax_2.legend()
    # ax_2.set_xlabel('Regret')
    # ax_2.set_ylabel('Egalitarian')
    # ax_2.set_zlabel('NSW')
    plt.title("3D Plot")
    plt.show()

def plot_regret(agents, avg_1, avg_2, var_1, var_2):
    plt.figure(figsize=(10, 6))
    plt.plot(agents, avg_1, label='SNSW Matching', marker='o', color='blue')
    plt.plot(agents, avg_2, label='NSW Matching', marker='x', color='red')
    plt.fill_between(agents, np.array(avg_1) - np.array(var_1), np.array(avg_1) + np.array(var_1), alpha=0.2)
    plt.fill_between(agents, np.array(avg_2) - np.array(var_2), np.array(avg_2) + np.array(var_2), alpha=0.2)

    plt.xlabel('Number of Agents', fontsize = 20)
    plt.ylabel('Average Regret Value', fontsize = 20)
    plt.title('Comparison of Matching Criteria', fontsize = 30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()

def plot_egalitarian(agents, avg_1, avg_2, var_1, var_2):
    plt.figure(figsize=(10, 6))
    plt.plot(agents, avg_1, label='SNSW Matching', marker='o', color='blue')
    plt.plot(agents, avg_2, label='NSW Matching', marker='x', color='red')
    plt.fill_between(agents, np.array(avg_1) - np.array(var_1), np.array(avg_1) + np.array(var_1), alpha=0.2)
    plt.fill_between(agents, np.array(avg_2) - np.array(var_2), np.array(avg_2) + np.array(var_2), alpha=0.2)

    plt.xlabel('Number of Agents', fontsize = 20)
    plt.ylabel('Average Egalitarian Welfare', fontsize = 20)
    plt.title('Comparison of Matching Criteria', fontsize = 30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()

def plot_disparity(agents, avg_1, avg_2, var_1, var_2):
    plt.figure(figsize=(10, 6))
    plt.plot(agents, avg_1, label='SNSW Matching', marker='o', color='blue')
    plt.plot(agents, avg_2, label='NSW Matching', marker='x', color='red')
    plt.fill_between(agents, np.array(avg_1) - np.array(var_1), np.array(avg_1) + np.array(var_1), alpha=0.2)
    plt.fill_between(agents, np.array(avg_2) - np.array(var_2), np.array(avg_2) + np.array(var_2), alpha=0.2)

    plt.xlabel('Number of Agents', fontsize = 20)
    plt.ylabel('Average Sex-equality Measure', fontsize = 20)
    plt.title('Comparison of Matching Criteria', fontsize = 30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()



def plot_nsw(agents, avg_1, avg_2, var_1, var_2):
    plt.figure(figsize=(10, 6))
    plt.plot(agents, avg_1, label='SNSW Matching', marker='o', color='blue')
    plt.plot(agents, avg_2, label='NSW Matching', marker='x', color='red')
    plt.fill_between(agents, np.array(avg_1) - np.array(var_1), np.array(avg_1) + np.array(var_1), alpha=0.2)
    plt.fill_between(agents, np.array(avg_2) - np.array(var_2), np.array(avg_2) + np.array(var_2), alpha=0.2)

    plt.xlabel('Number of Agents', fontsize = 20)
    plt.ylabel('Average NSW Value', fontsize = 20)
    plt.title('Comparison of Matching Criteria', fontsize = 30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()

def plot_circle(n, c1, c2, c3, c4, d1, d2, d3, d4, e1, e2, e3, e4, nsw1, nsw2, nsw3, nsw4):
    measures = [r'$\tilde{\mu}^{r}$', r'$\tilde{\mu}^{e}$', r'$\tilde{\mu}^{d}$', r'$\tilde{\mu}^{nsw}$']
    # measures of egalitarian, regret, sex-equality and nsw
    algs = [r'$\mathcal{B}_1$', r'$\mathcal{B}_2$', r'$\mathcal{B}_3$', r'SNSW-Alg']
    # theiralgo, min-regret algo, sex-equality algo, ouralgo
    num_vars = len(measures)
    # 4 measures, num_vars = 4
    alg1 = [c1, d1, e1, nsw1]
    # regret algo measures
    alg2 = [c2, d2, e2, nsw2]
    # egalitarian algo measures
    alg3 = [c3, d3, e3, nsw3]
    # sex-equality algo measures
    alg4 = [c4, d4, e4, nsw4]
    # NSW algo measures

    c_max = max(c1, c2, c3, c4)
    # max regret
    d_max = max(d1, d2, d3, d4)
    # max egalitarian
    e_max = max(e1, e2, e3, e4)
    # max sex-equality
    nsw_max = max(nsw1, nsw2, nsw3, nsw4)
    # max nsw
    c_min = min(c1, c2, c3, c4)
    # min regret
    d_min = min(d1, d2, d3, d4)
    # min egalitarian
    e_min = min(e1, e2, e3, e4)
    # min sex-equality
    nsw_min = min(nsw1, nsw2, nsw3, nsw4)
    # min nsw
    maxes = [c_max, d_max, e_max, nsw_max]
    # max values of the measures
    mins = [c_min, d_min, e_min, nsw_min]
    # min values of the measures
    # normalization follows
    alg1 = normalize(alg1, maxes, mins)
    alg2 = normalize(alg2, maxes, mins)
    alg3 = normalize(alg3, maxes, mins)
    alg4 = normalize(alg4, maxes, mins)
    
    area1 = 0.5 * alg1[2] * (alg1[1] + alg1[3])
    area2 = 0.5 * alg2[3] * (alg2[0] + alg2[2])
    area3 = 0.5 * alg3[0] * (alg3[1] + alg3[3])
    area4 = 0.5 * alg4[1] * (alg4[0] + alg4[2])
    areas = [area1, area2, area3, area4]

    values = [alg2, alg1, alg3, alg4]
    # print(values)
    colors = ['blue', 'red', 'purple', 'green']
    markers = ['o', 'x', 's', '*']
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(20, 12), subplot_kw=dict(polar=True))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, polar=True)
    for i in range(4):
        ax.plot(angles, values[i], marker=markers[i],markersize=10, color=colors[i], linewidth=1, label = algs[i])
        ax.fill(angles, values[i], color=colors[i], alpha=0.25)
        # label the area
        # ax.text(angles[(i+2)%4], 0.5, f'Area: {areas[i]:.2f}', fontsize=12, color=colors[i], ha='center', va='top')
    ax.text(angles[2], 0.65, f'Area: {areas[0]:.5f}', fontsize=25, color=colors[1], ha='center', va='top')
    ax.text(angles[3], 0.25, f'Area: {areas[1]:.5f}', fontsize=25, color=colors[0], ha='center', va='top')
    ax.text(angles[0], 0.65, f'Area: {areas[2]:.5f}', fontsize=25, color=colors[2], ha='center', va='top')
    ax.text(angles[1], 0.25, f'Area: {areas[3]:.5f}', fontsize=25, color=colors[3], ha='center', va='top')
    ax.set_xticks(angles)
    ax.set_xticklabels(measures, fontsize=25)
    # ax.yticks(fontsize=2)
    ax.tick_params(axis='y', labelsize=25)

    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=30)
    # ax.set_aspect('equal')
    # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout()
    # mng = plt.get_current_fig_manager()
    # try:
    #     mng.window.state('iconic')  # Qt5Agg backend
    # except AttributeError:
    #     try:
    #         mng.window.showMaximized()  # For Qt backend
    #     except:
    #         pass  # Skip if backend doesn't support window maximizing
    output_folder = "./circle_plots"
    os.makedirs(output_folder, exist_ok=True)
    filename = f"n={n}_create_preflist_circular_plot.pdf"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight', format='pdf')

def plot_bps(agents, avg_bp, avg_ba, max_bp, max_ba, min_bp, min_ba):
    plt.figure(figsize=(10, 6))
    plt.plot(agents, avg_bp, label='# Blocking Pairs', marker='o', color='blue')
    plt.plot(agents, avg_ba, label='# Blocking Agents', marker='x', color='red')
    # Shaded region for variance
    avg_bp = np.array(avg_bp)
    avg_ba = np.array(avg_ba)
    # var_bp = np.array(var_bp)
    # var_ba = np.array(var_ba)

    plt.fill_between(agents, min_bp, max_bp, color='blue', alpha=0.2)
    plt.fill_between(agents, min_ba, max_ba, color='red', alpha=0.2)
    plt.xlabel('Number of Agents', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    # plt.title('Comparison of Blocking pairs across NSW and SNSW Matchings', fontsize=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()

process_all_files('./')