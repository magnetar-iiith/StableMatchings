"""Finds the worst ratio for a particular n
ratio of eg(Me) to eg(Msnsw)"""

import json
import os
from measures import egalitarian_welfare

def find_worst_ratio(folder_path):
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        output_folder = "./worst_ratios"
        os.makedirs(output_folder, exist_ok=True)
        with open(file_path, "r") as f:
            ratio_min = float('inf')
            for line in f:
                data = json.loads(line)
                egalitarian_matching = data["egalitarian"]
                nash_welfare_matching = data["nsw"]
                preflist = data['preflist']
                egalitarian_Me = egalitarian_welfare(matching=egalitarian_matching, preflist=preflist)
                egalitarian_Msnsw = egalitarian_welfare(matching=nash_welfare_matching, preflist=preflist)
                ratio_curr = egalitarian_Msnsw / egalitarian_Me
                if ratio_curr < ratio_min:
                    ratio_min = ratio_curr
                    output_filepath = os.path.join(output_folder, f"worst_ratio_{len(preflist[0])}")
                    with open(output_filepath, "w") as results_file:
                        results_file.write(json.dumps(line))
            print(len(preflist[0]))
            print(ratio_min)

folder_path = "paper_matchings/OneDrive_1_12-06-2026/"
find_worst_ratio(folder_path)
