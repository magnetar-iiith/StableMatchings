import json
import os
import glob
import re

def check_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    if not lines:
        return
        
    first_line = json.loads(lines[0])
    if "preflist_common" not in first_line:
        print(f"Skipping {filepath}: no preflist_common found")
        return
        
    preflist_common = first_line["preflist_common"]
    pref_men = preflist_common[0]
    pref_women = preflist_common[1]
    
    basename = os.path.basename(filepath)
    
    # Parse the filename to find which men and women are modified
    parts = basename.split('_')
    
    modified_entities = {}
    
    for i in range(len(parts) - 1):
        if parts[i] == 'man':
            idx = int(parts[i+1])
            modified_entities[f"pref_man_{idx}"] = pref_men[idx]
        elif parts[i] == 'woman':
            idx = int(parts[i+1])
            modified_entities[f"pref_woman_{idx}"] = pref_women[idx]
            
    min_ratio = float('inf')
    base_ratio = None
    
    for line in lines[1:]:
        data = json.loads(line)
        ratio = data["scores"]["ratio"]
        if ratio < min_ratio:
            min_ratio = ratio
            
        # Check if this line corresponds to base_preflist
        is_base = True
        for key, expected_pref in modified_entities.items():
            if data[key] != expected_pref:
                is_base = False
                break
                
        if is_base:
            base_ratio = ratio
            
    if base_ratio is None:
        print(f"File: {basename} - ERROR: base_preflist not found among modifications")
    else:
        is_min = (base_ratio <= min_ratio)
        print(f"File: {basename}")
        print(f"  Base Ratio: {base_ratio}")
        print(f"  Min Ratio across all mods: {min_ratio}")
        print(f"  Is base ratio the minimum? {is_min}")

def main():
    dir_path = "./AAAI_2027_final_exps/"
    files = glob.glob(os.path.join(dir_path, "*.json"))
    for file in sorted(files):
        check_file(file)

if __name__ == "__main__":
    main()
