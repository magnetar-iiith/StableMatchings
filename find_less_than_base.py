import json
import os
import glob

def find_less_than_base(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    if not lines:
        return
        
    first_line = json.loads(lines[0])
    preflist_common = first_line["preflist_common"]
    pref_men = preflist_common[0]
    pref_women = preflist_common[1]
    
    basename = os.path.basename(filepath)
    parts = basename.split('_')
    
    modified_entities = {}
    for i in range(len(parts) - 1):
        if parts[i] == 'man':
            idx = int(parts[i+1])
            modified_entities[f"pref_man_{idx}"] = pref_men[idx]
        elif parts[i] == 'woman':
            idx = int(parts[i+1])
            modified_entities[f"pref_woman_{idx}"] = pref_women[idx]
            
    # First pass: find base ratio
    base_ratio = None
    for line in lines[1:]:
        data = json.loads(line)
        is_base = True
        for key, expected_pref in modified_entities.items():
            if data[key] != expected_pref:
                is_base = False
                break
        if is_base:
            base_ratio = data["scores"]["ratio"]
            break
            
    if base_ratio is None:
        print(f"ERROR: base_ratio not found for {basename}")
        return
        
    # Second pass: find all instances where ratio < base_ratio
    less_than_base_instances = []
    for line in lines[1:]:
        data = json.loads(line)
        if data["scores"]["ratio"] < base_ratio:
            less_than_base_instances.append(data)
            
    print(f"\nFile: {basename}")
    print(f"Base Ratio: {base_ratio}")
    print(f"Number of instances with ratio < base ratio: {len(less_than_base_instances)}")
    
    if less_than_base_instances:
        print("First instance:")
        print(json.dumps(less_than_base_instances[0], indent=2))
        
        # Save all instances to a file for the user
        out_filename = f"less_than_base_{basename}"
        out_filepath = os.path.join(os.path.dirname(filepath), out_filename)
        with open(out_filepath, 'w') as out_f:
            for inst in less_than_base_instances:
                out_f.write(json.dumps(inst) + "\n")
        print(f"Saved all {len(less_than_base_instances)} instances to {out_filename}")

def main():
    dir_path = "./AAAI_2027_final_exps/"
    files = glob.glob(os.path.join(dir_path, "man_0_all_modifications_n=*.json"))
    for file in sorted(files):
        find_less_than_base(file)

if __name__ == "__main__":
    main()
