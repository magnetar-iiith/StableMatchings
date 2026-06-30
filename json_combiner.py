import os

input_dir = "latest_worst_ratio_n=5"
output_file = "latest_worst_ratio_n=5/combined.json"
with open(output_file, 'w') as f:
    for filename in os.listdir(input_dir):
        if filename == "combined.json":
            continue
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'r') as f2:
            for line in f2:
                f.write(line)
