"""Extracts data from real-world dataset"""

import re
from pathlib import Path
import numpy as np

def data_processor(file_path):
    """Processes the text file present in file_path"""
    text = Path(file_path).read_text(encoding="utf-8")

    # Split into treatment blocks
    blocks = re.split(
        r'_{20,}\n\n(?=\d+/31:)',
        text
    )

    markets = []

    for block in blocks:
        header_match = re.search(
            r'(\d+)/31:\s*(.*?)\n\(Experimental markets:\s*(\d+)\)',
            block,
            re.S
        )

        if not header_match:
            continue

        treatment_id = int(header_match.group(1))
        treatment_name = header_match.group(2).strip()
        experimental_markets = int(header_match.group(3))

        lines = block.splitlines()

        # rows beginning with food-
        food_rows = [ln for ln in lines if re.match(r'^food-\d+', ln.strip())]

        if not food_rows:
            continue

        food_weights = []
        color_weights = []

        for row in food_rows:

            # extract all payoff pairs "x, y"
            pairs = re.findall(r'(\d+),\s*(\d+)', row)

            food_weights.append([int(x) for x, y in pairs])
            color_weights.append([int(y) for x, y in pairs])

        food_weights = np.array(food_weights)
        color_weights = np.array(color_weights)

        markets.append({
            "treatment_id": treatment_id,
            "treatment_name": treatment_name,
            "experimental_markets": experimental_markets,
            "n_food_agents": food_weights.shape[0],
            "n_color_agents": food_weights.shape[1],
            "food_weights": food_weights,
            "color_weights": color_weights,
        })

    return markets
    # # Example: print summary
    # for m in markets:
    #     print(
    #         f"{m['treatment_id']:>2}: {m['treatment_name']}"
    #         f" | agents=({m['n_food_agents']}, {m['n_color_agents']})"
    #         f" | experimental_markets={m['experimental_markets']}"
    #     )
    # # Example: access one market
    # m = markets[0]

    # print("\nTreatment:", m["treatment_name"])
    # print("Food weight matrix:")
    # print(m["food_weights"])

    # print("\nColor weight matrix:")
    # print(m["color_weights"])]
