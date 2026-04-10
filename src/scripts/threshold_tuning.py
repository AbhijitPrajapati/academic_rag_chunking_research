import json

SIMILARITY_THRESHOLD = 0.7

with open("sim.json", "r") as f:
    similarities = json.load(f)

for thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.9]:
    dip_idx = [i for i in range(len(similarities)) if similarities[i] < thresh]
    sizes = [dip_idx[i + 1] - dip_idx[i] for i in range(len(dip_idx) - 1)]
    print(f"{thresh}: {round(sum(sizes) / len(sizes), 3)}")
