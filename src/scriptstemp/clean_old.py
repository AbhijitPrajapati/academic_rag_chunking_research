import re
from wordsegment import load, segment

load()


def get_all(text):
    n = 14
    pattern = rf"\b\w{{{n + 1},}}\b"
    return re.findall(pattern, text)


for j in range(6, 7):
    with open(f"data/processed/p{j}.json", "r") as f:
        text = f.read()

    artifacts = get_all(text)
    print(f"{len(artifacts)} artifacts in {j}")
    print(artifacts[0])
    for a in artifacts:
        text = text.replace(a, " ".join(segment(a)))

    with open(f"data/processed/p{j}c.json", "w") as f:
        f.write(text)
