from chromadb import PersistentClient
import numpy as np
from src.chunking import semantic_chunking
from tqdm import tqdm
import json


def load_papers():
    for i in tqdm(range(10)):
        fn = f"p{i}"
        with open(f"data/processed/{fn}.json") as f:
            data = json.load(f)
        yield data, fn

# chunk semantically with different similarity thresholds to see if high std subsides or not, as well as chunk per document count

client = PersistentClient("chunks")

fixed = client.get_collection("fixed_length")
sent = client.get_collection("sentence_based")
semantic = client.get_collection("semantic")

all_papers = set([m["title"] for m in fixed.get()["metadatas"]])  # type: ignore


f = [c["title"] for c in fixed.get()["metadatas"]]  # type: ignore
sen = [c["title"] for c in sent.get()["metadatas"]]  # type: ignore
sem = [c["title"] for c in semantic.get()["metadatas"]]  # type: ignore

diff = [sem.count(p) - sen.count(p) for p in all_papers]
print(np.mean(diff))
print(diff)

# Chunks Lengths
# Fixed:
# Mean: 1437.15 STD: 372.32

# Sentence Based:
# Mean: 936.48 STD: 443.34

# Semantic:
# Mean: 1033.91 STD: 1973.23
