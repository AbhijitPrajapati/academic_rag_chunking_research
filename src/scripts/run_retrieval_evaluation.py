from chromadb import PersistentClient
import json
from src.retrieval import retrieve_chunks
from src.metrics import Evaluator
from .constants import SIMILARITY_THRESHOLDS

client = PersistentClient("chunks")


def main():
    with open("data/prompts/prompts.json") as f:
        data = json.load(f)
    queries = []
    targets = []
    for p in data:
        queries.append(p["question"])
        targets.append(p["evidence"])

    evaluator = Evaluator(targets)

    fixed = client.get_collection("fixed_length")
    sentence_based = client.get_collection("sentence_based")
    semantic = client.get_collection("semantic")

    fixed_chunks = retrieve_chunks(queries, 5, fixed)
    sentence_based_chunks = retrieve_chunks(queries, 5, sentence_based)
    semantic_chunks = {
        thresh: retrieve_chunks(queries, 5, semantic, similarity_threshold=thresh)
        for thresh in SIMILARITY_THRESHOLDS
    }

    results = {
        "fixed_length": evaluator.get_metrics(fixed_chunks),  # type: ignore
        "sentence_based": evaluator.get_metrics(sentence_based_chunks),  # type: ignore
        "semantic": {k: evaluator.get_metrics(v) for k, v in semantic_chunks.items()},  # type: ignore
    }

    with open("results/results.json", "w") as f:
        json.dump(results, f)
