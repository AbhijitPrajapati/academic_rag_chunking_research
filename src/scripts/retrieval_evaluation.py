from chromadb import PersistentClient
import json
from src.retrieval import retrieve_chunks
from src.metrics import Evaluator
from src.constants import (
    EVAL_K,
    EVAL_LENGTH_THRESHOLD,
    EVAL_RATIO_THRESHOLD,
    EVAL_N_CHUNKS,
)

client = PersistentClient("chunks")


def main():
    with open("data/prompts/prompts.json", errors="ignore") as f:
        data = json.load(f)
    queries = []
    targets = []
    for p in data:
        queries.append(p["question"])
        targets.append(p["evidence"])

    evaluator = Evaluator(targets, EVAL_K, EVAL_RATIO_THRESHOLD, EVAL_LENGTH_THRESHOLD)

    results = []
    for n_chunks in EVAL_N_CHUNKS:
        res = {"n_chunks": n_chunks}
        for method in ["fixed_length", "sentence_based", "semantic"]:
            collection = client.get_collection(method)
            chunks = retrieve_chunks(queries, n_chunks, collection)
            res[method] = evaluator.get_metrics(chunks)  # type: ignore
        results.append(res)

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
