import matplotlib.pyplot as plt
import numpy as np
import json


def main():
    with open("results/retrieval.json", "r") as f:
        data = json.load(f)

    data = [data["fixed_length"], data["sentence_based"], data["semantic"]]
    mrp, recall_1, recall_5, precision_1, precision_5 = map(
        list,
        zip(
            *(
                (
                    m["mean_reciprocal_rank"],
                    m["recall@k"]["1"],
                    m["recall@k"]["5"],
                    m["precision@k"]["1"],
                    m["precision@k"]["5"],
                )
                for m in data
            )
        ),
    )

    method_labels = ["Fixed Length", "Sentence Based", "Semantic"]

    fig_mrp, ax_mrp = plt.subplots(figsize=(6, 6))

    ax_mrp.bar(method_labels, mrp, color="steelblue", alpha=0.7)
    ax_mrp.set_title("Mean Reciprocal Rank")
    ax_mrp.grid(axis="y", alpha=0.3)

    fig_recall, ax_recall = plt.subplots(figsize=(6, 6))

    x = np.arange(len(method_labels))
    width = 0.25

    ax_recall.bar(x - width / 2, recall_1, width, label="k=1", alpha=0.8)
    ax_recall.bar(x + width / 2, recall_5, width, label="k=5", alpha=0.8)

    ax_recall.set_title("Recall@K")
    ax_recall.set_xticks(x)
    ax_recall.set_xticklabels(method_labels)
    ax_recall.legend()
    ax_recall.grid(axis="y", alpha=0.3)

    fig_precision, ax_precision = plt.subplots(figsize=(6, 6))

    width = 0.25

    ax_precision.bar(x - width / 2, precision_1, width, label="k=1", alpha=0.8)
    ax_precision.bar(x + width / 2, precision_5, width, label="k=5", alpha=0.8)

    ax_precision.set_title("Precision@K")
    ax_precision.set_xticks(x)
    ax_precision.set_xticklabels(method_labels)
    ax_precision.legend()
    ax_precision.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
