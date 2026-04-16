import matplotlib.pyplot as plt
import numpy as np
import json

# values to display
N_CHUNKS = [3]
METHODS = ["sentence_based", "semantic"]
K_VAL = [1, 3, 5]
METRICS = ["precision@k", "recall@k", "mean_reciprocal_rank"]


def main():
    with open("results/results.json", "r") as f:
        data = json.load(f)

    for n_chunks_sector in data:
        n_chunks = n_chunks_sector["n_chunks"]
        if n_chunks not in N_CHUNKS:
            continue
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(len(METHODS) * 2, 6))
            if metric == "mean_reciprocal_rank":
                values = [n_chunks_sector[m][metric] for m in METHODS]
                ax.bar(METHODS, values, color="steelblue", alpha=0.7)
            else:
                width = 1 / len(K_VAL)
                x = np.arange(len(METHODS))
                for i, k in enumerate(K_VAL):
                    values = [n_chunks_sector[m][metric][str(k)] for m in METHODS]
                    ax.bar(
                        x + i * width - width / 2,
                        values,
                        width * 3 / 4,
                        label=f"k={k}",
                        alpha=0.8,
                    )
                ax.set_xticks(x)
                ax.set_xticklabels(METHODS)
                ax.legend()

            ax.set_title(f"{metric} n_chunks={n_chunks}")
            ax.grid(axis="y", alpha=0.3)

    # data = [data["fixed_length"], data["sentence_based"], data["semantic"]]
    # mrp, recall_1, recall_3, recall_5, precision_1, precision_3, precision_5 = [
    #     [] for _ in range(7)
    # ]
    # for m in data:
    #     mrp.append(m["mean_reciprocal_rank"])
    #     recall_1.append(m["recall@k"]["1"])
    #     recall_3.append(m["recall@k"]["3"])
    #     recall_5.append(m["recall@k"]["5"])
    #     precision_1.append(m["precision@k"]["1"])
    #     precision_3.append(m["precision@k"]["3"])
    #     precision_5.append(m["precision@k"]["5"])

    # method_labels = ["Fixed Length", "Sentence Based", "Semantic"]

    # fig_mrp, ax_mrp = plt.subplots(figsize=(5, 6))
    # mrp = [m["mean_reciprocal_rank"] for m in data]

    # ax_mrp.bar(method_labels, mrp, color="steelblue", alpha=0.7)
    # ax_mrp.set_title("Mean Reciprocal Rank")
    # ax_mrp.grid(axis="y", alpha=0.3)

    # fig_recall, ax_recall = plt.subplots(figsize=(6, 6))

    # x = np.arange(len(method_labels))
    # width = 0.25

    # ax_recall.bar(x, recall_1, width, label="k=1", alpha=0.8)
    # ax_recall.bar(x + width, recall_3, width, label="k=3", alpha=0.8)
    # ax_recall.bar(x + width * 2, recall_5, width, label="k=5", alpha=0.8)

    # ax_recall.set_title("Recall@K")
    # ax_recall.set_xticks(x)
    # ax_recall.set_xticklabels(method_labels)
    # ax_recall.legend()
    # ax_recall.grid(axis="y", alpha=0.3)

    # fig_precision, ax_precision = plt.subplots(figsize=(6, 6))

    # width = 0.25

    # ax_precision.bar(x - width, precision_1, width, label="k=1", alpha=0.8)
    # ax_precision.bar(x, precision_3, width, label="k=3", alpha=0.8)
    # ax_precision.bar(x + width, precision_5, width, label="k=5", alpha=0.8)

    # ax_precision.set_title("Precision@K")
    # ax_precision.set_xticks(x)
    # ax_precision.set_xticklabels(method_labels)
    # ax_precision.legend()
    # ax_precision.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
