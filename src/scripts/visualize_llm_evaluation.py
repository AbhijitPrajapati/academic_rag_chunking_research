import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json


def main():
    with open("results/generated_response.json", "r") as f:
        data = json.load(f)

    methods = ["fixed_length", "sentence_based", "semantic"]
    points = [data["points"][m] for m in methods]
    win_rate = [data["win_rate"][m] for m in methods]

    method_labels = ["Fixed Length", "Sentence Based", "Semantic"]

    fig_bar, ax_bar = plt.subplots(figsize=(6, 6))

    x = np.arange(len(method_labels))
    width = 0.25

    ax_bar.bar(x - width / 2, points, width, label="Avg. Points", alpha=0.8)
    ax_bar.bar(x + width / 2, win_rate, width, label="Win Rate", alpha=0.8)

    ax_bar.set_title("Points and Win Rate")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(method_labels)
    ax_bar.legend()
    ax_bar.grid(axis="y", alpha=0.3)

    fig_mat, ax_mat = plt.subplots(figsize=(6, 6))

    pairwise_mat = np.zeros((3, 3))
    for i1, m1 in enumerate(methods):
        for i2, m2 in enumerate(methods):
            if m2 == m1:
                continue
            pairwise_mat[i1][i2] = data["pairwise"][m1][m2]

    sns.heatmap(
        pairwise_mat, annot=True, xticklabels=method_labels, yticklabels=method_labels
    )
    ax_mat.set_xlabel("Competing Method")
    ax_mat.set_ylabel("Winning Method")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
