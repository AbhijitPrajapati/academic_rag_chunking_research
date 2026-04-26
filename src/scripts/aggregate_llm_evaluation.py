import json


def main():
    with open("results/response_evaluation.json", "r") as f:
        data = json.load(f)
    points = {"fixed_length": 0.0, "sentence_based": 0.0, "semantic": 0.0}
    win_rate = {"fixed_length": 0.0, "semantic": 0.0, "sentence_based": 0.0}
    pairwise = {
        "fixed_length": {"sentence_based": 0.0, "semantic": 0.0},
        "sentence_based": {"fixed_length": 0.0, "semantic": 0.0},
        "semantic": {"fixed_length": 0.0, "sentence_based": 0.0},
    }
    for question in data:
        ranking = list(
            map(
                lambda r: question["shuffle_key"][int(r)],
                question["response"].split(" "),
            )
        )
        points[ranking[0]] += 2
        points[ranking[1]] += 1
        win_rate[ranking[0]] += 1
        pairwise[ranking[0]][ranking[1]] += 1
        pairwise[ranking[0]][ranking[2]] += 1
        pairwise[ranking[1]][ranking[2]] += 1
    for m in points.keys():
        points[m] /= len(data)
        win_rate[m] /= len(data)
        for cm in pairwise[m].keys():
            pairwise[m][cm] /= len(data)

    with open("results/generated_response.json", "w") as f:
        json.dump(
            {"points": points, "win_rate": win_rate, "pairwise": pairwise}, f, indent=4
        )


if __name__ == "__main__":
    main()
