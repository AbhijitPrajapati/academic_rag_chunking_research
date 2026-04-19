from src.nlp import tokenizer


def get_bigrams(texts_list: list[list[str]]):
    out = []
    for texts in texts_list:
        intermediate = []
        for text in texts:
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            intermediate.append(set(zip(tokens[:-1], tokens[1:])))
        out.append(intermediate)
    return out


class Evaluator:
    def __init__(
        self,
        targets_list: list[list[str]],
        k_vals,
        overlap_relevancy_ratio_threshold,
        overlap_relevancy_length_threshold,
    ):
        self.targets_list_bigrams = get_bigrams(targets_list)
        self.k_vals = k_vals
        self.orrt = overlap_relevancy_ratio_threshold
        self.orlt = overlap_relevancy_length_threshold

    def get_metrics(self, chunks_list: list[list[str]]):
        bigrams = get_bigrams(chunks_list)
        mrp = 0.0
        recall = {k: 0.0 for k in self.k_vals}
        precision = {k: 0.0 for k in self.k_vals}
        for chunks, targets in zip(bigrams, self.targets_list_bigrams):
            num_rel = {k: 0 for k in self.k_vals}
            min_rank = len(chunks) + 1
            for target in targets:
                for i, chunk in enumerate(chunks[: max(self.k_vals)]):
                    overlap = len(chunk & target)
                    if (overlap / len(target)) > self.orrt and overlap > self.orlt:
                        for k in num_rel.keys():
                            if i < k:
                                num_rel[k] += 1
                        min_rank = min(min_rank, i + 1)
                        break
            mrp += 1 / min_rank
            for k, v in num_rel.items():
                recall[k] += v / len(targets)
                precision[k] += v / k

        norm = len(self.targets_list_bigrams)
        mrp /= norm
        for k in self.k_vals:
            recall[k] /= norm
            precision[k] /= norm

        return {
            "mean_reciprocal_rank": mrp,
            "recall@k": recall,
            "precision@k": precision,
        }
