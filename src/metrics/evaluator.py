from nltk.tokenize import word_tokenize

K_EVAL = [1, 3, 5]
NORM_THRESHOLD = 0.3
LEN_THRESHOLD = 2


def is_relevant(overlap, target_len):
    return (overlap / target_len) > NORM_THRESHOLD and overlap > LEN_THRESHOLD


def get_bigrams(texts_list: list[list[str]]):
    out = []
    for texts in texts_list:
        intermediate = []
        for text in texts:
            tokens = [t for t in word_tokenize(text.lower().strip()) if t.isalnum()]
            intermediate.append(set(zip(tokens[:-1], tokens[1:])))
        out.append(intermediate)
    return out


class Evaluator:
    def __init__(self, targets_list: list[list[str]]):
        self.targets_list_bigrams = get_bigrams(targets_list)

    def get_metrics(self, chunks_list: list[list[str]]):
        bigrams = get_bigrams(chunks_list)
        mrp = 0.0
        recall = {k: 0.0 for k in K_EVAL}
        precision = {k: 0.0 for k in K_EVAL}
        for chunks, targets in zip(bigrams, self.targets_list_bigrams):
            num_rel = {k: 0 for k in K_EVAL}
            min_rank = None
            for i, chunk in enumerate(chunks[: max(K_EVAL)]):
                for target in targets:
                    overlap = len(chunk & target)
                    if is_relevant(overlap, len(target)):
                        for k in num_rel.keys():
                            if i < k:
                                num_rel[k] += 1
                        if min_rank is None:
                            min_rank = i + 1
                        break
            if min_rank is None:
                min_rank = len(chunks) + 1
            mrp += 1 / min_rank
            for k, v in num_rel.items():
                recall[k] += v / len(targets)
                precision[k] += v / k

        norm = len(self.targets_list_bigrams)
        mrp /= norm
        for k in K_EVAL:
            recall[k] /= norm
            precision[k] /= norm

        return {
            "mean_reciprocal_rank": mrp,
            "recall@k": recall,
            "precision@k": precision,
        }
