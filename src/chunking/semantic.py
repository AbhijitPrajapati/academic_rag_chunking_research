from nltk import sent_tokenize
from src.nlp import embedding_model
from src.constants import SIMILARITY_THRESHOLD


def semantic_chunking(sections):
    chunks = []
    section_names = []
    for section in sections:
        sentences = sent_tokenize(section["text"])
        if len(sentences) < 4:
            chunks.append(section["text"])
            section_names.append(section["section"])
            continue
        windows = [" ".join(sentences[i : i + 3]) for i in range(len(sentences) - 2)]
        embeddings = embedding_model.encode(windows)
        similarities = embedding_model.similarity_pairwise(
            embeddings[:-1], embeddings[1:]
        )

        c = []
        current = [windows[0]]
        skip_iter = False
        for i, s in enumerate(sentences[3:]):
            if skip_iter:
                skip_iter = False
                continue
            if similarities[i] < SIMILARITY_THRESHOLD:
                c.append(" ".join(current))
                current = [sentences[i + 2]]
                skip_iter = True
                continue
            current.append(s)
        if current:
            c.append(" ".join(current))
        chunks.extend(c)
        section_names.extend([section["section"]] * len(c))
    return chunks, section_names
