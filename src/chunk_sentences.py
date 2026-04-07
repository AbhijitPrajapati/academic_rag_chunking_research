def chunk_sentences(sentences, similarity_threshold, sentence_overlap, embedding_model):
    embeddings = embedding_model.encode(sentences)
    similarities = embedding_model.similarity_pairwise(embeddings[:-1], embeddings[1:])

    dip_idx = [
        i for i in range(len(similarities)) if similarities[i] < similarity_threshold
    ]

    chunks = []
    current = [sentences[0]]
    for i, s in enumerate(sentences[1:]):
        if i in dip_idx:
            chunks.append(" ".join(current))
            ovr = min(sentence_overlap, len(current))
            current = current[-ovr:] + [sentences[i + 1]]
        else:
            current.append(s)
    if current:
        chunks.append(" ".join(current))
    return chunks
