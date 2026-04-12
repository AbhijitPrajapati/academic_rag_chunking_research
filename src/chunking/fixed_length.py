from nltk.tokenize import word_tokenize


def fixed_length_chunking(sections):
    chunks = []
    section_names = []
    for section in sections:
        i = 0
        words = word_tokenize(section["text"])
        while i < len(words):
            chunk_words = words[i : i + 250]
            chunks.append(" ".join(chunk_words))
            section_names.append(section["section"])
            i += 200
    return chunks, section_names
