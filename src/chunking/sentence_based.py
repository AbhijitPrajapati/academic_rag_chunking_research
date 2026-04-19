from nltk import sent_tokenize
from src.constants import FIXED_SENTENCES


def sentence_based_chunking(sections):
    chunks = []
    section_names = []
    for section in sections:
        sentences = sent_tokenize(section["text"])
        i = 0
        while i < len(sentences):
            chunk_sentences = sentences[i : i + FIXED_SENTENCES]
            chunks.append(" ".join(chunk_sentences))
            section_names.append(section["section"])
            i += FIXED_SENTENCES - 1
    return chunks, section_names
