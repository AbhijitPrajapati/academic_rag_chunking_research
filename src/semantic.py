from nltk import sent_tokenize
from util.embedding import embedding_model, chunk_sentences_semantically
from util.vector_store import add_chunks, get_or_init_method
from util.load_papers import load_papers


SIMILARITY_THRESHOLD = 0.6


def main():
    collection = get_or_init_method("semantic")

    for data, fn in load_papers():
        chunks = []
        sections = []
        for section in data["content"]:
            c = chunk_sentences_semantically(
                sent_tokenize(section["text"]), SIMILARITY_THRESHOLD, 1
            )
            chunks.extend(c)
            sections.extend([section["section"]] * len(c))
        embeddings = embedding_model.encode(chunks)
        add_chunks(collection, fn, data["title"], chunks, embeddings, sections)


if __name__ == "__main__":
    main()
