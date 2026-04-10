from nltk import sent_tokenize
from util.embedding import embedding_model, chunk_sentences_semantically
from util.vector_store import add_chunks
from util.load_papers import load_papers


def main():
    for data, fn in load_papers():
        chunks = []
        sections = []
        for section in data["content"]:
            for res in [0.5, 0.6, 0.65]:
                c = chunk_sentences_semantically(sent_tokenize(section["text"]), res, 1)
                chunks.extend(c)
                sections.extend([section["section"]] * len(c))
        embeddings = embedding_model.encode(chunks)
        add_chunks("multiresolution", fn, data["title"], chunks, embeddings, sections)


if __name__ == "__main__":
    main()
