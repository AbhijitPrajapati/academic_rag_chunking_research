from util.embedding import embedding_model
from util.vector_store import add_chunks, get_or_init_method
from util.load_papers import load_papers


def main():
    collection = get_or_init_method("fixed_length")

    for data, fn in load_papers():
        chunks = []
        sections = []
        for section in data["content"]:
            i = 0
            words = section["text"].split()
            while i < len(words):
                chunk_words = words[i : i + 250]
                chunks.append(" ".join(chunk_words))
                sections.append(section["section"])
                i += 200
        embeddings = embedding_model.encode(chunks)
        add_chunks(collection, fn, data["title"], chunks, embeddings, sections)


if __name__ == "__main__":
    main()
