from nltk import sent_tokenize
from util.embedding import embedding_model
from util.vector_store import add_chunks, get_or_init_method
from util.load_papers import load_papers


def main():
    collection = get_or_init_method("sentence_based")

    for data, fn in load_papers():
        chunks = []
        sections = []
        for section in data["content"]:
            sentences = sent_tokenize(section["text"])
            i = 0
            while i < len(sentences):
                chunk_sentences = sentences[i : i + 5]
                chunks.append(" ".join(chunk_sentences))
                sections.append(section["section"])
                i += 4
        embeddings = embedding_model.encode(chunks)
        add_chunks(collection, fn, data["title"], chunks, embeddings, sections)


if __name__ == "__main__":
    main()
