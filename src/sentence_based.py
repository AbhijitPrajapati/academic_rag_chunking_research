from nltk import sent_tokenize
from util.embedding import embedding_model
from util.vector_store import add_chunks
from util.load_papers import load_papers


def main():
    for data, fn in load_papers():
        chunks = []
        sections = []
        for section in data["content"]:
            sentences = sent_tokenize(section["text"])
            chunk = []
            for s in sentences:
                if len(chunk) >= 8:
                    chunks.append(" ".join(chunk))
                    sections.append(section["section"])
                    chunk = []
                    continue
                chunk.append(s)
            if len(chunk) != 0:
                chunks.append(" ".join(chunk))
                sections.append(section["section"])
        embeddings = embedding_model.encode(chunks)
        add_chunks("sentence_based", fn, data["title"], chunks, embeddings, sections)


if __name__ == "__main__":
    main()
