from util.embedding import embedding_model
from util.vector_store import add_chunks
from util.load_papers import load_papers


def main():
    for data, fn in load_papers():
        chunks = []
        sections = []
        for section in data["content"]:
            chunk = ""
            for word in section["text"].split(" "):
                if len(chunk) > 2000:
                    chunks.append(chunk)
                    sections.append(section["section"])
                    chunk = ""
                    continue
                chunk += word + " "
            if len(chunk) != 0:
                chunks.append(chunk)
                sections.append(section["section"])
        embeddings = embedding_model.encode(chunks)
        add_chunks("fixed_length", fn, data["title"], chunks, embeddings, sections)


if __name__ == "__main__":
    main()
