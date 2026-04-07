from chromadb import PersistentClient
import json
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from nltk import sent_tokenize
from chunk_sentences import chunk_sentences

SIMILARITY_THRESHOLD = 0.7


def main():
    client = PersistentClient("chunks")
    collection = client.get_or_create_collection("sentence_based")

    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", backend="torch")

    for i in range(10):
        fn = f"p{i}"
        with open(f"data/processed/{fn}.json") as f:
            data = json.load(f)
        mds = []
        chunks = []
        ids = []
        for section in data["content"]:
            chunks.extend(
                chunk_sentences(
                    sent_tokenize(section["text"]),
                    SIMILARITY_THRESHOLD,
                    1,
                    embedding_model,
                )
            )
            mds.append({"section": section["section"], "title": data["title"]})
            ids.append(f"{fn}_{str(uuid4())}")
        embeddings = embedding_model.encode(chunks)
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=mds)


if __name__ == "__main__":
    main()
