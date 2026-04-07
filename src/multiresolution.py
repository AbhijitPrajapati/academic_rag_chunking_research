from chromadb import PersistentClient
import json
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from nltk import sent_tokenize
from chunk_sentences import chunk_sentences


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
            for res in [0.5, 0.7, 0.8]:
                chunks.extend(
                    chunk_sentences(
                        sent_tokenize(section["text"]),
                        res,
                        1,
                        embedding_model,
                    )
                )
                mds.extend([{"section": section["section"], "title": data["title"]}])
                ids.extend([f"{fn}_{str(uuid4())}" for ])
        embeddings = embedding_model.encode(chunks)
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=mds)


if __name__ == "__main__":
    main()
