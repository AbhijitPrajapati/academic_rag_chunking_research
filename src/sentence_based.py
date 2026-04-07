from chromadb import PersistentClient
import json
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from nltk import sent_tokenize


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
            sentences = sent_tokenize(section["text"])
            chunk = []
            for s in sentences:
                if len(chunk) >= 8:
                    chunks.append(" ".join(chunk))
                    mds.append({"section": section["section"], "title": data["title"]})
                    ids.append(f"{fn}_{str(uuid4())}")
                    chunk = []
                    continue
                chunk.append(s)
            if len(chunk) != 0:
                chunks.append(" ".join(chunk))
                mds.append({"section": section["section"], "title": data["title"]})
                ids.append(f"{fn}_{str(uuid4())}")
        embeddings = embedding_model.encode(chunks)
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=mds)


if __name__ == "__main__":
    main()
