from chromadb import PersistentClient
import json
from uuid import uuid4
from sentence_transformers import SentenceTransformer


def main():
    client = PersistentClient("chunks")
    collection = client.get_or_create_collection("fixed_length")

    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", backend="torch")

    for i in range(10):
        fn = f"p{i}"
        with open(f"data/processed/{fn}.json") as f:
            data = json.load(f)
        mds = []
        chunks = []
        ids = []
        for section in data["content"]:
            chunk = ""
            for word in section["text"].split(" "):
                if len(chunk) > 2000:
                    chunks.append(chunk)
                    mds.append({"section": section["section"], "title": data["title"]})
                    ids.append(f"{fn}_{str(uuid4())}")
                    chunk = ""
                    continue
                chunk += word + " "
            if len(chunk) != 0:
                chunks.append(chunk)
                mds.append({"section": section["section"], "title": data["title"]})
                ids.append(f"{fn}_{str(uuid4())}")
        embeddings = embedding_model.encode(chunks)
        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=mds)


if __name__ == "__main__":
    main()
