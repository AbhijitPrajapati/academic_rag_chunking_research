from chromadb import PersistentClient
from uuid import uuid4

client = PersistentClient("chunks")


def add_chunks(method, fn, title, chunks, embeddings, sections):
    collection = client.get_or_create_collection(method)
    ids = [f"{fn}_{str(uuid4())}" for _ in range(len(chunks))]
    mds = [{"title": title, "section": s} for s in sections]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=mds)  # type: ignore
