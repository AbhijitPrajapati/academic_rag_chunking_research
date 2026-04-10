from chromadb import PersistentClient
from uuid import uuid4

client = PersistentClient("chunks")


def clear_db():
    for c in client.list_collections():
        client.delete_collection(c.name)


def delete_method_chunks(method):
    client.delete_collection(method)


def get_or_init_method(method):
    return client.get_or_create_collection(method)


def add_chunks(collection, fn, title, chunks, embeddings, sections):
    ids = [f"{fn}_{str(uuid4())}" for _ in range(len(chunks))]
    mds = [{"title": title, "section": s} for s in sections]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=mds)  # type: ignore
