from chromadb import PersistentClient

client = PersistentClient("chunks")
print(client.list_collections())

collection = client.get_collection("semantic")

print(collection.get(include=["documents"])["documents"][1:4])  # type: ignore
