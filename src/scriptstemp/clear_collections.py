from chromadb import PersistentClient

client = PersistentClient("chunks")

client.delete_collection("fixed_length")
