from chromadb import PersistentClient
import numpy as np

client = PersistentClient("chunks")

fixed = client.get_collection("fixed_length").get()
sentence_based = client.get_collection("sentence_based").get()
semantic = client.get_collection("semantic").get()

fl = [len(d) for d in fixed["documents"]]  # type: ignore
senl = [len(d) for d in sentence_based["documents"]]  # type: ignore
seml = [len(d) for d in semantic["documents"]]  # type: ignore

print(np.mean(fl), np.std(fl))
print(np.mean(senl), np.std(senl))
print(np.mean(seml), np.std(seml))
