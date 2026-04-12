from chromadb import PersistentClient
from uuid import uuid4
import json
from tqdm import tqdm
from src.chunking import (
    fixed_length_chunking,
    semantic_chunking,
    sentence_based_chunking,
)
from src.embedding import embedding_model
from .constants import SIMILARITY_THRESHOLDS


client = PersistentClient("chunks")


def load_papers():
    for i in range(10):
        fn = f"p{i}"
        with open(f"data/processed/{fn}.json") as f:
            data = json.load(f)
        yield data, fn


def store_chunks(collection, fn, title, chunks, section_names, *thresh):
    embeddings = embedding_model.encode(chunks)
    ids = [f"{fn}_{str(uuid4())}" for _ in range(len(chunks))]
    mds = (
        [
            {"title": title, "section": s, "similarity_threshold": thresh[0]}
            for s in section_names
        ]
        if len(thresh) != 0
        else [{"title": title, "section": s} for s in section_names]
    )
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=mds,  # type: ignore
    )


def main():
    fixed = client.get_or_create_collection("fixed_length")
    sentence_based = client.get_or_create_collection("sentence_based")
    semantic = client.get_or_create_collection("semantic")
    for data, fn in tqdm(load_papers()):
        store_chunks(fixed, fn, data["title"], *fixed_length_chunking(data["content"]))
        store_chunks(
            sentence_based, fn, data["title"], *sentence_based_chunking(data["content"])
        )
        for thresh in SIMILARITY_THRESHOLDS:
            store_chunks(
                semantic,
                fn,
                data["title"],
                *semantic_chunking(data["content"], thresh),
                thresh,
            )


if __name__ == "__main__":
    main()
