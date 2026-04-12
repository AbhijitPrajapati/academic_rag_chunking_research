from chromadb import Collection
from embedding import embedding_model


def retrieve_chunks(
    prompts, n_chunks, collection: Collection, similarity_threshold=None
):
    where = (
        None
        if similarity_threshold is None
        else {"similarity_threshold": similarity_threshold}
    )
    return collection.query(
        query_embeddings=embedding_model.encode(prompts),
        where=where,
        n_results=n_chunks,
    )["documents"]
