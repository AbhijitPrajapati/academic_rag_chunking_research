from util.vector_store import delete_method_chunks

METHOD_TO_DELETE = "sentence_based"

if __name__ == "__main__":
    delete_method_chunks(METHOD_TO_DELETE)
