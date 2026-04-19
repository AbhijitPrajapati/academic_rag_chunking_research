from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", backend="torch")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
