import json

from src.nlp import generate
from src.vector_store import retrieve_chunks
from src.constants import EVAL_N_CHUNKS


def build_prompt(question, context):
    return f"""
You are answering a question using ONLY the provided context.

Instructions:

Use only the information in the context.
Do not use prior knowledge.
If the answer is not contained in the context, say: "I cannot find the answer in the provided context."
Be concise and factual.

Question:
{question}

Context:
{"\n\n".join([f"[Context {i + 1}]\n{c}" for i, c in enumerate(context)])}

Answer:
"""


def main():
    with open("data/prompts/prompts.json", errors="ignore") as f:
        data = json.load(f)
    ids, questions = map(list, zip(*((p["id"], p["question"]) for p in data)))
    contexts_dict = {
        m: retrieve_chunks(questions, EVAL_N_CHUNKS, m)
        for m in ["fixed_length", "sentence_based", "semantic"]
    }
    responses_dict = {}
    for m, chunks in contexts_dict.items():
        print(f"Method: {m}")
        prompts = list(map(build_prompt, questions, chunks))  # type: ignore
        responses = generate(prompts)
        responses_dict[m] = [{"id": i, "response": r} for i, r in zip(ids, responses)]
    with open("results/llm_responses.json", "w") as f:
        json.dump(responses_dict, f, indent=4)


if __name__ == "__main__":
    main()
