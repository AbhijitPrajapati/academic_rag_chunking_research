import random
from src.groq import chat_response
import json
from tqdm import tqdm


def build_prompt(question, truth, a, b, c):
    return f"""
You are evaluating answers to a question.

Question:
{question}

Ground Truth Answer:
{truth}

Candidate Answers:
0: {a}
1: {b}
2: {c}

Evaluation Rules:

The most important factor is factual correctness relative to the ground truth.
Second is completeness (does the answer cover the key points).
Third is relevance and clarity.

If an answer says "I cannot find the answer in the provided context", it should be treated as incorrect.
If multiple answers abstain, rank them as the worst answers.

Do NOT prefer longer answers unless they are more correct.
Do NOT assume any answer is correct unless it aligns with the ground truth.

Rank the answers from best to worst.
The output must contain ONLY a sequence of number seperated by spaces. [eg., 1 0 2]
"""


def main():
    with open("data/prompts/prompts.json", errors="ignore") as f:
        data = json.load(f)
    questions = {q["id"]: (q["question"], q["answer"]) for q in data}
    with open("results/llm_responses.json") as f:
        data = json.load(f)
    fixed = {r["id"]: r["response"] for r in data["fixed_length"]}
    sentence_based = {r["id"]: r["response"] for r in data["sentence_based"]}
    semantic = {r["id"]: r["response"] for r in data["semantic"]}
    evaluation_responses = []
    for qid, (q, a) in tqdm(questions.items()):
        responses = [
            ("fixed_length", fixed[qid]),
            ("sentence_based", sentence_based[qid]),
            ("semantic", semantic[qid]),
        ]
        random.shuffle(responses)

        shuffled_responses = [r[1] for r in responses]
        llm_response = chat_response(build_prompt(q, a, *shuffled_responses))
        evaluation_responses.append(
            {
                "id": qid,
                "response": llm_response,
                "shuffle_key": [r[0] for r in responses],
            }
        )
    with open("results/response_evaluation_old.json", "w") as f:
        json.dump(evaluation_responses, f, indent=4)


if __name__ == "__main__":
    main()
