import random


def build_prompt(question, truth, a, b, c):
    a, b, c = random.sample((a, b, c), 3)
    return f"""
You are evaluating answers to a question.

Question:
{question}

Ground Truth Answer:
{truth}

Candidate Answers:
A: {a}
B: {b}
C: {c}

Evaluation Rules:

The most important factor is factual correctness relative to the ground truth.
Second is completeness (does the answer cover the key points).
Third is relevance and clarity.

If an answer says "I cannot find the answer in the provided context", it should be treated as incorrect.
If multiple answers abstain, rank them as the worst answers.

Do NOT prefer longer answers unless they are more correct.
Do NOT assume any answer is correct unless it aligns with the ground truth.

Rank the answers from best to worst.
The output msut contain ONLY a sequence of letters seperated by spaces. [eg., B A C]
"""


