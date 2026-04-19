from src.constants import FIXED_TOKENS, FIXED_OVERLAP
from src.nlp import tokenizer


def fixed_length_chunking(sections):
    chunks = []
    section_names = []
    for section in sections:
        text = section["text"]
        outputs = tokenizer(
            text,
            max_length=FIXED_TOKENS,
            truncation=True,
            stride=FIXED_OVERLAP,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        for i in range(len(outputs["input_ids"])):
            offsets = outputs["offset_mapping"][i]
            start_char = offsets[0][0]
            end_char = offsets[-1][1]
            chunks.append(text[start_char:end_char])
            section_names.append(section["section"])
    return chunks, section_names
