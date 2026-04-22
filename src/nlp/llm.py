from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
import torch

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", quantization_config=nf4_config
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
llm_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
)


def generate(prompts):
    output = llm_pipeline(
        [[{"role": "user", "content": p}] for p in prompts],
        return_full_text=False,
        batch_size=2,
    )
    return [o[0]["generated_text"] for o in output]
