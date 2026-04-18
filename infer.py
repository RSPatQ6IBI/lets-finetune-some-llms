# infer.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import cfg


def load_finetuned():
    tokenizer = AutoTokenizer.from_pretrained(cfg.output_dir, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # load LoRA adapter on top of base
    model = PeftModel.from_pretrained(base_model, cfg.output_dir)
    model.eval()
    return model, tokenizer


def solve(question: str, model, tokenizer) -> str:
    prompt = f"""<|im_start|>system
You are a helpful math assistant that solves word problems step by step.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,        # low temp = more deterministic for math
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    print("Loading fine-tuned model...")
    model, tokenizer = load_finetuned()

    questions = [
        "If John has 5 apples and gives 2 to Mary, how many does he have left?",
        "A train travels 60 miles per hour. How far will it travel in 2.5 hours?",
        "A store sells notebooks for $3 each. If Lisa buys 4 notebooks and pays with a $20 bill, how much change does she get?",
    ]

    for q in questions:
        print(f"\nQuestion : {q}")
        print(f"Answer   : {solve(q, model, tokenizer)}")
        print("-" * 60)