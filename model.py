# model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from utils import config


def load_model_and_tokenizer():
    print(f"Loading model: {config.cfg.model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.cfg.model_id,
        trust_remote_code=True,
    )
    # Qwen uses a specific pad token
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"         # avoid warnings with causal LM

    model = AutoModelForCausalLM.from_pretrained(
        config.cfg.model_id,
        torch_dtype=torch.float32,           # float32 required for CPU
        trust_remote_code=True,
    )

    # print baseline param count
    total = sum(p.numel() for p in model.parameters())
    print(f"Base model parameters: {total / 1e6:.1f}M")

    return model, tokenizer


def apply_lora(model):
    lora_config = LoraConfig(
        r=config.cfg.lora_r,
        lora_alpha=config.cfg.lora_alpha,
        lora_dropout=config.cfg.lora_dropout,
        target_modules=config.cfg.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # e.g. trainable params: 1,179,648 || all params: 494,032,896 || trainable%: 0.24

    return model