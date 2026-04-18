from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from utils.config import cfg
from datasets import load_dataset
import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from utils.load_data_ import get_data_columns_ 

def format_prompt(sample):
    """
    Orca-Math columns: 'question', 'answer'
    Format into a chat-style instruction prompt.
    """
    return {
        "text": f"""<|im_start|>system
You are a helpful math assistant that solves word problems step by step.<|im_end|>
<|im_start|>user
{sample['question']}<|im_end|>
<|im_start|>assistant
{sample['answer']}<|im_end|>"""
    }


def load_tokenized_data(tokenizer, dataset):
    print("Loading dataset...")
    # dataset = load_dataset(cfg.dataset_name, split="train")

    # # subset for CPU — remove this line to train on full data
    dataset = dataset.select(range(cfg.max_samples))

    # # train / eval split
    dataset = dataset.train_test_split(test_size=0.05, seed=cfg.seed)

    # apply prompt formatting
    dataset = dataset.map(format_prompt, remove_columns=dataset["train"].column_names)

    def tokenize(sample):
        tokens = tokenizer(
            sample["text"],
            truncation=True,
            max_length=cfg.max_seq_len,
            padding="max_length",
        )
        # labels = input_ids (causal LM — predict next token)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    print("Tokenizing...")
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    print(f"Train samples : {len(tokenized['train'])}")
    print(f"Eval  samples : {len(tokenized['test'])}")
    return tokenized["train"], tokenized["test"]



tokenizer = AutoTokenizer.from_pretrained(
    cfg.model_id,
    trust_remote_code=True,
)

# data_ques_, data_ans_, dataset = get_data_columns_()
# token_train, token_val_ = load_tokenized_data(tokenizer, dataset)



'''
class CustomDataset(Dataset):
    def __init__(self):
        from load_data_ import get_data_columns_, split_numpy_or_dataframe
        data_ques_, data_ans_ = get_data_columns_()
        res_ = split_numpy_or_dataframe(X=data_ques_, y=data_ans_, stratify=False, verbose=True)
        ques_data_ = res_['X_train']
        ans_data_ = res_['y_train']
        self.ques_data_    = np.array(ques_data_)
        self.ans_data_    = np.array(ans_data_)

    def __len__(self):
        """Total number of samples."""
        return len(self.ques_data_)

    def __getitem__(self, idx):
        """Return one sample and its label."""
        sample = self.ques_data_[idx]
        label  = self.ans_data_[idx]
        return sample, label

'''
