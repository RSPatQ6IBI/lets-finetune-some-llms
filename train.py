# train.py
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from model import load_model_and_tokenizer, apply_lora
# from data import load_data
# from utils import load_data
from utils import config
from utils import the_custom_dataset_ob_ as csob
from utils.load_data_ import get_data_columns_

def main():
    # ── 1. Load model & tokenizer ──────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer()

    # ── 2. Wrap with LoRA ──────────────────────────────────────────────
    model = apply_lora(model)

    # ── 3. Load & tokenize data ────────────────────────────────────────

    data_ques_, data_ans_, dataset = get_data_columns_()
    train_data, eval_data = csob.load_tokenized_data(tokenizer, dataset)
    # train_data, eval_data = csob.load_data(tokenizer)

    # ── 4. Data collator ───────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # ── 5. Training arguments ──────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=config.cfg.output_dir,
        num_train_epochs=config.cfg.num_epochs,
        per_device_train_batch_size=config.cfg.batch_size,
        per_device_eval_batch_size=config.cfg.batch_size,
        gradient_accumulation_steps=config.cfg.grad_accum_steps,
        learning_rate=config.cfg.learning_rate,
        weight_decay=config.cfg.weight_decay,
        warmup_ratio=config.cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=config.cfg.logging_steps,
        save_steps=config.cfg.save_steps,
        eval_strategy="steps",
        eval_steps=config.cfg.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",               # set "tensorboard" if you want logs
        # no_cuda=True,                   # force CPU
        seed=config.cfg.seed,
        fp16=False,                     # CPU doesn't support fp16
        bf16=False,
        dataloader_num_workers=0,       # safer on CPU
    )

    # ── 6. Trainer ─────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        # tokenizer=tokenizer,
        data_collator=collator,
    )

    # ── 7. Train ───────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── 8. Save final adapter weights ─────────────────────────────────
    print(f"\nSaving model to {config.cfg.output_dir}")
    trainer.save_model(config.cfg.output_dir)
    tokenizer.save_pretrained(config.cfg.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()