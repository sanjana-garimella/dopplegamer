"""LoRA fine-tuning entrypoint for the SFT agent.

GPU recommended. Run:

    python -m agents.sft.train \\
        --model meta-llama/Llama-3.2-1B \\
        --data data/game_data.db \\
        --output checkpoints/sft_best \\
        --epochs 3 --lora-rank 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.sft.data import iter_examples
from agents.sft.model import SFTConfig, attach_lora, load_model_and_tokenizer


def build_dataset(db_path: str | Path, val_split: float = 0.1):
    from datasets import Dataset

    examples = list(iter_examples(db_path))
    if not examples:
        raise RuntimeError(f"no SFT examples found in {db_path}")
    
    # Shuffle for randomness
    import random
    random.seed(42)
    random.shuffle(examples)
    
    # Split into train/val
    n_val = int(len(examples) * val_split)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]
    
    train_ds = Dataset.from_dict(
        {
            "prompt": [e.prompt for e in train_examples],
            "completion": [e.completion for e in train_examples],
        }
    )
    val_ds = Dataset.from_dict(
        {
            "prompt": [e.prompt for e in val_examples],
            "completion": [e.completion for e in val_examples],
        }
    )
    return train_ds, val_ds


def tokenize_fn(tokenizer, max_length: int = 512):
    def fn(batch):
        full = [p + " " + c for p, c in zip(batch["prompt"], batch["completion"])]
        out = tokenizer(full, truncation=True, max_length=max_length, padding="max_length")
        # Mask prompt tokens with -100 so loss is only computed on completion tokens.
        labels = []
        for i, (prompt, input_ids) in enumerate(zip(batch["prompt"], out["input_ids"])):
            prompt_ids = tokenizer(
                prompt + " ",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )["input_ids"]
            prompt_len = len(prompt_ids)
            masked = [-100] * prompt_len + list(input_ids[prompt_len:])
            # Align length to max_length (padding tokens are also -100)
            pad_id = tokenizer.pad_token_id
            masked = [
                -100 if tid == pad_id else lbl
                for tid, lbl in zip(input_ids, masked + [-100] * (len(input_ids) - len(masked)))
            ]
            labels.append(masked[:len(input_ids)])
        out["labels"] = labels
        return out

    return fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--data", default="data/game_data.db")
    parser.add_argument("--output", default="checkpoints/sft_best")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--quantize-4bit", action="store_true")
    args = parser.parse_args()

    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, EarlyStoppingCallback

    cfg = SFTConfig(model_name=args.model, lora_rank=args.lora_rank, quantize_4bit=args.quantize_4bit)
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = attach_lora(model, cfg)

    train_ds, val_ds = build_dataset(args.data, val_split=args.val_split)
    train_ds = train_ds.map(tokenize_fn(tokenizer), batched=True, remove_columns=["prompt", "completion"])
    val_ds = val_ds.map(tokenize_fn(tokenizer), batched=True, remove_columns=["prompt", "completion"])

    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
