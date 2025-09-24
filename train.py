import os
import numpy as np
import torch
from datasets import load_dataset
import sacrebleu
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# -------- CONFIG --------
MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 8
OUTPUT_DIR = "runs/mt-opus-en-hi"
EPOCHS = 3
# ------------------------

def load_data(data_dir="data"):
    data_files = {
        "train": os.path.join(data_dir, "train.jsonl"),
        "validation": os.path.join(data_dir, "valid.jsonl"),
        "test": os.path.join(data_dir, "test.jsonl"),
    }
    ds = load_dataset("json", data_files=data_files)
    return ds

def preprocess_function(examples, tokenizer):
    inputs = examples["src"]
    targets = examples["tgt"]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="longest")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="longest")
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    return model_inputs



def main():
    print("Checking GPU availability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = load_data()

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [[lbl] for lbl in decoded_labels]
        bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)
        return {"bleu": bleu.score}

    print("Tokenizing...")
    tokenized = dataset.map(
        lambda ex: preprocess_function(ex, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        num_train_epochs=EPOCHS,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()
    print("Training finished. Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()