# train_lora_auto_detect.py
import os
import logging
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback
)

# PEFT / LoRA
from peft import LoraConfig, get_peft_model, TaskType

# Try sacrebleu
try:
    import sacrebleu
    _HAS_SACREBLEU = True
except Exception:
    _HAS_SACREBLEU = False

# ---------------- CONFIG ----------------
MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 8
OUTPUT_DIR = "runs/mt-opus-en-hi-lora-auto"
EPOCHS = 5
SEED = 42

# LoRA settings (tweak if needed)
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Logging / eval frequency (tune to your dataset size)
LOGGING_STEPS = 50
EVAL_STEPS = 200
# ----------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # Replace pad token id by -100 for labels
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def detect_lora_target_modules(model, verbose=True):
    """
    Inspect model.named_modules() and collect likely attention projection suffix names.
    Returns a sorted list of unique suffix names (e.g. 'q_proj', 'v_proj', 'q', 'v').
    """
    candidates = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            last = name.split(".")[-1].lower()
            # tokens to look for in the name
            tokens = ("q_proj","k_proj","v_proj","q_lin","k_lin","v_lin","q_proj","q","k","v","query","key","value","proj","out_proj","linear")
            if any(tok in last for tok in tokens):
                candidates.add(last)
    # also try to include longer meaningful names like 'q_proj' found in full name
    if not candidates:
        # try scanning full name tokens (fallback)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                n = name.lower()
                if any(tok in n for tok in ("q_proj","v_proj","k_proj","out_proj","proj")):
                    candidates.add(n.split(".")[-1])
    # Filter too-long names
    candidates = {c for c in candidates if len(c) <= 60}
    detected = sorted(candidates)
    if verbose:
        logger.info("Detected candidate Linear module name suffixes for LoRA target_modules: %s", detected)
    return detected

def compute_metrics_from_preds(preds, labels, tokenizer):
    # preds: generated token ids or tuple
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    refs = [[lbl] for lbl in decoded_labels]

    if _HAS_SACREBLEU:
        bleu = sacrebleu.corpus_bleu(decoded_preds, refs)
        score = bleu.score
    else:
        # fallback: simple tokenized BLEU via nltk if installed (rare)
        try:
            from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
            tokenized_preds = [p.split() for p in decoded_preds]
            tokenized_refs = [[r[0].split()] for r in refs]
            score = nltk_corpus_bleu(tokenized_refs, tokenized_preds) * 100.0
        except Exception:
            score = 0.0
    return {"bleu": score}

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    return compute_metrics_from_preds(preds, labels, compute_metrics.tokenizer)

class EvalPrinterCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {})
        if not metrics:
            return
        step = metrics.get("step", None) or state.global_step
        epoch = metrics.get("epoch", None) or state.epoch
        eval_loss = metrics.get("eval_loss")
        # possible keys: eval_bleu, bleu, bleu_score
        eval_bleu = metrics.get("eval_bleu") or metrics.get("bleu") or metrics.get("bleu_score")
        parts = [f"[Eval] step={step}"]
        if epoch is not None:
            try:
                parts.append(f"epoch={epoch:.2f}")
            except Exception:
                parts.append(f"epoch={epoch}")
        if eval_loss is not None:
            parts.append(f"loss={eval_loss:.4f}")
        if eval_bleu is not None:
            parts.append(f"BLEU={eval_bleu:.2f}")
        print(" | ".join(parts))

def main():
    set_seed(SEED)
    logger.info("Loading dataset...")
    dataset = load_data()

    logger.info("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Auto-detect LoRA target modules
    detected_targets = detect_lora_target_modules(model, verbose=True)
    if not detected_targets:
        # If detection failed, give helpful instructions and try a safe fallback set
        # fallback common suffixes — PEFT will match substrings against module names
        fallback = ["q_proj", "v_proj", "k_proj", "q", "v", "k", "query", "value"]
        # check which fallbacks actually appear in model.named_modules()
        present = set()
        for n, _ in model.named_modules():
            nl = n.lower()
            for tok in fallback:
                if tok in nl:
                    present.add(tok)
        if present:
            detected_targets = sorted(present)
            logger.info("Using fallback-detected targets: %s", detected_targets)
        else:
            # show user how to inspect modules
            msg = (
                "Could not auto-detect target modules for LoRA. Please inspect the model's linear modules to "
                "find attention projection names. Example snippet to run in python REPL:\n\n"
                "from transformers import AutoModelForSeq2SeqLM\n"
                "model = AutoModelForSeq2SeqLM.from_pretrained('<model-name>')\n"
                "for n, m in model.named_modules():\n"
                "    if isinstance(m, torch.nn.Linear):\n"
                "        print(n)\n\n"
                "Then pick suffix tokens like 'q_proj', 'v_proj', 'q', 'v', 'k_proj' and use them as target_modules."
            )
            raise ValueError(msg)

    target_modules_for_lora = detected_targets
    logger.info("Using LoRA target_modules = %s", target_modules_for_lora)

    # Build LoRA config
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules_for_lora,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    logger.info("Wrapping model with PEFT/LoRA...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(lambda ex: preprocess_function(ex, tokenizer), batched=True, remove_columns=dataset["train"].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

    # attach tokenizer to compute_metrics
    compute_metrics.tokenizer = tokenizer

    # Training args with periodic evaluation
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        num_train_epochs=EPOCHS,
        save_total_limit=3,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=True,
        run_name="mt_en_hi_lora_auto"
    )

    # generation_config compatibility fallback
    if not hasattr(training_args, "generation_config"):
        setattr(training_args, "generation_config", None)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EvalPrinterCallback()]
    )

    logger.info("Starting training (LoRA enabled, auto-detected targets)...")
    trainer.train()

    logger.info("Training finished. Saving model and tokenizer...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    try:
        model.save_pretrained(OUTPUT_DIR)
    except Exception:
        pass

    # Final evaluation on validation & test
    logger.info("Final evaluation on validation set (full):")
    val_results = trainer.evaluate(tokenized["validation"])
    logger.info("Validation results: %s", val_results)

    logger.info("Predict & evaluate on test set (full):")
    preds = trainer.predict(tokenized["test"])
    test_metrics = compute_metrics_from_preds(preds.predictions, preds.label_ids, tokenizer)
    logger.info("Test BLEU: %.4f", test_metrics["bleu"])

if __name__ == "__main__":
    main()
