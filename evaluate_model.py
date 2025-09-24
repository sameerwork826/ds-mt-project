import numpy as np
import sacrebleu
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model & tokenizer
model_dir = "runs/mt-opus-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Load test data
dataset = load_dataset("json", data_files={"test": "data/test.jsonl"})

def preprocess(examples):
    inputs = tokenizer(examples["src"], truncation=True, padding="max_length", max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["tgt"], truncation=True, padding="max_length", max_length=128)
    return inputs, labels

# Run prediction
preds, refs = [], []
for example in dataset["test"]:
    inputs = tokenizer(example["src"], return_tensors="pt", truncation=True, padding=True, max_length=128)
    output = model.generate(**inputs, max_length=128, num_beams=4)
    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    preds.append(pred)
    refs.append([example["tgt"]])   # sacrebleu expects list of references

bleu = sacrebleu.corpus_bleu(preds, refs)
print("Test BLEU:", bleu.score)
