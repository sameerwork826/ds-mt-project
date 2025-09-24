import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate(texts, model_path="runs/mt-opus-en-hi", max_length=128, num_beams=4):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="runs/mt-opus-en-hi")
    parser.add_argument("--text", type=str, required=True, help="Text or multiple sentences separated by ||")
    args = parser.parse_args()
    texts = args.text.split("||")
    translations = translate(texts, model_path=args.model_path)
    for src, tgt in zip(texts, translations):
        print("SRC:", src)
        print("TRANSLATION:", tgt)
        print("---")
