# English → Hindi Machine Translation (Transformer) — Demo Project

This repository contains a full minimal project to fine-tune a pretrained seq2seq transformer for **English → Hindi** translation, run inference, and demo via a Streamlit app. It is designed for learning, interviews, and quick demos.

## What is included
- `train.py` — fine-tuning script using Hugging Face `transformers` + `datasets`.
- `infer.py`  — simple inference script for translating input sentences.
- `app.py`    — Streamlit demo to translate English -> Hindi (loads a saved model directory).
- `requirements.txt` — Python dependencies.
- `Dockerfile` — to containerize the Streamlit demo (serves the app).
- `questions.md` — curated list of interview questions you may be asked about the project.
- Example data format description (use `data/*.jsonl` with `{"src":"...", "tgt":"..."}` per line).

## Quick start (demo only)
1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. To run the demo Streamlit app locally (after you have a trained model or a released model path):
```bash
streamlit run app.py -- --model_path runs/mt-opus-en-hi
```
By default the app expects a directory `runs/mt-opus-en-hi` with `pytorch_model.bin` and tokenizer files (from `transformers`).

## Train
Put your `data/train.jsonl`, `data/valid.jsonl`, `data/test.jsonl` files and run:
```bash
python train.py
```

## Docker (for demo)
Build:
```bash
docker build -t mt-en-hi-demo:latest .
```
Run:
```bash
docker run -p 8501:8501 mt-en-hi-demo:latest
```

## Notes & tips
- Model used: `Helsinki-NLP/opus-mt-en-hi` — a small, efficient model for EN→HI. For better quality on low-resource pairs consider back-translation or M2M/mbart models.
- Use `fp16=True` for faster training on modern GPUs.
- For evaluation, BLEU is included (sacrebleu). Consider chrF / COMET for better correlation with human judgments.

---


## Added
- `colab_demo.ipynb` — Colab notebook that demonstrates loading the model and inference.
- `app_improved.py` — Improved Streamlit UI with model selection and batch upload.
