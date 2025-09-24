import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import time
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

st.set_page_config(page_title="EN → HI Translator (Improved)", layout="wide")

st.title("English → Hindi Translator — Improved Demo")
st.markdown("""
An improved demo with model selection (local folder or Hugging Face Hub), file upload for batch translation, and history.
""")

col1, col2 = st.columns([2,1])

with col2:
    st.write("**Options**")
    model_source = st.radio("Model source", ("Local folder", "Hugging Face Hub"))
    if model_source == "Local folder":
        model_path = st.text_input("Local model path", value="runs/mt-opus-en-hi")
    else:
        model_path = st.text_input("HF model name", value="Helsinki-NLP/opus-mt-en-hi")
    num_beams = st.slider("Beams", 1, 8, 4)
    max_length = st.slider("Max generated length", 20, 200, 100)
    use_gpu = st.checkbox("Force GPU if available", value=True)

with col1:
    st.write("**Translate single sentence or multiple (separate by ||)**")
    input_text = st.text_area("Enter English text", value="How are you?||I love machine learning.")
    uploaded_file = st.file_uploader("Or upload a .txt file (one sentence per line) for batch translation", type=["txt"])
    translate_button = st.button("Load model & Translate")

# session state for model and history
if "mt_model" not in st.session_state:
    st.session_state["mt_model"] = None
if "mt_tokenizer" not in st.session_state:
    st.session_state["mt_tokenizer"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []

def load_model(mp):
    # load model and tokenizer into session state
    try:
        tokenizer = AutoTokenizer.from_pretrained(mp, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(mp)
        device = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None

def generate(texts, tokenizer, model, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
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

if translate_button:
    with st.spinner("Loading model..."):
        tokenizer, model, device = load_model(model_path)
        if tokenizer is None:
            st.stop()
        st.session_state["mt_model"] = model
        st.session_state["mt_tokenizer"] = tokenizer
        st.success("Model loaded.")

    # prepare texts
    texts = []
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8").strip().splitlines()
        texts = [line for line in content if line.strip()]
    else:
        texts = [t.strip() for t in input_text.split("||") if t.strip()]

    with st.spinner("Translating..."):
        start = time.time()
        translations = generate(texts, st.session_state["mt_tokenizer"], st.session_state["mt_model"], device)
        duration = time.time() - start

    # show results side-by-side
    for src, tgt in zip(texts, translations):
        st.markdown("**EN:** " + src)
        st.markdown("**HI:** " + tgt)
        st.markdown("---")
        st.session_state["history"].append({"src": src, "tgt": tgt})

    st.info(f"Translated {len(texts)} sentences in {duration:.2f}s")

# show history
if st.session_state["history"]:
    st.subheader("Translation history (last 50)")
    for item in st.session_state["history"][-50:][::-1]:
        st.write("EN:", item["src"])
        st.write("HI:", item["tgt"])
        st.markdown("---")

# batch download of history as txt
if st.session_state["history"]:
    joined = "\n".join([f"{h['src']}\t{h['tgt']}" for h in st.session_state["history"]])
    st.download_button("Download history (TSV)", data=joined, file_name="translations_history.tsv", mime="text/tab-separated-values")
