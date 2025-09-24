# Interview Questions & Answers — English→Hindi MT Project

Here are **20 commonly asked questions** (with answers) you can expect in interviews about this project.

---

### 1. Why did you choose `Helsinki-NLP/opus-mt-en-hi`?
It is a lightweight, pretrained EN→HI translation model from Hugging Face that works well out-of-the-box for low-resource languages. Using a pretrained model reduces training cost and time compared to building from scratch.

### 2. Why not train from scratch?
Training from scratch requires millions of parallel sentence pairs and high compute. Fine-tuning a pretrained model is more efficient and effective in practice.

### 3. Why do we replace padding tokens with `-100` in labels?
`CrossEntropyLoss` ignores `-100` during training. This ensures that padding tokens do not contribute to the loss.

### 4. What does `predict_with_generate=True` do?
It tells the Trainer to use the model’s `generate()` function for evaluation instead of teacher-forced predictions, so metrics like BLEU evaluate real decoded translations.

### 5. Why use BLEU? What are its limitations?
BLEU measures n-gram overlap with references. It is widely used but fails to capture meaning variations and synonyms. Alternatives include chrF (character n-gram), COMET, or BLEURT.

### 6. How does beam search improve translation quality?
Beam search explores multiple candidate translations simultaneously, picking the highest-scoring sequence. It balances diversity and accuracy, but is slower than greedy decoding.

### 7. What is the purpose of `no_repeat_ngram_size`?
It prevents the model from repeating the same n-grams in output, reducing issues like “I am I am I am”.

### 8. How did you structure the dataset?
Each line is JSON with `{"src": "English sentence", "tgt": "Hindi sentence"}`. Train/valid/test splits allow proper evaluation.

### 9. How do you handle script differences (Latin vs Devanagari)?
Tokenizers (SentencePiece for OPUS models) already support both scripts. We just ensure normalization and avoid transliteration.

### 10. What data augmentation could you use?
- Back-translation: translate Hindi monolingual data into English and add to training set.  
- Noise injection: slight word swaps or deletions.  
- Paraphrasing: augment source sentences with paraphrases.

### 11. What are key hyperparameters?
- Learning rate (e.g., 3e-5 to 5e-4)  
- Batch size (GPU memory dependent)  
- Max source/target length  
- Beam size during inference

### 12. How to prevent overfitting?
- Early stopping on validation BLEU  
- Label smoothing  
- More training data  
- Dropout or smaller model

### 13. How would you deploy this model?
Options include:
- Streamlit app (for demos)  
- FastAPI/Flask REST service for production  
- TorchScript/ONNX for optimized inference  
- Containerization via Docker

### 14. How to optimize for CPU inference?
Use quantization (e.g., `torch.quantization` or `onnxruntime`), distillation to smaller models, or pruning.

### 15. What are common errors in MT?
- Word order mistakes  
- Missing words  
- Literal translations losing idiomatic meaning  
- Gender/politeness issues in Hindi

### 16. How would you evaluate beyond BLEU?
- Human evaluation for fluency/adequacy  
- COMET or BLEURT for semantic similarity  
- chrF for character-level evaluation (good for morphologically rich languages like Hindi)

### 17. Why use Seq2Seq instead of a plain encoder or decoder?
Machine translation requires mapping a full source sequence into a target sequence. Seq2Seq with attention or transformers handles alignment between input and output tokens.

### 18. Why use Hugging Face Trainer instead of raw PyTorch?
Trainer abstracts training loops, handles distributed training, evaluation, and checkpointing. It reduces boilerplate and ensures reproducibility.

### 19. How would you scale this to multiple languages?
Use multilingual models like mBART or M2M100 that support many-to-many translation. Fine-tune them on English↔Hindi for better performance in low-resource scenarios.

### 20. What ethical concerns exist in deploying MT systems?
- Incorrect translations in sensitive domains (medical/legal) can cause harm  
- Cultural/gender bias in translations  
- Over-reliance on automated translations without human verification

---

**Tip for interviews**: Always back your answers with practical examples (e.g., “Validation BLEU improved from 17 to 24 after back-translation”).
