# GPT-2 Fine-Tuning with Injected Knowledge (RAG-style Prompting)

This repository demonstrates how to fine-tune a language model using a simple **retrieval-augmented generation (RAG)-style format**, where each training sample includes additional context (e.g., from a knowledge base). This setup is ideal for domain-specific assistants like customer service bots, internal tools, or support agents.

---

## Overview

We inject external information into the prompt under an `[INFO]` tag, and train the model to respond to user queries accordingly:

**Training format:**
```
[INFO]: {external knowledge}
[USER]: {user input}
[RESPONSE]: {expected response}
```

---

## Project Structure

```bash
.
├── fine_tune.py         # Main training script (your current code)
├── requirements.txt # Dependencies (see below)
├── README.md        # You're here!
└── LICENSE       # MIT License
```

---

## How to Run

1. **Install dependencies**:

```bash
pip install transformers datasets torch
```

2. **Run the script**:

```bash
python train.py
```

The model will be fine-tuned and saved to `./finetuned-model`.

---

## What's Inside

- **Model**: GPT-2 (`transformers` by Hugging Face)
- **Tokenizer**: Uses GPT-2 tokenizer with EOS token as padding
- **Data**: Simulated training set of input-output QA pairs
- **RAG-style augmentation**: Each sample includes injected context from an imaginary knowledge base
- **Trainer**: Hugging Face `Trainer` API for easy training management
- **GPU support**: Automatically enables FP16 if CUDA is available

---

## Example Input After Formatting

```text
[INFO]: Company Knowledge Base: For account-related issues, follow security protocols.
[USER]: How do I reset my password?
[RESPONSE]: Click on 'Forgot Password' and follow the instructions.
```

---

## Notes for Customization

- Replace the `data` dictionary with your own dataset (CSV, JSON, etc.)
- Dynamically fetch or inject context from a knowledge base for real-world use
- Experiment with larger models (e.g. `gpt2-medium`) or LoRA-style fine-tuning for efficiency

---

## Output

Trained models are saved to:

```
./finetuned-model/
```

You can load them later with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./finetuned-model")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

---

## Future Ideas

- Use [PEFT/LoRA](https://github.com/huggingface/peft) for lightweight training
- Add validation + evaluation metrics
- Stream context dynamically (true RAG)
- Use your own tokenizer for domain-specific language

---

## License

MIT — free to use, modify, and redistribute.

---

## Questions?

Feel free to open an issue or contribute!
