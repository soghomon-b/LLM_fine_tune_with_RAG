from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict
from datasets import Dataset, DatasetDict
import torch
import os

# Load tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Avoid padding issues
model = AutoModelForCausalLM.from_pretrained(model_name)

# Simulated dataset
data = {
    "train": [
        {"input": "How do I reset my password?", "output": "Click on 'Forgot Password' and follow the instructions."},
        {"input": "What is your return policy?", "output": "You can return items within 30 days of purchase."}
    ]
}

# Example RAG-style info (can be retrieved dynamically in real usage)
info = "Company Knowledge Base: For account-related issues, follow security protocols."

# Format dataset with injected info
def format_example(example):
    prompt = f"[INFO]: {info}\n[USER]: {example['input']}\n[RESPONSE]: {example['output']}"
    return {"text": prompt}

# Format dataset with injected info
formatted_data = [format_example(x) for x in data["train"]]

# Create Dataset and wrap in DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_dict({"text": [x["text"] for x in formatted_data]})
})


# Tokenize dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset["train"].map(tokenize, batched=True)

# Training setup
training_args = TrainingArguments(
    output_dir="./finetuned-model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()

# save model
os.makedirs("fine_tuned", exist_ok=True)
torch.save(trainer.model.state_dict(), "fine_tuned/model.bin")
print("model saved in fine_tuned/model.bin")
