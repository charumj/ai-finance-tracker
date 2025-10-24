# train_model_cpu.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch

# Use CPU only
device = torch.device("cpu")
print("Using device:", device)

# Load your brand-category CSV
df = pd.read_csv("brand_category.csv")  # must have columns: brand, category

# Create training text
df['text'] = "Bought in " + df['brand'] + " for 100"

# Encode labels
labels = df['category'].unique().tolist()
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
df['label'] = df['category'].map(label2id)

# Split data
train_df, test_df = train_test_split(df[['text','label']], test_size=0.1, random_state=42)

# Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
).to(device)

# Tokenize function
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

# Training arguments (CPU optimized)
training_args = TrainingArguments(
    output_dir="./expense_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # smaller batch for CPU
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    report_to="none"  # avoid extra logging
)

# Metrics
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save model & tokenizer
trainer.save_model("./expense_model")
tokenizer.save_pretrained("./expense_model")

print("Training complete! Model saved in ./expense_model")
