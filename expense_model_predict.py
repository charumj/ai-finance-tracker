# save this as expense_model_predict.py
import zipfile
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Step 1: Unzip the trained model
with zipfile.ZipFile("expense_model.zip", 'r') as zip_ref:
    zip_ref.extractall("expense_model")

# Step 2: Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("expense_model")
model = AutoModelForSequenceClassification.from_pretrained("expense_model")
model.eval()  # set to evaluation mode

# Step 3: Function to predict category
def predict_category(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = int(torch.argmax(outputs.logits, axis=-1))
    return model.config.id2label[pred_id]

# Step 4: Load your CSV
df = pd.read_csv("brand_category.csv")  # replace with your CSV path
df['text'] = "Bought in " + df['Brand'].astype(str) + " for 100"  # create input text

# Step 5: Generate predictions
df['Predicted_Category'] = df['text'].apply(predict_category)

# Step 6: Save predictions
df.to_csv("brand_category_predicted.csv", index=False)
print("✅ Predictions saved to brand_category_predicted.csv")
