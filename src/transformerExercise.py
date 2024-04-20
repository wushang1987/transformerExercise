from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "bert-base-uncased"  # Example model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print(model)