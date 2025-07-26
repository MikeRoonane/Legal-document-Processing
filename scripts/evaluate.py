import torch
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from train import get_trainer
from model import compute_metrics
from training import get_dataCollator, get_tokenized_dataset
from preprocessing import get_dataset
save_path = "../models"

model = AutoModelForSequenceClassification.from_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(save_path, use_fast=True)

device = torch.device("cuda" if torch.cuda.is_available() else "ckcpu")
model.to(device)
dataset,eurovoc = get_dataset()
tokenized_dataset = get_tokenized_dataset(dataset,tokenizer)
data_collator = get_dataCollator(tokenizer)

trainer = get_trainer(
    model,
    tokenized_dataset["train"],
    tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"], metric_key_prefix="eval")
print("Evaluation complete.")
print("Metrics:")
print(metrics)
print("Model evaluation finished successfully.")
