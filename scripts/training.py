from preprocessing import get_dataset, tokenize_function, get_labels
from model import load_tokenizer, load_model, compute_metrics
from train import get_trainer, get_training_args,get_lastest_checkpoint 
from transformers import DataCollatorWithPadding
import torch
import numpy as np
import os

class FloatLabelsDataCollator(DataCollatorWithPadding):
        def __call__(self, features):
            batch = super().__call__(features)
            batch["labels"] = batch["labels"].float()
            return batch

def get_tokenized_dataset(dataset,tokenizer):

    print("Dataset loaded successfully.")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer=tokenizer),
        load_from_cache_file=True,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names
        )
    tokenized_dataset.set_format("torch")

    return tokenized_dataset

def get_dataCollator(tokenizer):
    data_collator = FloatLabelsDataCollator(tokenizer = tokenizer)
    return data_collator
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    checkpoint = "nlpaueb/legal-bert-base-uncased"
    tokenizer = load_tokenizer(checkpoint)
    data_collator = get_dataCollator(tokenizer)
    dataset, eurovoc = get_dataset()
    descriptions, id2label, label2id = get_labels(dataset, eurovoc)
    print("id2label:", id2label)
    print("label2id:", label2id)
    tokenized_dataset = get_tokenized_dataset(dataset,tokenizer)
    num_labels = len(descriptions) 
    model = load_model(checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id)
    model.to(device)
    print(f"Model config problem_type: {model.config.problem_type}") 
    print(f"Model output head layers: {model.classifier}") 

    trainer = get_trainer(model, tokenized_dataset["train"], tokenized_dataset["validation"], compute_metrics,data_collator)

    last_checkpoint = get_lastest_checkpoint()
    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model("../models")

if __name__ == "__main__":
    main()