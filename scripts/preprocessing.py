import json
from datasets import load_dataset
from datasets import DatasetDict
import numpy as np
import torch
import torch.nn.functional as F

def get_labels(dataset,eurovoc):
    labels = dataset["train"].features["labels"].feature.names
    descriptions = [eurovoc.get(label, {"en": "Unknown"})["en"] for label in labels]

    id2label = {label: idx for idx, label in enumerate(descriptions)}
    label2id = {idx: label for label, idx in id2label.items()}
    return descriptions, label2id, id2label

def get_dataset():
    with open("../data/eurovoc_descriptors.json") as f:
        eurovoc = json.load(f)

    dataset = load_dataset("coastalcph/multi_eurlex", name="en", split="train")

    temp_split = dataset.train_test_split(test_size=0.2, seed=42)

    train_dataset = temp_split["train"]
    split_data = temp_split["test"].train_test_split(test_size=0.5, seed=42)
    val_dataset = split_data["train"]
    test_dataset = split_data["test"]

    dataset = DatasetDict({
        "train":train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }
    )
    return dataset,eurovoc

def tokenize_function(examples,tokenizer):
    tokenized_input = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    temp = []
    for label in examples["labels"]:
        one_hot = F.one_hot(torch.tensor(label),num_classes=21).float()
        one_hot = one_hot.sum(dim=0)
        temp.append(one_hot)
    one_hot = torch.stack(temp)
    tokenized_input["labels"] = one_hot.tolist()
    return tokenized_input


