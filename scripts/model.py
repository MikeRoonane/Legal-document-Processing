from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import EvalPrediction
import torch

def load_tokenizer(checkpoint: str):
    return AutoTokenizer.from_pretrained(checkpoint,use_fast=True)

def load_model(checkpoint: str, num_labels: int, id2label: dict, label2id: dict):
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        problem_type="multi_label_classification",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

def multi_label_metriccs(predictions,labels,threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.tensor(predictions))
    preds = (probs > threshold).float().numpy()
    labels = labels.float().numpy() if isinstance(labels, torch.Tensor) else labels
    f1 = f1_score(labels, preds, average="micro")
    precision = precision_score(labels, preds, average="micro")
    recall = recall_score(labels, preds, average="micro")
    return {"eval_f1": f1, "eval_precision": precision, "eval_recall": recall}

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    return multi_label_metriccs(preds,labels)
