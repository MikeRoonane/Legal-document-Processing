from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel
import torch
import torch.nn.functional as F
app = FastAPI()
model_path = "/models"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
class TextInput(BaseModel):
    text: str
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the Legal Text Classification API!"}

@app.post("/predict")
def predict(data: TextInput):
    input_text = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**input_text)
        logits = outputs.logits
        prob = F.softmax(logits,dim=1)
        predictions = torch.argmax(prob, dim=1).tolist()
        probabilities = prob.tolist()[0]
    return {
        "predictions": predictions,
        "probabilities": probabilities
    }
    
    