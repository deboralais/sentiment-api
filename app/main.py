from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    GPT2Tokenizer,
    GPT2ForSequenceClassification
)
import torch

app = FastAPI()

# -------------------------------
# Caminhos dos modelos
BERT_PATH = "app/model/modelo_bert.pth"
GPT2_PATH = "app/model/modelo_gpt2.pth"

# -------------------------------
# Load BERT
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
bert_model.load_state_dict(torch.load(BERT_PATH, map_location=torch.device("cpu")))
bert_model.eval()

# -------------------------------
# Load GPT-2
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  # evita erro de padding
gpt2_model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
gpt2_model.load_state_dict(torch.load(GPT2_PATH, map_location=torch.device("cpu")))
gpt2_model.eval()

# -------------------------------
# Pydantic Schema
class InputText(BaseModel):
    text: str

# -------------------------------
# Função para inferência comum
def predict_sentiment(model, tokenizer, text, is_gpt2=False):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    if is_gpt2:
        inputs["attention_mask"] = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        score = probs[0][pred_class].item()

    sentiment = "positivo" if pred_class == 1 else "negativo"
    return {"sentiment": sentiment, "score": round(score, 4)}

# -------------------------------
# Endpoints
@app.post("/analyze")
def analyze_bert(input_text: InputText):
    return predict_sentiment(bert_model, bert_tokenizer, input_text.text)

@app.post("/analyze-gpt2")
def analyze_gpt2(input_text: InputText):
    return predict_sentiment(gpt2_model, gpt2_tokenizer, input_text.text, is_gpt2=True)
