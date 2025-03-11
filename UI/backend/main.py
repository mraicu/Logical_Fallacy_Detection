from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizerFast, pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production (don't use "*")
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

FALLACY_TO_COLOR = {
# c0fcee
    'nonfallacy': "none",
    'fallacy':'#fcc0c0',
    'deductive fallacy': "#fcc0c0",
    'false causality': "#dffcc0",
    'faulty generalization': "#cbc0fc",
    'equivocation': "#fcc0da",
    'intentional': "#ffd9d3",
    'ad populum': "#e8c0fc",
    'appeal to emotion': "#e0c7fc",
    'circular reasoning': "#fcf2c0",
    'ad hominem': "#c0ffee",
    'fallacy of credibility': "#d1f8e5",
    'false dilemma': "#fac0fc",
    'fallacy of extension': "#f2fcc0",
    'fallacy of relevance': "#c9fcc0",
    'fallacy of logic': "#c0c6fc",
}

def classify_text(data, classification_type):
    if classification_type.lower() == 'binary':
        model_path = "../../model/outputs/21-02-2025_14-45-55_bert-2-classes-model.pickle"
    elif classification_type.lower() == '3-classes':
        model_path = "../../model/outputs/03-03-2025_16-23-08_bert-3-classes-model.pickle"
    elif classification_type.lower() == '5-classes':
        model_path = "../../model/outputs/03-03-2025_16-46-39_bert-5-classes-model.pickle"
    elif classification_type.lower() == '13-classes':
        model_path = "../../model/outputs/20-02-2025_10-26-38_bert-all-classes-model.pickle"
    else:
        model_path = "../../model/outputs/20-02-2025_10-26-38_bert-all-classes-model.pickle"


    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

    data = [x for x in data.split(".") if x]

    predictions = {}
    for d in data:
        label = nlp(d)[0]["label"]
        color = FALLACY_TO_COLOR.get(label, "#FFFFFF")
        predictions[d] = (label, color)
    return predictions


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


class TextRequest(BaseModel):
    text: str
    classification: str

@app.post("/analyze")
def process_text(request: TextRequest):
    processed_text = classify_text(request.text, request.classification)
    return {"predictions": processed_text}
