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

FALLACY_PROPS = {
    # c0fcee
    'nonfallacy': ["none", "Acesta este un argument valid."],
    'deductive fallacy': ["#fcc0c0", "Structură logică invalidă."],
    'false causality': ["#dffcc0", "Asocierea eronată între cauză și efect."],
    'faulty generalization': ["#cbc0fc", "Bazată pe dovezi insuficiente sau părtinitoare."],
    'equivocation': ["#fcc0da", "Folosirea ambiguă a unui termen în mai multe sensuri."],
    'intentional': ["#ffd9d3", "Folosirea deliberată a raționamentului greșit pentru a induce în eroare."],
    'ad populum': ["#e8c0fc", "Argumentul se bazează pe opinia majorității."],
    'appeal to emotion': ["#e0c7fc", "Folosirea emoțiilor pentru a convinge în locul argumentelor raționale."],
    'circular reasoning': ["#fcf2c0", "Concluzia este presupusă în premisă."],
    'ad hominem': ["#c0ffee", "Atac la persoană în loc de argument."],
    'fallacy of credibility': ["#d1f8e5", "Invocarea unei surse necalificate."],
    'false dilemma': ["#fac0fc", "Prezentarea a doar două opțiuni când există mai multe."],
    'fallacy of extension': ["#f2fcc0", "Exagerarea poziției oponentului pentru a o respinge mai ușor."],
    'fallacy of relevance': ["#c9fcc0", "Folosirea unor informații irelevante pentru a distrage atenția."],
    'fallacy of logic': ["#c0c6fc", "Termen general pentru orice raționament incorect."],
}


def classify_text(data, classification_type):
    if classification_type.lower() == 'binary':
        model_path = "../../model/outputs/21-02-2025_14-45-55_bert-2-classes-model.pickle"
    elif classification_type.lower() == '3-classes':
        model_path = "../../model/outputs/03-03-2025_16-23-08_bert-3-classes-model.pickle"
    elif classification_type.lower() == '5-classes':
        model_path = "../../model/outputs/03-03-2025_16-46-39_bert-5-classes-model.pickle"
    elif classification_type.lower() == '15-classes':
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
        props = FALLACY_PROPS.get(label, ["#FFFFFF", ""])
        predictions[d] = (label, props)
    return predictions


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


class TextRequest(BaseModel):
    selectedText: str
    classification: str


@app.post("/analyze")
def process_text(request: TextRequest):
    processed_text = classify_text(request.selectedText, request.classification)
    return {"predictions": processed_text}
