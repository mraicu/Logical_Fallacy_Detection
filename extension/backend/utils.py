from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertModel
from transformers import BertTokenizer
import pandas as pd
import torch.nn as nn
import torch

sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}


class BertWithSentiment(nn.Module):
    def __init__(self, model_name, num_labels, num_sentiment_classes=3):
        super(BertWithSentiment, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, return_dict=True)  # Ensure return_dict=True
        self.sentiment_embedding = nn.Embedding(num_sentiment_classes, 768)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768 * 2, num_labels)

    def forward(self, input_ids, attention_mask, sentiment, labels=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_embedding = bert_output.last_hidden_state[:, 0, :]
        sentiment_embed = self.sentiment_embedding(sentiment)
        combined = torch.cat((cls_token_embedding, sentiment_embed), dim=1)

        logits = self.classifier(self.dropout(combined))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)  # Ensure labels are (batch_size,) with class indices

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


def predict_with_sentiment(tokenizer_path, model_path, model_name, texts, sentiments, logical_fallacies):
    # Load model
    print('here')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get number of labels (you must know your training labels)
    # logical_fallacies = list(set(list(filtered_test_data['logical_fallacies'])))
    label2id = {label: id for id, label in enumerate(logical_fallacies)}
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)

    print('here')

    # Recreate model and load weights
    model = BertWithSentiment(model_name=model_name, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print('here3')

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # Tokenize test data
    def tokenize_function(texts, sentiments, tokenizer):
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        sentiment_ids = torch.tensor([sentiment_mapping[s] for s in sentiments])
        return inputs, sentiment_ids

    # Tokenize
    inputs, sentiment_ids = tokenize_function(texts, sentiments, tokenizer)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    sentiment_ids = sentiment_ids.to(device)

    # Remove token_type_ids
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    # Run inference
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], sentiment=sentiment_ids)
        predictions = torch.argmax(outputs["logits"], dim=1)

    # Convert predictions to labels
    predicted_labels = [id2label[pred.item()] for pred in predictions]

    return predicted_labels


analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    if pd.isna(text):
        print("There is a nan value")
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"
