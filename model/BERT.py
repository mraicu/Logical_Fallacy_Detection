import torch
import wandb
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertModel
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


class DataLoader(Dataset):
    """
    Custom Dataset class for handling tokenized text data and optional labels and sentiments.
    Inherits from torch.utils.data.Dataset.
    """

    def __init__(self, encodings, labels=None, sentiments=None):
        """
        Initializes the DataLoader class with encodings, and optionally labels and sentiments.

        Args:
            encodings (dict): A dictionary containing tokenized input text data
                              (e.g., 'input_ids', 'token_type_ids', 'attention_mask').
            labels (list, optional): A list of integer labels for the input text data.
            sentiments (list, optional): A list of sentiments corresponding to the input data.
        """
        self.encodings = encodings
        self.labels = labels
        self.sentiments = sentiments

    def __getitem__(self, idx):
        """
        Returns a dictionary containing tokenized data and optional label and sentiment for a given index.

        Args:
            idx (int): The index of the data item to retrieve.

        Returns:
            item (dict): A dictionary containing the tokenized data and the corresponding label/sentiment.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.sentiments is not None:
            item['sentiment'] = torch.tensor(self.sentiments[idx])

        return item

    def __len__(self):
        """
        Returns the number of data items in the dataset.

        Returns:
            (int): The number of data items in the dataset.
        """
        return len(next(iter(self.encodings.values())))


class BertWithSentiment(nn.Module):
    """
    A PyTorch neural network model that combines BERT representations with sentiment embeddings
    for enhanced text classification.

    This model integrates contextual embeddings from a pre-trained BERT model with additional
    sentiment information, represented as an embedding vector, to improve performance on tasks
    where sentiment is an important feature (e.g., argument classification, fallacy detection).

    Attributes:
        bert (BertModel): Pre-trained BERT model from Hugging Face Transformers.
        sentiment_embedding (nn.Embedding): Embedding layer to represent sentiment categories.
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (nn.Linear): Linear layer for classification using the combined representation.
    """

    def __init__(self, model_name, num_labels, num_sentiment_classes=3):
        """
        Initializes the BertWithSentiment model.

        Args:
            model_name (str): Name or path of the pre-trained BERT model.
            num_labels (int): Number of target classes for classification.
            num_sentiment_classes (int, optional): Number of distinct sentiment classes.
                Defaults to 3 (e.g., negative, neutral, positive).
        """
        super(BertWithSentiment, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, return_dict=True)  # Ensure return_dict=True
        self.sentiment_embedding = nn.Embedding(num_sentiment_classes, 768)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768 * 2, num_labels)

    def forward(self, input_ids, attention_mask, sentiment, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs (batch_size, sequence_length).
            attention_mask (torch.Tensor): Attention mask tensor (batch_size, sequence_length).
            sentiment (torch.Tensor): Tensor of sentiment class indices (batch_size,).
            labels (torch.Tensor, optional): True class labels (batch_size,). Used for loss computation.

        Returns:
            dict: A dictionary containing:
                - "logits" (torch.Tensor): Raw, unnormalized scores for each class (batch_size, num_labels).
                - "loss" (torch.Tensor, optional): Cross-entropy loss if labels are provided.
        """
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


def compute_metrics(pred):
    """
    Computes accuracy, F1, precision, and recall for a given set of predictions.

    Args:
        pred (obj): An object containing label_ids and predictions attributes.
            - label_ids (array-like): A 1D array of true class labels.
            - predictions (array-like): A 2D array where each row represents
              an observation, and each column represents the probability of
              that observation belonging to a certain class.

    Returns:
        dict: A dictionary containing the following metrics:
            - Accuracy (float): The proportion of correctly classified instances.
            - F1 (float): The macro F1 score, which is the harmonic mean of precision
              and recall. Macro averaging calculates the metric independently for
              each class and then takes the average.
            - Precision (float): The macro precision, which is the number of true
              positives divided by the sum of true positives and false positives.
            - Recall (float): The macro recall, which is the number of true positives
              divided by the sum of true positives and false negatives.
    """
    # Extract true labels from the input object
    labels = pred.label_ids

    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)

    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)

    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }


def compute_metrics_wandb(pred):
    """
    Computes accuracy, F1, precision, and recall for a given set of predictions,
    including per-class metrics and confusion matrix visualization.

    Args:
        pred (obj): An object containing label_ids and predictions attributes.
            - label_ids (array-like): A 1D array of true class labels.
            - predictions (array-like): A 2D array where each row represents
              an observation, and each column represents the probability of
              that observation belonging to a certain class.

    Returns:
        dict: A dictionary containing the following metrics:
            - Accuracy (float)
            - F1 score (macro)
            - Precision (macro)
            - Recall (macro)
            - Per-class metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Compute overall macro precision, recall, F1 score, and accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    # Calculate per-class precision, recall, and F1 score
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    class_acc = [
        np.mean(labels[preds == i] == i) if np.any(labels == i) else 0.0
        for i in range(len(class_precision))]

    # Log metrics per class to WandB
    class_metrics = {
        f"Precision_class_{i}": class_precision[i] for i in range(len(class_precision))
    }
    class_metrics.update({
        f"Recall_class_{i}": class_recall[i] for i in range(len(class_recall))
    })
    class_metrics.update({
        f"F1_class_{i}": class_f1[i] for i in range(len(class_f1))
    })
    class_metrics.update({
        f"Accuracy_class_{i}": class_acc[i] for i in range(len(class_acc))
    })

    wandb.log(class_metrics)  # Log class-wise metrics to WandB

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Log confusion matrix image to WandB
    cm_fig = wandb.Image(plt)
    wandb.log({"Confusion Matrix": cm_fig})
    plt.close()

    # Return the overall metrics dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }


sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}


def predict(text, tokenizer, model, sentiment=None):
    """
    Predicts the class label for a given input text, optionally using sentiment.

    Args:
        text (str): The input text for which the class label needs to be predicted.
        tokenizer: The tokenizer to convert text to model inputs.
        model: The trained model for prediction.
        sentiment (str, optional): The sentiment label (e.g., "positive", "neutral", "negative").
        sentiment_mapping (dict, optional): Mapping from sentiment string to integer ID.

    Returns:
        probs (torch.Tensor): Class probabilities for the input text.
        pred_label_idx (torch.Tensor): The index of the predicted class label.
        pred_label (str): The predicted class label.
    """
    # Tokenize input and move to GPU
    inputs = tokenizer(
        text, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to("cuda")

    # Add sentiment if provided
    if sentiment is not None:
        sentiment_tensor = torch.tensor([sentiment_mapping[sentiment]]).to("cuda")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            sentiment=sentiment_tensor,
        )
    else:
        outputs = model(**inputs)

    # Convert logits to probabilities
    probs = outputs[0].softmax(1) if isinstance(outputs, tuple) else outputs.softmax(1)

    # Get predicted class index
    pred_label_idx = probs.argmax()
    pred_label = model.config.id2label[pred_label_idx.item()]

    return probs, pred_label_idx, pred_label
