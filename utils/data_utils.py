import wandb
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

def read_data(train_file_name, test_file_name=None, dev_file_name=None):
    if (test_file_name != None and dev_file_name != None):

        train_df = pd.read_csv(train_file_name)
        test_df = pd.read_csv(test_file_name)
        dev_df = pd.read_csv(dev_file_name)

        train_dataset = train_df[['logical_fallacies', 'source_article_ro']]
        test_dataset = test_df[['logical_fallacies', 'source_article_ro']]
        dev_dataset = dev_df[['logical_fallacies', 'source_article_ro']]

        return train_dataset, test_dataset, dev_dataset
    else:
        dataset = pd.read_csv(train_file_name)
        size = dataset.shape[0]
        size_train = int(0.8 * size)
        size_test = int(0.2 * size)

        train_dataset = dataset.loc[:size_train - 1, :]
        test_dataset = dataset.loc[size_train:size_train + size_test - 1, :]
        dev_dataset = dataset.loc[size_train + size_test::]

        return train_dataset, test_dataset, dev_dataset


def filter_fallacies(train_data, test_data, dev_data, logical_fallacies):
    filtered_train_data = train_data[train_data.logical_fallacies.isin(logical_fallacies)]
    filtered_test_data = test_data[test_data.logical_fallacies.isin(logical_fallacies)]
    filtered_dev_data = dev_data[dev_data.logical_fallacies.isin(logical_fallacies)]
    return filtered_train_data, filtered_test_data, filtered_dev_data


def plot_training_curve(trainer):
    """
    Plots the training loss and accuracy over epochs for a Trainer instance and logs it to WandB.

    Args:
        trainer (transformers.Trainer): The Trainer instance after model training.
    """
    # Extract metrics from the trainer's log history
    metrics = trainer.state.log_history

    eval_accuracy = []
    eval_loss = []
    epochs = []

    # Extract training and evaluation losses from log history
    for log in metrics:
        if 'eval_Accuracy' in log:
            eval_accuracy.append(log['eval_Accuracy'])
            epochs.append(log['epoch'])  # Append epoch for training loss
        if 'eval_loss' in log:
            eval_loss.append(log['eval_loss'])

    # Check if there are mismatches in the length of eval_accuracy and eval_loss
    if len(eval_accuracy) != len(eval_loss):
        print(f"Warning: Mismatch in lengths of eval_accuracy ({len(eval_accuracy)}) and eval_loss ({len(eval_loss)})")

    # Ensure both lists have the same number of elements for plotting
    min_len = min(len(eval_accuracy), len(eval_loss))
    eval_accuracy = eval_accuracy[:min_len]
    eval_loss = eval_loss[:min_len]
    epochs = epochs[:min_len]

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, eval_accuracy, label='Training Accuracy', marker='o')
    plt.plot(epochs, eval_loss, label='Training Loss', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss & Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Log the plot to WandB
    plot = plt.gcf()  # Get the current figure
    wandb.log({"Train plot": wandb.Image(plot)})  # Log the plot as an image to WandB
    plt.close()  # Close the plot to free up memory
    return plot


def encode_labels(train_data, test_data, dev_data, label2id):
    # Ensure these are copies, not views
    train_data = train_data.copy()
    test_data = test_data.copy()
    dev_data = dev_data.copy()

    # Apply label encoding with .assign to avoid SettingWithCopyWarning
    train_data = train_data.assign(
        logical_fallacies_id=train_data['logical_fallacies'].apply(lambda x: label2id[x.strip()]))
    test_data = test_data.assign(
        logical_fallacies_id=test_data['logical_fallacies'].apply(lambda x: label2id[x.strip()]))
    dev_data = dev_data.assign(logical_fallacies_id=dev_data['logical_fallacies'].apply(lambda x: label2id[x.strip()]))

    # Confirm the function processed correctly by printing shapes
    print("train_data shape:", train_data.shape)
    print("test_data shape:", test_data.shape)
    print("dev_data shape:", dev_data.shape)

    return train_data, test_data, dev_data


def get_file_name(output_dir, model_name):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%d-%m-%Y_%H-%M-%S")
    trained_model_file = os.path.join(output_dir, formatted_time + "_" + str(model_name) + ".pickle")
    return trained_model_file
