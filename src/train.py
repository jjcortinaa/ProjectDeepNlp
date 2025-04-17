import torch
from torch import optim
from tqdm import tqdm
import data
import JointModelfunc
import LossJointModel
from gensim.models import KeyedVectors  # type: ignore
from utils import save_model
from collections import Counter
from torch.utils.data import DataLoader
from typing import Tuple


def compute_ner_class_weights(
    dataloader: DataLoader, num_classes: int = 9, ignore_index: int = -100
) -> torch.Tensor:
    """
    Computes the class weights for the NER task based on the frequency of each tag in the dataset.
    This helps to handle class imbalance by giving more weight to less frequent tags.
    Parameters:
    - dataloader (DataLoader): The data loader that provides batches of data.
    - num_classes (int): The number of classes for NER (default 9).
    - ignore_index (int): The index to ignore when calculating class frequencies (default -100).
    Returns:
    - torch.Tensor: A tensor containing the computed class weights, normalized to sum to 1.
    """
    total_counts = Counter()

    # Iterate over batches in the dataloader to count the occurrences of NER tags
    for batch in dataloader:
        ner_tags = batch["ner_tags"].view(-1)
        valid_tags = ner_tags[ner_tags != ignore_index].tolist()
        total_counts.update(valid_tags)

    total = sum(total_counts.values())
    weights = []

    # Compute the weight for each class
    for i in range(num_classes):
        count = total_counts.get(i, 1)  # Default to 1 if the class doesn't appear
        weights.append(total / count)

    # Convert weights to tensor and normalize
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights / weights.sum()  # Normalize the weights


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 1,
    lr: float = 1e-5,
    alpha: float = 1.0,
    beta: float = 1.0,
    device: torch.device = None,
) -> None:
    """
    Trains the Joint Model on the provided training data and evaluates it on validation data.

    Parameters:
    - model (torch.nn.Module): The model to train.
    - train_loader (DataLoader): The DataLoader for the training data.
    - val_loader (DataLoader): The DataLoader for the validation data.
    - num_epochs (int): Number of epochs to train the model (default 1).
    - lr (float): Learning rate for the optimizer (default 1e-5).
    - alpha (float): Weighting factor for the NER loss (default 1.0).
    - beta (float): Weighting factor for the SA loss (default 1.0).
    - device (torch.device): Device to train on (default None, uses GPU or CPU).
    """
    if device is None:
        device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Initialize optimizer and compute NER class weights
    optimizer = optim.Adam(model.parameters(), lr=lr)
    class_weights = compute_ner_class_weights(train_loader, num_classes=9).to(device)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        correct_sa = 0
        total_sa = 0
        correct_ner = 0
        total_ner = 0

        # Iterate over training batches
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            ner_tags = batch["ner_tags"].to(device)

            optimizer.zero_grad()

            # Forward pass
            ner_logits, sa_logits = model(input_ids)

            # Compute loss
            loss, _, _ = LossJointModel.joint_loss(
                ner_logits,
                sa_logits,
                ner_labels=ner_tags,
                sa_labels=labels,
                alpha=alpha,
                beta=beta,
                ner_class_weights=class_weights,
            )
            train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate Sentiment Analysis accuracy
            _, predicted_sa = torch.max(sa_logits, dim=1)
            correct_sa += (predicted_sa == labels).sum().item()
            total_sa += labels.size(0)

            # Calculate Named Entity Recognition accuracy
            ner_preds = torch.argmax(ner_logits, dim=-1)
            mask = input_ids != -100
            correct_ner += torch.sum((ner_preds == ner_tags) & mask)
            total_ner += mask.sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_sa_accuracy = 100 * correct_sa / total_sa
        train_ner_accuracy = 100 * correct_ner / total_ner

        print(
            f"Epoch {epoch + 1} - Train loss: {avg_train_loss:.4f} | "
            f"Train SA Accuracy: {train_sa_accuracy:.2f}% | Train NER Accuracy: {train_ner_accuracy:.2f}%"
        )

        # Validate the model after each epoch
        val_loss, val_sa_accuracy, val_ner_accuracy = validate_model(
            model, val_loader, device, class_weights
        )
        print(
            f"Epoch {epoch + 1} - Validation loss: {val_loss:.4f} | "
            f"Validation SA Accuracy: {val_sa_accuracy:.2f}% | Validation NER Accuracy: {val_ner_accuracy:.2f}%"
        )

    # Save the best model after training
    save_model(model, "best_model")


def validate_model(
    model: torch.nn.Module, val_loader: DataLoader, device: torch.device, class_weights: torch.Tensor
) -> Tuple[float, float, float]:
    """
    Validates the model on the validation dataset and computes various performance metrics.

    Parameters:
    - model (torch.nn.Module): The model to validate.
    - val_loader (DataLoader): The DataLoader for the validation data.
    - device (torch.device): Device to run validation on.
    - class_weights (torch.Tensor): The class weights for NER.

    Returns:
    - Tuple: A tuple containing:
        1. Average validation loss (float).
        2. Sentiment Analysis accuracy (float).
        3. Named Entity Recognition accuracy (float).
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct_sa = 0
    total_sa = 0
    correct_ner = 0
    total_ner = 0

    ner_correct_counts = {i: 0 for i in range(9)}
    ner_total_counts = {i: 0 for i in range(9)}

    with torch.no_grad():  # No gradients required during evaluation
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            ner_tags = batch["ner_tags"].to(device)

            # Forward pass
            ner_logits, sa_logits = model(input_ids)

            # Compute validation loss
            loss, _, _ = LossJointModel.joint_loss(
                ner_logits,
                sa_logits,
                ner_labels=ner_tags,
                sa_labels=labels,
                ner_class_weights=class_weights,
            )
            val_loss += loss.item()

            # Calculate Sentiment Analysis accuracy
            _, predicted_sa = torch.max(sa_logits, dim=1)
            correct_sa += (predicted_sa == labels).sum().item()
            total_sa += labels.size(0)

            # Calculate Named Entity Recognition accuracy
            ner_preds = torch.argmax(ner_logits, dim=-1)
            mask = input_ids != -100
            correct_ner += torch.sum((ner_preds == ner_tags) & mask)
            total_ner += mask.sum().item()

            for i in range(9):
                ner_correct_counts[i] += torch.sum(
                    (ner_preds == i) & (ner_tags == i) & mask
                ).item()
                ner_total_counts[i] += torch.sum((ner_tags == i) & mask).item()

    ner_accuracy_per_class = {
        i: (
            100 * ner_correct_counts[i] / ner_total_counts[i]
            if ner_total_counts[i] > 0
            else 0
        )
        for i in range(9)
    }

    avg_val_loss = val_loss / len(val_loader)
    val_sa_accuracy = 100 * correct_sa / total_sa
    val_ner_accuracy = 100 * correct_ner / total_ner

    for i in range(9):
        print(
            f"NER class {i}: {ner_accuracy_per_class[i]:.2f}% total: {ner_total_counts[i]} correct: {ner_correct_counts[i]}"
        )

    return avg_val_loss, val_sa_accuracy, val_ner_accuracy


if __name__ == "__main__":
    # Load pretrained embeddings
    w2v_path = "models/GoogleNews-vectors-negative300.bin.gz"
    embedding_weights = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    # Load data loaders for training, validation, and testing
    train_loader, val_loader, test_loader = data.load_sentiment_dataloaders(embedding_weights)

    embedding_weights = torch.tensor(embedding_weights.vectors, dtype=torch.float32)

    # Initialize the model
    model = JointModelfunc.JointModel(
        hidden_dim=256, num_layers=2, embedding_weights=embedding_weights
    )

    # Train the model
    train_model(
        model, train_loader, val_loader, num_epochs=35, lr=1e-5, alpha=1.0, beta=1.0
    )
