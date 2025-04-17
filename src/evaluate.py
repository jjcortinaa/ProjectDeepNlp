from gensim.models import KeyedVectors  # type: ignore
import LossJointModel
import torch
import data
from typing import Final, Tuple
from tqdm import tqdm
from utils import predict_alert

CHECKPOINT_PATH: Final[str] = "models/best_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> None:
    """
    Evaluates the model on the test data and prints the loss and accuracy metrics.

    Parameters:
    - model (torch.nn.Module): The trained model to evaluate.
    - test_loader (torch.utils.data.DataLoader): The DataLoader containing the test data.
    - device (torch.device): The device (CPU or GPU) on which to run the model.
    """
    avg_test_loss, test_sa_accuracy, test_ner_accuracy = compute_loss(
        model, test_loader, device, desc="Evaluating"
    )

    print(
        f"\nTest loss: {avg_test_loss:.4f} | "
        f"Test SA Accuracy: {test_sa_accuracy:.2f}% | "
        f"Test NER Accuracy: {test_ner_accuracy:.2f}%"
    )


def compute_loss(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str = "Evaluating"
) -> Tuple[float, float, float]:
    """
    Computes the loss and accuracy for sentiment analysis (SA) and named entity recognition (NER) on a given dataset.

    Parameters:
    - model (torch.nn.Module): The model to evaluate.
    - data_loader (torch.utils.data.DataLoader): The DataLoader containing the data to evaluate.
    - device (torch.device): The device (CPU or GPU) on which to run the model.
    - desc (str): The description for the tqdm progress bar (default is "Evaluating").

    Returns:
    - Tuple: A tuple containing:
        1. Average loss (float).
        2. Sentiment Analysis accuracy (float).
        3. Named Entity Recognition accuracy (float).
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_sa = 0
    total_sa = 0
    correct_ner = 0
    total_ner = 0

    with torch.no_grad():  # No gradients required during evaluation
        for batch in tqdm(data_loader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            ner_tags = batch["ner_tags"].to(device)

            # Forward pass
            ner_logits, sa_logits = model(input_ids)

            # Compute loss
            loss, _, _ = LossJointModel.joint_loss(
                ner_logits, sa_logits, ner_labels=ner_tags, sa_labels=labels
            )
            total_loss += loss.item()

            # Sentiment Analysis accuracy calculation
            _, predicted_sa = torch.max(sa_logits, dim=1)
            correct_sa += (predicted_sa == labels).sum().item()
            total_sa += labels.size(0)

            # Named Entity Recognition accuracy calculation
            ner_preds = torch.argmax(ner_logits, dim=-1)
            mask = input_ids != -100  # Ignore padding tokens
            correct_ner += torch.sum((ner_preds == ner_tags) & mask)
            total_ner += mask.sum().item()

            # Detect potential alerts in the data
            batch_alerts = predict_alert(
                sa_logits=sa_logits,
                train_loader=data_loader,
                input_ids=input_ids,
                ner_tags=ner_tags,
            )
            for alert in batch_alerts:
                if alert.strip():  # Only print non-empty alerts
                    print(f"\n{alert}")

    avg_loss = total_loss / len(data_loader)
    sa_accuracy = 100 * correct_sa / total_sa
    ner_accuracy = 100 * correct_ner / total_ner

    return avg_loss, sa_accuracy, ner_accuracy


def main() -> None:
    """
    Main function that loads the model, evaluates it on the test data, and prints the results.
    """
    w2v_path = "models/GoogleNews-vectors-negative300.bin.gz"
    embedding_weights = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    # Load the test data loader
    _, _, test_loader = data.load_sentiment_dataloaders(embedding_weights)

    # Convert embedding weights to a tensor
    embedding_weights = torch.tensor(embedding_weights.vectors, dtype=torch.float32)

    # Load the trained model from the checkpoint
    model = torch.jit.load(CHECKPOINT_PATH).to(device)
    model.eval()  # Set model to evaluation mode

    # Evaluate the model on the test data
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
