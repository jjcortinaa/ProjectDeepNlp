from gensim.models import KeyedVectors
import JointModelfunc, LossJointModel
import torch
import data
from typing import Final
from tqdm import tqdm

CHECKPOINT_PATH: Final[str] = "models/best_model.pt"
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_loader, device):
    avg_test_loss, test_sa_accuracy, test_ner_accuracy = compute_loss(model, test_loader, device, desc="Evaluating")
    print(f"Test loss: {avg_test_loss:.4f} | "
          f"Test SA Accuracy: {test_sa_accuracy:.2f}% | "
          f"Test NER Accuracy: {test_ner_accuracy:.2f}%")
    
def compute_loss(model, data_loader, device, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    correct_sa = 0
    total_sa = 0
    correct_ner = 0
    total_ner = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            ner_tags = batch['ner_tags'].to(device)

            ner_logits, sa_logits = model(input_ids)
            
            loss, loss_ner, loss_sa = LossJointModel.joint_loss(
                ner_logits, sa_logits, ner_labels=ner_tags, sa_labels=labels
            )
            total_loss += loss.item()

            _, predicted_sa = torch.max(sa_logits, dim=1)
            correct_sa += (predicted_sa == labels).sum().item()
            total_sa += labels.size(0)

            ner_preds = torch.argmax(ner_logits, dim=-1)
            mask = input_ids != -100
            correct_ner += torch.sum((ner_preds == ner_tags) & mask)
            total_ner += mask.sum().item()

    avg_loss = total_loss / len(data_loader)
    sa_accuracy = 100 * correct_sa / total_sa
    ner_accuracy = 100 * correct_ner / total_ner

    return avg_loss, sa_accuracy, ner_accuracy


def main():
    w2v_path="models/GoogleNews-vectors-negative300.bin.gz"
    embedding_weights = KeyedVectors.load_word2vec_format(w2v_path, binary=True) # Ejemplo, deberías cargar embeddings reales
    # Cargar datos
    train_loader, val_loader, test_loader = data.load_sentiment_dataloaders(embedding_weights)
    
    # Cargar el modelo con los embeddings (tendrás que tener los embeddings pre-entrenados)
    embedding_weights = torch.tensor(embedding_weights.vectors, dtype=torch.float32)
    model = torch.jit.load(CHECKPOINT_PATH).to(device)

    model.eval()

    evaluate_model(model,test_loader, device)


if __name__ == "__main__":
    main()

