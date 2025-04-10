import torch
from torch import optim
from tqdm import tqdm
import numpy as np
import data
import JointModelfunc, LossJointModel

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-5, alpha=1.0, beta=1.0, device=None):
    # Asegúrate de mover el modelo al dispositivo correcto (CPU o GPU)
    if device is None:
        device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    
    model.to(device)

    # Definir el optimizador (Adam)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Bucle de entrenamiento
    for epoch in range(num_epochs):
        model.train()  # Establecemos el modelo en modo de entrenamiento
        train_loss = 0.0
        correct_sa = 0
        total_sa = 0
        correct_ner = 0
        total_ner = 0
        
        # Iteramos sobre el DataLoader de entrenamiento
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            # Cargar datos en el dispositivo
            labels = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            

            # Propagación hacia adelante
            optimizer.zero_grad()  # Limpiar los gradientes del optimizador
            ner_logits, sa_logits = model(input_ids)

            # Cálculo de la pérdida
            loss, loss_ner, loss_sa = LossJointModel.joint_loss(
                ner_logits, sa_logits, ner_labels=input_ids, sa_labels=labels, alpha=alpha, beta=beta
            )
            train_loss += loss.item()

            # Actualizamos los gradientes
            loss.backward()
            optimizer.step()

            # Medir la precisión de Sentiment Analysis
            _, predicted_sa = torch.max(sa_logits, dim=1)
            correct_sa += (predicted_sa == labels).sum().item()
            total_sa += labels.size(0)

            # Medir la precisión de NER
            ner_preds = torch.argmax(ner_logits, dim=-1)
            mask = input_ids != -100  # Ignorar los tokens de padding
            correct_ner += (ner_preds == input_ids) & mask
            total_ner += mask.sum().item()

        # Promediamos la pérdida en todas las iteraciones
        avg_train_loss = train_loss / len(train_loader)
        train_sa_accuracy = 100 * correct_sa / total_sa
        train_ner_accuracy = 100 * correct_ner / total_ner

        print(f"Epoch {epoch + 1} - Train loss: {avg_train_loss:.4f} | "
              f"Train SA Accuracy: {train_sa_accuracy:.2f}% | Train NER Accuracy: {train_ner_accuracy:.2f}%")

        # Validación
        val_loss, val_sa_accuracy, val_ner_accuracy = validate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1} - Validation loss: {val_loss:.4f} | "
              f"Validation SA Accuracy: {val_sa_accuracy:.2f}% | Validation NER Accuracy: {val_ner_accuracy:.2f}%")

def validate_model(model, val_loader, device):
    model.eval()  # Establecer el modelo en modo de evaluación
    val_loss = 0.0
    correct_sa = 0
    total_sa = 0
    correct_ner = 0
    total_ner = 0

    with torch.no_grad():  # No necesitamos calcular gradientes durante la validación
        for batch in tqdm(val_loader, desc="Validating"):
            # Cargar datos en el dispositivo
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Propagación hacia adelante
            ner_logits, sa_logits = model(input_ids)

            # Cálculo de la pérdida
            loss, loss_ner, loss_sa = LossJointModel.joint_loss(
                ner_logits, sa_logits, ner_labels=input_ids, sa_labels=labels
            )
            val_loss += loss.item()

            # Medir la precisión de Sentiment Analysis
            _, predicted_sa = torch.max(sa_logits, dim=1)
            correct_sa += (predicted_sa == labels).sum().item()
            total_sa += labels.size(0)

            # Medir la precisión de NER
            ner_preds = torch.argmax(ner_logits, dim=-1)
            mask = input_ids != -100  # Ignorar los tokens de padding
            correct_ner += (ner_preds == input_ids) & mask
            total_ner += mask.sum().item()

    # Promediamos la pérdida en todas las iteraciones
    avg_val_loss = val_loss / len(val_loader)
    val_sa_accuracy = 100 * correct_sa / total_sa
    val_ner_accuracy = 100 * correct_ner / total_ner

    return avg_val_loss, val_sa_accuracy, val_ner_accuracy

if __name__ == "__main__":
    # Cargar datos
    train_loader, val_loader, test_loader = data.load_sentiment_dataloaders()
    
    # Cargar el modelo con los embeddings (tendrás que tener los embeddings pre-entrenados)
    embedding_weights = torch.randn(10000, 300)  # Ejemplo, deberías cargar embeddings reales
    model = JointModelfunc.JointModel(embedding_weights, hidden_dim=256, num_layers=2)

    # Entrenar el modelo
    train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-5, alpha=1.0, beta=1.0)
