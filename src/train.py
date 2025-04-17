import torch
from torch import optim
from tqdm import tqdm
import numpy as np
import data
import JointModelfunc, LossJointModel
from gensim.models import KeyedVectors
from utils import save_model, predict_alert



def train_model(model, train_loader, val_loader, num_epochs=1, lr=1e-5, alpha=1.0, beta=1.0, device=None):
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
            ner_tags = batch['ner_tags'].to(device)

            

            # Propagación hacia adelante
            optimizer.zero_grad()  # Limpiar los gradientes del optimizador
            ner_logits, sa_logits = model(input_ids)

            # Cálculo de la pérdida
            loss, loss_ner, loss_sa = LossJointModel.joint_loss(
                ner_logits, sa_logits, ner_labels=ner_tags, sa_labels=labels, alpha=alpha, beta=beta
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
            correct_ner += torch.sum((ner_preds == ner_tags) & mask)
            total_ner += mask.sum().item()

            #predict_alert(sa_logits, train_loader,input_ids,ner_tags)

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
    
    save_model(model, "best_model")

def validate_model(model, val_loader, device):
    model.eval()  # Establecer el modelo en modo de evaluación
    val_loss = 0.0
    correct_sa = 0
    total_sa = 0
    correct_ner = 0
    total_ner = 0

    # Contadores para cada etiqueta de NER
    ner_correct_counts = {i: 0 for i in range(9)}  # De 0 a 8
    ner_total_counts = {i: 0 for i in range(9)}  # De 0 a 8

    with torch.no_grad():  # No necesitamos calcular gradientes durante la validación
        for batch in tqdm(val_loader, desc="Validating"):
            # Cargar datos en el dispositivo
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            ner_tags = batch['ner_tags'].to(device)

            # Propagación hacia adelante
            ner_logits, sa_logits = model(input_ids)

            # Cálculo de la pérdida
            loss, loss_ner, loss_sa = LossJointModel.joint_loss(
                ner_logits, sa_logits, ner_labels=ner_tags, sa_labels=labels
            )
            val_loss += loss.item()

            # Medir la precisión de Sentiment Analysis
            _, predicted_sa = torch.max(sa_logits, dim=1)
            correct_sa += (predicted_sa == labels).sum().item()
            total_sa += labels.size(0)

            # Medir la precisión de NER
            ner_preds = torch.argmax(ner_logits, dim=-1)
            mask = input_ids != -100  # Ignorar los tokens de padding
            correct_ner += torch.sum((ner_preds == ner_tags) & mask)
            total_ner += mask.sum().item()

            # Contar aciertos por cada clase de NER
            for i in range(9):  # Para cada clase de NER (0 a 8)
                ner_correct_counts[i] += torch.sum((ner_preds == i) & (ner_tags == i) & mask).item()
                ner_total_counts[i] += torch.sum((ner_tags == i) & mask).item()

    # Calcular el porcentaje de precisión por cada clase de NER
    ner_accuracy_per_class = {
        i: (100 * ner_correct_counts[i] / ner_total_counts[i] if ner_total_counts[i] > 0 else 0)
        for i in range(9)
    }

    # Promediamos la pérdida en todas las iteraciones
    avg_val_loss = val_loss / len(val_loader)
    val_sa_accuracy = 100 * correct_sa / total_sa
    val_ner_accuracy = 100 * correct_ner / total_ner

    # Mostrar las precisiones por cada clase de NER
    for i in range(9):
        print(f"NER class {i}: {ner_accuracy_per_class[i]:.2f}% total: {ner_total_counts[i]} correct: {ner_correct_counts[i]}")

    return avg_val_loss, val_sa_accuracy, val_ner_accuracy


if __name__ == "__main__":
    w2v_path="models/GoogleNews-vectors-negative300.bin.gz"
    embedding_weights = KeyedVectors.load_word2vec_format(w2v_path, binary=True) # Ejemplo, deberías cargar embeddings reales
    # Cargar datos
    train_loader, val_loader, test_loader = data.load_sentiment_dataloaders(embedding_weights)
    
    # Cargar el modelo con los embeddings (tendrás que tener los embeddings pre-entrenados)
    embedding_weights = torch.tensor(embedding_weights.vectors, dtype=torch.float32)
    model = JointModelfunc.JointModel(hidden_dim=256, num_layers=2, embedding_weights=embedding_weights)


    # Entrenar el modelo
    train_model(model, train_loader, val_loader, num_epochs=8, lr=1e-5, alpha=1.0, beta=1.0)
