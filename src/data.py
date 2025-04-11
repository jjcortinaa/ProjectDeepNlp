import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import ast
from torch.nn.utils.rnn import pad_sequence
import torch

class NERSentimentEmbeddingDataset(Dataset):
    def __init__(self, dataframe,embeddings):
        self.tokens = dataframe["tokens"].apply(ast.literal_eval).tolist()
        self.pos_tags = dataframe["pos_tags"].apply(ast.literal_eval).tolist()
        self.chunk_tags = dataframe["chunk_tags"].apply(ast.literal_eval).tolist()
        self.ner_tags = dataframe["ner_tags"].apply(ast.literal_eval).tolist()
        self.sentences = dataframe["sentence"].tolist()
        self.labels = [1 if label == "POSITIVE" else 0 for label in dataframe["label"]]

        # Cargar el modelo de Word2Vec
        self.word2vec = embeddings
        self.embedding_dim = self.word2vec.vector_size

        # Crear vocabulario
        self.vocab = {word: idx for idx, word in enumerate(self.word2vec.index_to_key)}

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        pos_tags = self.pos_tags[idx]
        chunk_tags = self.chunk_tags[idx]
        ner_tags = self.ner_tags[idx]
        sentence = self.sentences[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Obtener embeddings por palabra
        word_embeddings = []
        token_ids = []  # Lista de IDs de tokens
        for word in tokens:
            word_lower = word.lower()
            if word_lower in self.word2vec:
                word_embeddings.append(self.word2vec[word_lower])
                token_ids.append(self.vocab.get(word_lower, 0))  # Asignar un ID, 0 si no está en el vocabulario
            else:
                word_embeddings.append(np.zeros(self.embedding_dim))
                token_ids.append(0)  # ID para palabras desconocidas

        embeddings_tensor = torch.tensor(word_embeddings, dtype=torch.float32)
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)

        return {
            "tokens": tokens,
            "pos_tags": pos_tags,
            "chunk_tags": chunk_tags,
            "ner_tags": ner_tags,
            "sentence": sentence,
            "label": label,
            "embeddings": embeddings_tensor,
            "input_ids": token_ids_tensor  # Agregar input_ids con los índices de las palabras
        }


def collate_fn(batch):
    # Extraer input_ids, embeddings y etiquetas del lote
    input_ids = [item["input_ids"] for item in batch]  # Añadimos 'input_ids'
    embeddings = [item["embeddings"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    ner_tags = [torch.tensor(item["ner_tags"]) for item in batch]

    # Realizar padding en las secuencias de input_ids y embeddings
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0)
    padded_ner_tags = pad_sequence(ner_tags, batch_first=True, padding_value=0)

    return {
        "input_ids": padded_input_ids,  # Asegúrate de devolver 'input_ids'
        "embeddings": padded_embeddings,
        "labels": labels,
        "ner_tags":padded_ner_tags
    }


def load_sentiment_dataloaders(embeddings, data_path="data/NER_SA_csvs", batch_size=3):
    df_train = pd.read_csv(os.path.join(data_path, "train.csv"))
    df_val = pd.read_csv(os.path.join(data_path, "validation.csv"))
    df_test = pd.read_csv(os.path.join(data_path, "test.csv"))

    train_dataset = NERSentimentEmbeddingDataset(df_train, embeddings=embeddings)
    val_dataset = NERSentimentEmbeddingDataset(df_val, embeddings=embeddings)
    test_dataset = NERSentimentEmbeddingDataset(df_test, embeddings=embeddings)

    # Aquí se pasa la función collate_fn para el padding
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader