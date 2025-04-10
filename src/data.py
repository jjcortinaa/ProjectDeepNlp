import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import numpy as np
import ast

class NERSentimentEmbeddingDataset(Dataset):
    def __init__(self, dataframe, w2v_path="../models/GoogleNews-vectors-negative300.bin.gz"):
        self.tokens = dataframe["tokens"].apply(ast.literal_eval).tolist()
        self.pos_tags = dataframe["pos_tags"].apply(ast.literal_eval).tolist()
        self.chunk_tags = dataframe["chunk_tags"].apply(ast.literal_eval).tolist()
        self.ner_tags = dataframe["ner_tags"].apply(ast.literal_eval).tolist()
        self.sentences = dataframe["sentence"].tolist()
        self.labels = [1 if label == "POSITIVE" else 0 for label in dataframe["label"]]

        self.word2vec = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        self.embedding_dim = self.word2vec.vector_size

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
        for word in tokens:
            word_lower = word.lower()
            if word_lower in self.word2vec:
                word_embeddings.append(self.word2vec[word_lower])
            else:
                word_embeddings.append(np.zeros(self.embedding_dim))

        embeddings_tensor = torch.tensor(word_embeddings, dtype=torch.float32)

        return {
            "tokens": tokens,
            "pos_tags": pos_tags,
            "chunk_tags": chunk_tags,
            "ner_tags": ner_tags,
            "sentence": sentence,
            "label": label,
            "embeddings": embeddings_tensor
        }


def load_sentiment_dataloaders(data_path="data/NER_SA_csvs", batch_size=32):
    df_train = pd.read_csv(os.path.join(data_path, "train.csv"))
    df_val = pd.read_csv(os.path.join(data_path, "validation.csv"))
    df_test = pd.read_csv(os.path.join(data_path, "test.csv"))

    train_dataset = NERSentimentEmbeddingDataset(df_train)
    val_dataset = NERSentimentEmbeddingDataset(df_val)
    test_dataset = NERSentimentEmbeddingDataset(df_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


