import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import ast
from torch.nn.utils.rnn import pad_sequence
from typing import Any, List, Dict
from gensim.models import KeyedVectors  # type: ignore


class NERSentimentEmbeddingDataset(Dataset):
    """
    A custom Dataset class that handles tokenized sentences with NER and sentiment labels,
    and provides word embeddings using pretrained embeddings (Word2Vec or similar).
    """

    tokens: List[List[str]]  # List of tokenized words (for each sentence)
    pos_tags: List[Any]  # Part-of-speech tags for each token (type Any since it's not specified)
    chunk_tags: List[Any]  # Chunk tags for each token (type Any)
    ner_tags: List[Any]  # Named entity recognition tags for each token (type Any)
    sentences: List[str]  # Original sentences (as strings)
    labels: List[int]  # Sentiment labels (1 for positive, 0 for negative)
    word2vec: KeyedVectors  # Pretrained word embeddings (Gensim's KeyedVectors)
    embedding_dim: int  # Dimensionality of the word embeddings
    vocab: Dict[str, int]  # Vocabulary mapping each word to a unique index
    vocab_aux: Dict[int, str]  # Reverse vocabulary: mapping each index to the corresponding word

    def __init__(self, dataframe: pd.DataFrame, embeddings: KeyedVectors):
        """
        Initializes the NERSentimentEmbeddingDataset with data and word embeddings.

        Parameters:
        - dataframe: A pandas DataFrame containing tokenized sentences, part-of-speech tags, chunk tags,
          NER tags, sentences, and sentiment labels.
        - embeddings: Pretrained word embeddings (such as Word2Vec).
        """
        # Convert string representations of lists into actual lists (e.g., tokens, pos_tags)
        self.tokens = dataframe["tokens"].apply(ast.literal_eval).tolist()
        self.pos_tags = dataframe["pos_tags"].apply(ast.literal_eval).tolist()
        self.chunk_tags = dataframe["chunk_tags"].apply(ast.literal_eval).tolist()
        self.ner_tags = dataframe["ner_tags"].apply(ast.literal_eval).tolist()
        self.sentences = dataframe["sentence"].tolist()
        self.labels = [1 if label == "POSITIVE" else 0 for label in dataframe["label"]]

        # Store the embeddings and initialize vocabulary
        self.word2vec = embeddings
        self.embedding_dim = self.word2vec.vector_size
        self.vocab = {word: idx for idx, word in enumerate(self.word2vec.index_to_key)}
        self.vocab_aux = {idx: word for idx, word in enumerate(self.word2vec.index_to_key)}

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a sample from the dataset by index.
        Parameters:
        - idx: The index of the sample to retrieve.
        Returns:
        - A dictionary containing the tokens, POS tags, chunk tags, NER tags, sentence, label,
          word embeddings, and token IDs for the given sample.
        """
        # Retrieve the data for a specific sample
        tokens = self.tokens[idx]
        pos_tags = self.pos_tags[idx]
        chunk_tags = self.chunk_tags[idx]
        ner_tags = self.ner_tags[idx]
        sentence = self.sentences[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Prepare word embeddings and token IDs
        word_embeddings = []
        token_ids = []
        for word in tokens:
            word_lower = word.lower()
            if word_lower in self.word2vec:
                word_embeddings.append(self.word2vec[word_lower])  # Get the word embedding
                token_ids.append(self.vocab.get(word_lower, 0))  # Get the token ID (default 0 if not found)
            else:
                word_embeddings.append(np.zeros(self.embedding_dim))  # Use zero vector for unknown words
                token_ids.append(0)  # Token ID for unknown words

        # Convert embeddings and token IDs to tensors
        embeddings_tensor = torch.tensor(np.array(word_embeddings), dtype=torch.float32)
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)

        # Return the processed data
        return {
            "tokens": tokens,
            "pos_tags": pos_tags,
            "chunk_tags": chunk_tags,
            "ner_tags": ner_tags,
            "sentence": sentence,
            "label": label,
            "embeddings": embeddings_tensor,
            "input_ids": token_ids_tensor
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function to pad the batch and create a mini-batch from the dataset.
    Parameters:
    - batch: A list of dictionaries, each containing a sample from the dataset.
    Returns:
    - A dictionary containing padded input IDs, embeddings, labels, and NER tags for the mini-batch.
    """
    # Extract input data for the batch
    input_ids = [item["input_ids"] for item in batch]
    embeddings = [item["embeddings"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    ner_tags = [torch.tensor(item["ner_tags"]) for item in batch]

    # Pad the sequences to have equal length within the batch
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0)
    padded_ner_tags = pad_sequence(ner_tags, batch_first=True, padding_value=0)

    # Return the padded data
    return {
        "input_ids": padded_input_ids,
        "embeddings": padded_embeddings,
        "labels": labels,
        "ner_tags": padded_ner_tags
    }


def load_sentiment_dataloaders(
    embeddings: KeyedVectors, data_path: str = "data/NER_SA_csvs", batch_size: int = 3
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads sentiment data into train, validation, and test DataLoaders.
    Parameters:
    - embeddings: Pretrained word embeddings (e.g., Word2Vec).
    - data_path: Path to the folder containing the CSV files for the dataset.
    - batch_size: The batch size for DataLoader.
    Returns:
    - A tuple containing the DataLoader for train, validation, and test datasets.
    """
    # Read the datasets from CSV files
    df_train = pd.read_csv(os.path.join(data_path, "train.csv"))
    df_val = pd.read_csv(os.path.join(data_path, "validation.csv"))
    df_test = pd.read_csv(os.path.join(data_path, "test.csv"))

    # Create the dataset objects
    train_dataset = NERSentimentEmbeddingDataset(df_train, embeddings=embeddings)
    val_dataset = NERSentimentEmbeddingDataset(df_val, embeddings=embeddings)
    test_dataset = NERSentimentEmbeddingDataset(df_test, embeddings=embeddings)

    # Create DataLoader for each dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Return the DataLoaders
    return train_loader, val_loader, test_loader
