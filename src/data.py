import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TextSentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer_name="distilbert-base-uncased", max_length=128):
        self.texts = dataframe["sentence"].tolist()
        self.labels = [1 if label == "POSITIVE" else 0 for label in dataframe["label"]]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label
        }


def load_sentiment_dataloaders(data_path="../data/NER_SA_csvs", batch_size=32):
    df_train = pd.read_csv(os.path.join(data_path, "train.csv"))
    df_val = pd.read_csv(os.path.join(data_path, "validation.csv"))
    df_test = pd.read_csv(os.path.join(data_path, "test.csv"))

    train_dataset = TextSentimentDataset(df_train)
    val_dataset = TextSentimentDataset(df_val)
    test_dataset = TextSentimentDataset(df_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_sentiment_dataloaders()
    batch = next(iter(train_loader))
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Attention Mask shape:", batch["attention_mask"].shape)
    print("Labels shape:", batch["label"].shape)
