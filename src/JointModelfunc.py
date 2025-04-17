import torch
from torch import nn
from typing import Tuple


class JointModel(torch.nn.Module):
    """
    A joint model that performs both Named Entity Recognition (NER) and Sentiment Analysis (SA).
    This model uses an LSTM network to extract features and classifiers for NER and SA tasks.
    """

    def __init__(
        self,
        embedding_weights: torch.Tensor,  # Pretrained word embeddings for the input tokens
        hidden_dim: int,  # The hidden state dimension for the LSTM layer
        num_layers: int,  # Number of layers in the LSTM network
        num_ner_tags: int = 9,  # Number of NER tags (default is 9)
        num_sa_tags: int = 2,  # Number of Sentiment Analysis tags (default is 2: positive/negative)
    ) -> None:
        """
        Initializes the model with embedding weights, LSTM configurations, and classifier layers.

        Parameters:
        - embedding_weights (torch.Tensor): Pretrained word embeddings for the input tokens.
        - hidden_dim (int): The dimension of the LSTM's hidden states.
        - num_layers (int): The number of LSTM layers.
        - num_ner_tags (int): The number of possible NER tags (default 9).
        - num_sa_tags (int): The number of possible Sentiment Analysis tags (default 2: positive/negative).
        """
        super().__init__()

        # Embedding layer using the pretrained embedding weights
        self.embedding: nn.Embedding = nn.Embedding.from_pretrained(
            embedding_weights, freeze=True  # Freeze the embedding weights during training
        )

        # LSTM layer that processes the input sequences
        self.lstm: nn.LSTM = nn.LSTM(
            embedding_weights.shape[1],  # Input dimension is the size of the word embeddings
            hidden_size=hidden_dim,  # Hidden dimension for LSTM
            num_layers=num_layers,  # Number of layers in the LSTM
            bidirectional=True,  # Use bidirectional LSTM to capture both forward and backward context
        )

        # Linear layer for NER classification (using the LSTM output)
        self.ner_classifier: nn.Linear = nn.Linear(
            in_features=hidden_dim * 2,  # Since the LSTM is bidirectional, multiply by 2
            out_features=num_ner_tags,  # Number of NER tags
        )

        # Linear layer for SA classification (using the mean of LSTM output)
        self.sa_classifier: nn.Linear = nn.Linear(
            in_features=hidden_dim * 2,  # Again, account for bidirectional LSTM
            out_features=num_sa_tags,  # Number of SA tags
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass of the model, computing both NER and SA logits.

        Parameters:
        - inputs (torch.Tensor): The input tensor containing tokenized sentences (shape: [batch_size, seq_length]).

        Returns:
        - Tuple: A tuple of two tensors:
            1. NER logits (after softmax): Tensor of shape [batch_size, seq_length, num_ner_tags].
            2. SA logits (after softmax): Tensor of shape [batch_size, num_sa_tags].
        """
        # Obtain the word embeddings for the input tokens
        embedded_words = self.embedding(inputs)

        # Pass the embedded words through the LSTM layer
        lstm_out, (h, c) = self.lstm(embedded_words)

        # Compute sentiment analysis logits by averaging the LSTM output across the sequence
        sa_logit = self.sa_classifier(torch.mean(lstm_out, dim=1))

        # Compute NER logits directly from the LSTM output
        ner_logits = self.ner_classifier(lstm_out)

        # Apply softmax to the logits to get probabilities (for NER and SA)
        return torch.softmax(ner_logits, dim=-1), torch.softmax(sa_logit, dim=-1)
