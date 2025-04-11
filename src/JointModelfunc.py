import torch
from torch import nn
from typing import Tuple

class JointModel(torch.nn.Module):

    def __init__(self, embedding_weights: torch.Tensor, hidden_dim: int, num_layers: int, num_ner_tags: int = 9, num_sa_tags: int = 2):
        super().__init__()

        self.embedding: nn.Embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=True)
        self.lstm: nn.LSTM = nn.LSTM(embedding_weights.shape[1], hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)

        self.ner_classifier: nn.Linear = nn.Linear(in_features=hidden_dim *2, out_features=num_ner_tags)
        self.sa_classifier: nn.Linear = nn.Linear(in_features=hidden_dim *2, out_features=num_sa_tags)

    def forward(self, inputs: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]: # type: ignore
        #inputs has to contain word idxs
        embedded_words = self.embedding(inputs)
        lstm_out, (h, c) = self.lstm(embedded_words)

        sa_logit = self.sa_classifier(torch.mean(lstm_out,dim=1))
        ner_logits = self.ner_classifier(lstm_out)

        return torch.softmax(ner_logits, dim=-1), torch.softmax(sa_logit, dim=-1)
