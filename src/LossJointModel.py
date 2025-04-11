from typing import Tuple
import torch
from torch import nn

def joint_loss(
    ner_logits: torch.Tensor,     # (batch_size, seq_len, num_ner_tags)
    sa_logits: torch.Tensor,      # (batch_size, num_sa_tags)
    ner_labels: torch.Tensor,     # (batch_size, seq_len)
    sa_labels: torch.Tensor,      # (batch_size,)
    alpha: float = 1.0,           # peso para NER
    beta: float = 1.0             # peso para SA
) -> Tuple[torch.Tensor, float, float]:

    # Flatten NER logits y labels para CrossEntropy
    batch_size, seq_len, num_ner_tags = ner_logits.size()
    ner_logits_flat = ner_logits.view(-1, num_ner_tags)         # (batch * seq_len, num_ner_tags)
    ner_labels_flat = ner_labels.view(-1)                       # (batch * seq_len)

    # CrossEntropy para NER con padding ignorado
    loss_ner = nn.CrossEntropyLoss(weight=torch.Tensor([1,20,20,20,20,10,20,20,20]),ignore_index=-100)(ner_logits_flat, ner_labels_flat)

    # CrossEntropy para SA
    loss_sa = nn.CrossEntropyLoss()(sa_logits, sa_labels)

    # Combinamos con pesos
    total_loss = alpha * loss_ner + beta * loss_sa

    return total_loss, loss_ner.item(), loss_sa.item()