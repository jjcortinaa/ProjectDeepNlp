from typing import Tuple, Optional
import torch
from torch import nn
from torch import Tensor


def joint_loss(
    ner_logits: torch.Tensor,  # (batch_size, seq_len, num_ner_tags) The NER (Named Entity Recognition) logits
    sa_logits: torch.Tensor,   # (batch_size, num_sa_tags) The SA (Sentiment Analysis) logits
    ner_labels: torch.Tensor,  # (batch_size, seq_len) The true labels for NER
    sa_labels: torch.Tensor,   # (batch_size,) The true labels for SA
    alpha: float = 1.0,        # Weight for NER loss component
    beta: float = 1.0,         # Weight for SA loss component
    ner_class_weights: Optional[Tensor] = None,  # Class weights for NER, used to balance the class distribution
) -> Tuple[torch.Tensor, float, float]:
    """
    Computes the joint loss for both Sentiment Analysis (SA) and Named Entity Recognition (NER).

    Parameters:
    - ner_logits (torch.Tensor): The raw predictions for NER (shape: batch_size x seq_len x num_ner_tags).
    - sa_logits (torch.Tensor): The raw predictions for Sentiment Analysis (shape: batch_size x num_sa_tags).
    - ner_labels (torch.Tensor): The true labels for NER (shape: batch_size x seq_len).
    - sa_labels (torch.Tensor): The true labels for SA (shape: batch_size).
    - alpha (float): Weight for the NER loss. Default is 1.0.
    - beta (float): Weight for the SA loss. Default is 1.0.
    - ner_class_weights (Optional[Tensor]): Class weights for NER to handle class imbalance. Default is None.

    Returns:
    - Tuple[torch.Tensor, float, float]: A tuple containing:
      1. Total loss (combined NER and SA loss).
      2. NER loss (float).
      3. SA loss (float).
    """

    # Get the number of NER tags (classes)
    _, _, num_ner_tags = ner_logits.size()

    # Flatten the logits and labels to calculate the loss for NER
    ner_logits_flat = ner_logits.view(-1, num_ner_tags)
    ner_labels_flat = ner_labels.view(-1)

    # Compute NER loss using CrossEntropyLoss
    loss_ner = nn.CrossEntropyLoss(weight=ner_class_weights, ignore_index=-100)(
        ner_logits_flat, ner_labels_flat
    )

    # Compute SA loss using CrossEntropyLoss
    loss_sa = nn.CrossEntropyLoss()(sa_logits, sa_labels)

    # Combine the losses using the alpha and beta weights
    total_loss = alpha * loss_ner + beta * loss_sa

    # Return the total loss and individual losses for NER and SA
    return total_loss, loss_ner.item(), loss_sa.item()
