"""
Alternative training loss functions.
"""

import numpy as np
import torch
import torch.nn as nn

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class ConstrativeLoss(nn.Module):
    """
    Contrastive Loss for embedding learning.
    Encourages embeddings of similar samples to be close and dissimilar samples to be apart.
    """
    def __init__(self, margin: float = 1.0) -> None:
        super(ConstrativeLoss, self).__init__()
        self.margin = margin
    
    def contrastive_loss(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            n_positives: int = 2,
            n_negatives: int = 2,
            margin: float = 1.0
        ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape (batch_size, num_classes).
        labels : torch.Tensor
            Binary labels of shape (batch_size,).
        n_positives : int, default=4
            Number of positive samples to consider per anchor.
        n_negatives : int, default=4
            Number of negative samples to consider per anchor.
        margin : float, default=1.0
            Margin for contrastive loss.

        Returns
        -------
        loss : torch.Tensor
            Computed contrastive loss.
        """
        batch_size = logits.size(0)
        num_pairs = 0
        loss = 0.0

        # Compute contrastive loss with hard negative mining
        n_positives = min(n_positives, batch_size - 1)  # n_positives positive samples per anchor
        n_negatives = min(n_negatives, batch_size - 1)  # n_negatives negative samples per anchor
        
        for i in range(batch_size):
            # Find positive and negative indices
            positive_idx = torch.where(labels == labels[i])[0]
            negative_idx = torch.where(labels != labels[i])[0]
            
            # Pair with up to n_positives positive samples
            if len(positive_idx) > 0:
                perm_idx = torch.randperm(len(positive_idx))[:n_positives].tolist()
                pos_samples = positive_idx[perm_idx]
                for pos_j in pos_samples:
                    if pos_j != i:
                        # Compute distance: want to be close
                        distance = torch.norm(logits[i] - logits[pos_j])
                        loss += distance ** 2
                        num_pairs += 1
            
            # Pair with up to n_negatives negative samples
            if len(negative_idx) > 0:
                perm_idx = torch.randperm(len(negative_idx))[:n_negatives].tolist()
                neg_samples = negative_idx[perm_idx]
                for neg_j in neg_samples:
                    # Compute distance: want to be at least margin apart
                    distance = torch.norm(logits[i] - logits[neg_j])
                    loss += torch.clamp(margin - distance, min=0) ** 2
                    num_pairs += 1
            
        # Normalize loss by number of pairs
        loss = loss / max(num_pairs, 1)
        return loss

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.contrastive_loss(logits, labels, margin=self.margin)
