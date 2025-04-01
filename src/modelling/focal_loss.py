import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions as logits, shape [batch_size, num_classes]
            targets: Either:
                  - Class indices, shape [batch_size], or
                  - One-hot encoded targets, shape [batch_size, num_classes]
        """
        # Check if targets are one-hot encoded
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            # Convert one-hot encoded targets to class indices
            targets = torch.argmax(targets, dim=1)
        
        # Standard cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # Get probabilities using softmax
        logp = F.log_softmax(inputs, dim=1)
        prob = torch.exp(logp)
        
        # Get probability for the target class for each sample
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply focal weighting
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
