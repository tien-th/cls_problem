import torch.nn as nn
import torch

class Focal_KD(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(Focal_KD, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        logits,x_kd, x_fm = inputs
        BCE_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(logits, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = ((1-pt)**self.gamma) * BCE_loss

        KD_loss = torch.mean((x_kd - x_fm).pow(2))
        total_loss = F_loss + KD_loss

        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss
