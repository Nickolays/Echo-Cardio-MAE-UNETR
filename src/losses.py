import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- Loss Functions -------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum() + self.smooth
        den = probs.sum() + targets.sum() + self.smooth
        dice = 1 - (num / den)
        bce = self.bce(logits, targets)
        return dice + bce


class DiceCELoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        # CrossEntropyLoss expects logits of shape (B, C, H, W) and targets of shape (B, H, W), where each element is an integer in [0, C‑1]
        ce_loss = self.ce(logits, targets)   

        # One-hot encode targets: (B, H, W) -> (B, C, H, W)
        targets_onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)

        dims = (0, 2, 3)  # sum over batch + spatial
        intersection = (probs * targets_onehot).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_onehot.sum(dim=dims)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return ce_loss + dice_loss
    

# class DiceFocalLoss(nn.Module):
#     def __init__(self, smooth=1.0, gamma=2.0):
#         super().__init__()
#         self.smooth = smooth
#         self.gamma = gamma

#     def forward(self, logits, targets):
#         probs = torch.sigmoid(logits)
#         num = 2 * (probs * targets).sum() + self.smooth
#         den = probs.sum() + targets.sum() + self.smooth
#         dice = 1 - (num / den)
#         # Focal
#         pt = torch.where(targets == 1, probs, 1 - probs)
#         focal = (1 - pt).pow(self.gamma).mean()
#         return dice + focal

class DiceFocalLoss(nn.Module):
    def __init__(self, num_classes=3, gamma=2.0, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        # Focal loss via log-softmax trick
        logp = F.log_softmax(logits, dim=1)           # (B, C, H, W)
        p = logp.exp()
        targets_onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        focal = -(1 - p) ** self.gamma * logp * targets_onehot
        focal_loss = focal.sum(dim=1).mean()

        # Dice loss (same as above)
        intersection = (p * targets_onehot).sum(dim=(0, 2, 3))
        union = p.sum(dim=(0, 2, 3)) + targets_onehot.sum(dim=(0, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return focal_loss + dice_loss