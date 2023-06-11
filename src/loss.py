import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss:
    def __init__(self, smooth=1.0):
        self.smooth = smooth

    def __call__(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = 1 - (
            (2.0 * intersection + self.smooth)
            / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)
        )
        return loss.mean()


class DiceBCELoss:
    def __init__(self, dice_smooth=1.0, bce_weight=0.5):
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=dice_smooth)

    def __call__(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = F.sigmoid(pred)
        dice = self.dice_loss(pred, target)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        return loss


class AveragedHausdorffLoss(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, set1, set2):
        assert set1.ndimension() == 2, "got %s" % set1.ndimension()
        assert set2.ndimension() == 2, "got %s" % set2.ndimension()
        assert set1.size()[1] == set2.size()[1], "got %s and %s." % (
            set2.size()[1],
            set2.size()[1],
        )

        d2_matrix = self.cdist(set1, set2)

        # Modified Chamfer Loss
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

        res = term_1 + term_2

        return res

    def cdist(x, y):
        differences = x.unsqueeze(1) - y.unsqueeze(0)
        distances = torch.sum(differences**2, -1).sqrt()
        return distances


class DiceBCEHausdorffLoss:
    def __init__(self, dice_smooth=1.0, bce_weight=0.5):
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.hausdorff_loss = AveragedHausdorffLoss()

    def __call__(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = F.sigmoid(pred)
        dice = self.dice_loss(pred, target)
        hausdorff = self.hausdorff_loss(pred, target)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) + hausdorff
        return loss
