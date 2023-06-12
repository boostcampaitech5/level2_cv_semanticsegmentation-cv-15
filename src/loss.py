import torch.nn.functional as F


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
