import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from monai.losses import DiceLoss

class DiceCrossEntropyLoss(nn.Module):
    """
    DiceLoss + CrossEntropyLoss
    """
    def __init__(self, ce_weight: float | int=0.1, dice_weight: float | int=0.9, **kwargs):
        """
        类构造函数
        :param kwargs: monai.losses.DiceLoss的参数
        :return:
        """
        super(DiceCrossEntropyLoss, self).__init__()
        total_weight = ce_weight + dice_weight
        self.ce_weight = ce_weight * 1.0 / total_weight
        self.dice_weight = dice_weight * 1.0 / total_weight
        self.cross_entropy = CrossEntropyLoss()
        self.dice = DiceLoss(**DiceCrossEntropyLoss.__get_valid_params(kwargs))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        损失函数的前向传播
        :param pred 预测张量
        :param target 标签张量
        :return: 损失
        """
        pred_ch, target_ch = pred.shape[1], target.shape[1]
        if pred_ch != target_ch and target_ch == 1:
            target_ce = torch.squeeze(target, dim=1).long()
        elif not torch.is_floating_point(target):
            target_ce = target.to(dtype=pred.dtype)
        else: target_ce = target

        ce_loss = self.cross_entropy(pred, target_ce)
        dice_loss = self.dice(pred, target)
        return (self.ce_weight * ce_loss + self.dice_weight * dice_loss).mean(dim=0).flatten()

    @staticmethod
    def __get_valid_params(kwargs: dict) -> dict:
        """
        获取DiceLoss的有效参数
        """
        import inspect
        valid_keys = set(inspect.signature(DiceLoss).parameters.keys())
        return {key: value for key, value in kwargs.items() if key in valid_keys}