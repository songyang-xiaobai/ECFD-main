# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import weight_reduce_loss
from torch.nn.functional import one_hot
import torch.nn.functional as F


def dice_loss(seg_out, target, ignore_index):
    bs, nclass, H, W = seg_out.shape
    # dice_seg_out = torch.cat([seg_out, torch.zeros((bs, 1, H, W), device=seg_out.device)], dim=1)
    target = target.clone()
    if ignore_index is not None:
        target[target == ignore_index] = nclass

    target = F.one_hot(target, num_classes=nclass + 1).permute(0, 3, 1, 2)
    dice = BinaryDiceLoss()
    total_loss = torch.zeros(1).to(device=seg_out.device)
    for i in range(target.shape[1]):
        if i != nclass:
            dice_loss_item = dice(seg_out[:, i], target[:, i])
            total_loss = total_loss + dice_loss_item

    return total_loss / nclass


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target)) * 2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss


def naive_dice_loss(pred,
                    target,
                    weight=None,
                    eps=1e-3,
                    reduction='mean',
                    avg_factor=None):
    """Calculate naive dice loss, the coefficient in the denominator is the
    first power instead of the second power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input, 1)
    c = torch.sum(target, 1)
    d = (2 * a + eps) / (b + c + eps)
    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module(force=True)
class DiceLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-3):
        """Dice Loss, there are two forms of dice loss is supported:

            - the one proposed in `V-Net: Fully Convolutional Neural
                Networks for Volumetric Medical Image Segmentation
                <https://arxiv.org/abs/1606.04797>`_.
            - the dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """

        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        self.ignore_index = ignore_index

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        if self.naive_dice:
            loss = self.loss_weight * naive_dice_loss(
                pred,
                target,
                weight,
                eps=self.eps,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            loss = self.loss_weight * dice_loss(
                pred,
                target,
                ignore_index=self.ignore_index,
                )

        return loss
