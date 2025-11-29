"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch import Tensor
from typing import Optional
from loguru import logger

from nndet.losses.base import reduction_helper
from nndet.utils import make_onehot_batch


def one_hot_smooth(data,
                   num_classes: int,
                   smoothing: float = 0.0,
                   ):
    targets = torch.empty(size=(*data.shape, num_classes), device=data.device)\
        .fill_(smoothing / num_classes)\
        .scatter_(-1, data.long().unsqueeze(-1), 1. - smoothing)
    return targets


@torch.jit.script
def focal_loss_with_logits(
        logits: torch.Tensor,
        target: torch.Tensor, gamma: float,
        alpha: float = -1,
        reduction: str = "mean",
        ) -> torch.Tensor:
    """
    Focal loss
    https://arxiv.org/abs/1708.02002

    Args:
        logits: predicted logits [N, dims]
        target: (float) binary targets [N, dims]
        gamma: balance easy and hard examples in focal loss
        alpha: balance positive and negative samples [0, 1] (increasing
            alpha increase weight of foreground classes (better recall))
        reduction: 'mean'|'sum'|'none'
            mean: mean of loss over entire batch
            sum: sum of loss over entire batch
            none: no reduction

    Returns:
        torch.Tensor: loss

    See Also
        :class:`BFocalLossWithLogits`, :class:`FocalLossWithLogits`
    """
    bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

    p = torch.sigmoid(logits)
    pt = (p * target + (1 - p) * (1 - target))

    focal_term = (1. - pt).pow(gamma)
    loss = focal_term * bce_loss

    if alpha >= 0:
        alpha_t = (alpha * target + (1 - alpha) * (1 - target))
        loss = alpha_t * loss

    return reduction_helper(loss, reduction=reduction)


class FocalLossWithLogits(nn.Module):
    def __init__(self,
                 gamma: float = 2,
                 alpha: float = -1,
                 class_weights: Optional[torch.Tensor] = None,  # 新增参数
                 reduction: str = "sum",
                 loss_weight: float = 1.,
                 ):
        """
        Enhanced Focal loss with class weighting support for class imbalance

        Args:
            gamma: balance easy and hard examples in focal loss
            alpha: balance positive and negative samples [0, 1] (increasing
                alpha increase weight of foreground classes (better recall))
            class_weights: Tensor of shape (num_classes,) for class-specific weights
            reduction: 'mean'|'sum'|'none'
                mean: mean of loss over entire batch
                sum: sum of loss over entire batch
                none: no reduction
            loss_weight: scalar to balance multiple losses
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                ) -> torch.Tensor:
        """
        Compute focal loss with optional class weighting

        Args:
            logits: predicted logits [N, C, dims], where N is the batch size,
                C number of classes, dims are arbitrary spatial dimensions
            targets: targets encoded as numbers [N, dims], where N is the
                batch size, dims are arbitrary spatial dimensions

        Returns:
            torch.Tensor: loss
        """
        n_classes = logits.shape[1] + 1
        target_onehot = make_onehot_batch(targets, n_classes=n_classes).float()
        target_onehot = target_onehot[:, 1:]  # 移除背景类

        # 计算基础focal loss
        base_loss = focal_loss_with_logits(
            logits, target_onehot,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction='none',  # 先不归约，应用类别权重后再归约
            )
        
        # 应用类别权重
        if self.class_weights is not None:
            # 为每个样本应用对应类别的权重
            class_indices = torch.argmax(target_onehot, dim=1)  # [N]
            sample_weights = self.class_weights[class_indices].unsqueeze(1)  # [N, 1]
            sample_weights = sample_weights.expand_as(target_onehot)  # [N, C]
            base_loss = base_loss * sample_weights
        
        # 应用归约
        if self.reduction == 'mean':
            base_loss = base_loss.mean()
        elif self.reduction == 'sum':
            base_loss = base_loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

        return self.loss_weight * base_loss


class BCEWithLogitsLossOneHot(torch.nn.BCEWithLogitsLoss):
    def __init__(self,
                 *args,
                 num_classes: int,
                 smoothing: float = 0.0,
                 loss_weight: float = 1.,
                 **kwargs,
                 ):
        """
        BCE loss with one hot encoding of targets

        Args:
            num_classes: number of classes
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
        """
        super().__init__(*args, **kwargs)
        self.smoothing = smoothing
        if smoothing > 0:
            logger.info(f"Running label smoothing with smoothing: {smoothing}")
        self.num_classes = num_classes
        self.loss_weight = loss_weight

    def forward(self,
                input: Tensor,
                target: Tensor,
                ) -> Tensor:
        """
        Compute bce loss based on one hot encoding

        Args:
            input: logits for all foreground classes [N, C]
                N is the number of anchors, and C is the number of foreground
                classes
            target: target classes. 0 is treated as background, >0 are
                treated as foreground classes. [N] is the number of anchors

        Returns:
            Tensor: final loss
        """
        target_one_hot = one_hot_smooth(
            target, num_classes=self.num_classes + 1, smoothing=self.smoothing)  # [N, C + 1]
        target_one_hot = target_one_hot[:, 1:]  # background is implicitly encoded

        return self.loss_weight * super().forward(input, target_one_hot.float())


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self,
                 *args,
                 loss_weight: float = 1.,
                 **kwargs,
                 ) -> None:
        """
        Same as CE from pytorch with additional loss weight for uniform API
        """
        super().__init__(*args, **kwargs)
        self.loss_weight = loss_weight

    def forward(self,
                input: Tensor,
                target: Tensor,
                ) -> Tensor:
        """
        Same as CE from pytorch
        """
        return self.loss_weight * super().forward(input, target)


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss - 专门为医学影像极度类别不平衡设计
    https://arxiv.org/abs/2009.14119
    
    特别适用于前列腺检测等任务：
    - 极度类别不平衡 (class0 vs class1-4)
    - 高假阳性问题
    - 小病灶检测
    """
    def __init__(self,
                 gamma_neg: float = 4,      # 负样本focusing (减少假阳性)
                 gamma_pos: float = 1,      # 正样本focusing (保持召回率)  
                 clip: float = 0.05,        # 概率裁剪 (处理困难负样本)
                 alpha: float = 0.25,       # 正负样本平衡
                 class_weights: Optional[torch.Tensor] = None,
                 reduction: str = "sum",
                 loss_weight: float = 1.0):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None
            
        logger.info(f"AsymmetricFocalLoss: γ_neg={gamma_neg}, γ_pos={gamma_pos}, "
                   f"clip={clip}, α={alpha}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Asymmetric Focal Loss
        
        Args:
            logits: 预测logits [N, C, spatial_dims]
            targets: 目标类别 [N, spatial_dims] (0=background, >0=foreground)
        """
        n_classes = logits.shape[1] + 1
        target_onehot = make_onehot_batch(targets, n_classes=n_classes).float()
        target_onehot = target_onehot[:, 1:]  # 移除背景类
        
        # 计算sigmoid概率
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # 概率裁剪 - 防止过度confident的负样本
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 计算基础损失
        los_pos = target_onehot * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - target_onehot) * torch.log(xs_neg.clamp(min=1e-8))
        
        # 不对称focusing - 正负样本使用不同的gamma
        loss = (los_pos * (1 - xs_pos) ** self.gamma_pos + 
                los_neg * xs_pos ** self.gamma_neg)
        
        # 应用alpha平衡
        alpha_t = self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)
        loss = alpha_t * loss
        loss = -loss.sum(dim=1)  # 对类别维度求和
        
        # 应用类别权重
        if self.class_weights is not None:
            # 获取每个样本的主要类别
            class_indices = torch.argmax(target_onehot, dim=1)
            sample_weights = self.class_weights[class_indices]
            loss = loss * sample_weights
            
        return self.loss_weight * reduction_helper(loss, self.reduction)
