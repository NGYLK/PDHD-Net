"""
Modern Classification Loss Functions for nnDetection
2025年更新的先进损失函数，特别适用于医学影像类别不平衡问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional

from nndet.losses.base import reduction_helper


class PolyLoss(nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    https://arxiv.org/abs/2204.12511 (2022)
    
    相比Focal Loss的优势：
    1. 收敛更稳定
    2. 对超参数不敏感  
    3. 在医学影像等不平衡数据上表现更好
    """
    def __init__(self,
                 epsilon: float = 1.0,
                 alpha: float = 1.0,
                 reduction: str = "mean",
                 loss_weight: float = 1.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.label_smoothing = label_smoothing
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            input: 预测logits [N, C]
            target: 目标类别 [N]
        """
        # 标签平滑
        if self.label_smoothing > 0:
            num_classes = input.size(-1)
            target_smooth = torch.zeros_like(input)
            target_smooth.fill_(self.label_smoothing / (num_classes - 1))
            target_smooth.scatter_(1, target.unsqueeze(1).long(), 1.0 - self.label_smoothing)
            ce_loss = -target_smooth * F.log_softmax(input, dim=1)
            ce_loss = ce_loss.sum(dim=1)
        else:
            ce_loss = F.cross_entropy(input, target.long(), reduction='none')
            
        # PolyLoss计算
        pt = torch.exp(-ce_loss)
        poly_loss = self.alpha * (self.epsilon + 1.0 - pt)
        
        return self.loss_weight * reduction_helper(poly_loss, self.reduction)


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Loss For Multi-Label Classification
    https://arxiv.org/abs/2009.14119 (2021)
    
    专门为极度类别不平衡设计，在医学影像检测中表现卓越
    """
    def __init__(self,
                 gamma_neg: float = 4,
                 gamma_pos: float = 1, 
                 clip: float = 0.05,
                 alpha: float = 1.0,
                 reduction: str = "mean",
                 loss_weight: float = 1.0):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            input: 预测logits [N, C]  
            target: 目标类别 [N]
        """
        # 转换为one-hot
        num_classes = input.size(-1)
        target_one_hot = torch.zeros_like(input)
        target_one_hot.scatter_(1, target.unsqueeze(1).long(), 1.0)
        
        # 计算概率
        xs_pos = torch.sigmoid(input)
        xs_neg = 1.0 - xs_pos

        # 概率裁剪
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 计算损失
        los_pos = target_one_hot * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - target_one_hot) * torch.log(xs_neg.clamp(min=1e-8))
        
        # 应用不对称的gamma
        loss = los_pos * (1 - xs_pos) ** self.gamma_pos + \
               los_neg * xs_pos ** self.gamma_neg
        
        loss = -self.alpha * loss.sum(dim=1)
        return self.loss_weight * reduction_helper(loss, self.reduction)


class AdaptiveFocalLoss(nn.Module):
    """
    自适应Focal Loss - 动态调整gamma和alpha参数
    根据训练过程中的类别表现自动调整权重
    """
    def __init__(self,
                 num_classes: int,
                 initial_gamma: float = 2.0,
                 initial_alpha: float = 0.25,
                 adaptation_rate: float = 0.01,
                 reduction: str = "sum",
                 loss_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.adaptation_rate = adaptation_rate
        self.reduction = reduction
        self.loss_weight = loss_weight
        
        # 可学习的参数
        self.gamma = nn.Parameter(torch.tensor(initial_gamma))
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        
        # 记录类别统计
        self.register_buffer('class_counts', torch.zeros(num_classes))
        self.register_buffer('class_errors', torch.zeros(num_classes))

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """自适应调整参数的Focal Loss"""
        # 更新类别统计
        if self.training:
            self._update_class_stats(input, target)
            
        # 计算标准Focal Loss
        ce_loss = F.cross_entropy(input, target.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 动态alpha计算
        alpha_t = torch.where(target == 0, 
                             1 - self.alpha,
                             self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return self.loss_weight * reduction_helper(focal_loss, self.reduction)
    
    def _update_class_stats(self, input: Tensor, target: Tensor):
        """更新类别统计信息"""
        with torch.no_grad():
            pred = torch.argmax(input, dim=1)
            for c in range(self.num_classes):
                mask = (target == c)
                if mask.sum() > 0:
                    self.class_counts[c] += mask.sum().float()
                    self.class_errors[c] += (pred[mask] != c).sum().float()


class CompoundLoss(nn.Module):
    """
    组合损失函数 - 结合多种损失的优势
    适用于复杂的医学影像多分类任务
    """
    def __init__(self,
                 num_classes: int,
                 focal_weight: float = 0.6,
                 poly_weight: float = 0.3, 
                 ce_weight: float = 0.1,
                 **kwargs):
        super().__init__()
        self.focal_loss = AsymmetricFocalLoss(**kwargs)
        self.poly_loss = PolyLoss(**kwargs)  
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        
        self.focal_weight = focal_weight
        self.poly_weight = poly_weight
        self.ce_weight = ce_weight
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """组合多种损失"""
        focal = self.focal_loss(input, target)
        poly = self.poly_loss(input, target)
        ce = self.ce_loss(input, target)
        
        total_loss = (self.focal_weight * focal + 
                     self.poly_weight * poly + 
                     self.ce_weight * ce)
        
        return total_loss

