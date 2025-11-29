"""
Modern Classifier Heads for nnDetection
2025年更新的先进分类器，集成最新损失函数
"""

import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor

from nndet.arch.heads.classifier import BaseClassifier
from nndet.losses.modern_classification import (
    PolyLoss,
    AsymmetricFocalLoss, 
    AdaptiveFocalLoss,
    CompoundLoss
)


class PolyClassifier(BaseClassifier):
    """
    使用PolyLoss的分类器
    
    PolyLoss优势：
    - 比Focal Loss收敛更稳定
    - 对超参数不敏感
    - 在医学影像不平衡数据上表现更好
    """
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 num_classes: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 prior_prob: Optional[float] = None,
                 epsilon: float = 1.0,
                 alpha: float = 1.0,
                 label_smoothing: float = 0.1,  # 添加标签平滑
                 reduction: str = "mean",
                 loss_weight: float = 1.,
                 **kwargs):
        """
        Args:
            epsilon: PolyLoss多项式系数 (推荐1.0-2.0)
            alpha: 损失缩放因子 (推荐0.5-2.0) 
            label_smoothing: 标签平滑参数 (推荐0.05-0.15)
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes + 1,  # 包含背景类
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            **kwargs,
        )

        self.loss = PolyLoss(
            epsilon=epsilon,
            alpha=alpha,
            label_smoothing=label_smoothing,
            reduction=reduction,
            loss_weight=loss_weight,
        )
        self.logits_convert_fn = nn.Softmax(dim=1)

    def box_logits_to_probs(self, box_logits: Tensor) -> Tensor:
        """转换为概率，移除背景类"""
        return self.logits_convert_fn(box_logits)[:, 1:]


class AsymmetricClassifier(BaseClassifier):
    """
    使用AsymmetricFocalLoss的分类器
    
    特别适合医学影像的极度类别不平衡：
    - 对正负样本使用不同的聚焦程度
    - 对稀有病灶类别有更好的敏感性
    - 在医学影像检测中表现卓越
    """
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 num_classes: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 prior_prob: Optional[float] = None,
                 gamma_neg: float = 4,      # 负样本聚焦度（更高=更关注困难负样本）
                 gamma_pos: float = 1,      # 正样本聚焦度
                 clip: float = 0.05,        # 概率裁剪
                 alpha: float = 1.0,
                 reduction: str = "mean",
                 loss_weight: float = 1.,
                 **kwargs):
        """
        Args:
            gamma_neg: 负样本聚焦度，推荐4-6（医学影像中负样本更多）
            gamma_pos: 正样本聚焦度，推荐0.5-2  
            clip: 概率裁剪，防止数值不稳定，推荐0.01-0.1
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            **kwargs,
        )

        self.loss = AsymmetricFocalLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            alpha=alpha,
            reduction=reduction,
            loss_weight=loss_weight,
        )
        self.logits_convert_fn = nn.Sigmoid()


class AdaptiveClassifier(BaseClassifier):
    """
    自适应分类器 - 训练过程中动态调整参数
    
    优势：
    - 根据训练进度自动调整损失函数参数
    - 适应数据分布的变化
    - 减少超参数调优工作量
    """
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 num_classes: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 prior_prob: Optional[float] = None,
                 initial_gamma: float = 2.0,
                 initial_alpha: float = 0.25,
                 adaptation_rate: float = 0.01,
                 reduction: str = "sum",
                 loss_weight: float = 1.,
                 **kwargs):
        """
        Args:
            initial_gamma: 初始gamma值
            initial_alpha: 初始alpha值
            adaptation_rate: 参数自适应学习率
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            **kwargs,
        )

        self.loss = AdaptiveFocalLoss(
            num_classes=num_classes,
            initial_gamma=initial_gamma,
            initial_alpha=initial_alpha,
            adaptation_rate=adaptation_rate,
            reduction=reduction,
            loss_weight=loss_weight,
        )
        self.logits_convert_fn = nn.Sigmoid()


class CompoundClassifier(BaseClassifier):
    """
    组合损失分类器 - 结合多种损失函数的优势
    
    适用场景：
    - 复杂的医学影像多分类任务
    - 需要同时考虑多种优化目标
    - 对单一损失函数效果不满意的情况
    """
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 num_classes: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 prior_prob: Optional[float] = None,
                 focal_weight: float = 0.6,    # Focal Loss权重
                 poly_weight: float = 0.3,     # Poly Loss权重
                 ce_weight: float = 0.1,       # CE Loss权重
                 **kwargs):
        """
        Args:
            focal_weight: AsymmetricFocal损失权重
            poly_weight: Poly损失权重  
            ce_weight: 交叉熵损失权重
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            **kwargs,
        )

        self.loss = CompoundLoss(
            num_classes=num_classes,
            focal_weight=focal_weight,
            poly_weight=poly_weight,
            ce_weight=ce_weight,
            **kwargs
        )
        self.logits_convert_fn = nn.Sigmoid()

