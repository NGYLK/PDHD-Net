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
import math
import torch.nn as nn

from typing import Optional, TypeVar
from torch import Tensor
from abc import abstractmethod
from loguru import logger

from nndet.losses.classification import (
    FocalLossWithLogits,
    BCEWithLogitsLossOneHot,
    CrossEntropyLoss,
)
import torch.nn.functional as F

CONV_TYPES = (nn.Conv2d, nn.Conv3d)


class Classifier(nn.Module):
    @abstractmethod
    def compute_loss(self, pred_logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """
        Compute classification loss (cross entropy loss)

        Args:
            pred_logits (Tensor): predicted logits
            targets (Tensor): classification targets

        Returns:
            Tensor: classification loss
        """
        raise NotImplementedError

    @abstractmethod
    def box_logits_to_probs(self, box_logits: Tensor) -> Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            box_logits (Tensor): bounding box logits [N, C], C=number of classes

        Returns:
            Tensor: probabilities
        """
        raise NotImplementedError


class BaseClassifier(Classifier):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 num_classes: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 **kwargs
                 ):
        """
        Base class to build classifier heads with typical conv structure
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                classifier
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            kwargs: keyword arguments passed to first and internal convolutions

        Notes:
            `self.loss` needs to be overwritten in subclasses
            `self.logits_convert_fn` needs to be overwritten in subclasses
        """
        super().__init__()
        self.dim = conv.dim
        self.num_levels = num_levels
        self.num_convs = num_convs

        self.num_classes = num_classes
        self.anchors_per_pos = anchors_per_pos

        self.in_channels = in_channels
        self.internal_channels = internal_channels

        self.conv_internal = self.build_conv_internal(conv, add_norm=add_norm, **kwargs)
        self.conv_out = self.build_conv_out(conv)

        self.loss: Optional[nn.Module] = None
        self.logits_convert_fn: Optional[nn.Module] = None
        self.init_weights()

    def build_conv_internal(self, conv, **kwargs):
        """
        Build internal convolutions
        """
        _conv_internal = nn.Sequential()
        _conv_internal.add_module(
            name="c_in",
            module=conv(
                self.in_channels,
                self.internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **kwargs,
            ))
        for i in range(self.num_convs):
            _conv_internal.add_module(
                name=f"c_internal{i}",
                module=conv(
                    self.internal_channels,
                    self.internal_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    **kwargs,
                ))
        return _conv_internal

    def build_conv_out(self, conv):
        """
        Build final convolutions
        """
        out_channels = self.num_classes * self.anchors_per_pos
        return conv(
            self.internal_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            add_norm=False,
            add_act=False,
            bias=True,
        )

    def forward(self,
                x: torch.Tensor,
                level: int,
                **kwargs,
                ) -> torch.Tensor:
        """
        Forward input

        Args:
            x (torch.Tensor): input feature map of size (N x C x Y x X x Z)

        Returns:
            torch.Tensor: classification logits for each anchor
                (N x anchors x num_classes)
        """
        class_logits = self.conv_out(self.conv_internal(x))

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        class_logits = class_logits.permute(*axes)
        class_logits = class_logits.contiguous()
        class_logits = class_logits.view(x.size()[0], -1, self.num_classes)
        return class_logits

    def compute_loss(self, pred_logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """
        Base classifier with cross entropy loss (in general hard negative
        example mining should be done before this)

        Args:
            pred_logits (Tensor): predicted logits
            targets (Tensor): classification targets

        Returns:
            Tensor: classification loss
        """
        return self.loss(pred_logits, targets.long(), **kwargs)

    def box_logits_to_probs(self, box_logits: Tensor) -> Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            box_logits (Tensor): bounding box logits [N, C]
                N = number of anchors, C=number of foreground classes

        Returns:
            Tensor: probabilities
        """
        return self.logits_convert_fn(box_logits)

    def init_weights(self) -> None:
        """
        Init weights with prior prob
        """
        if self.prior_prob is not None:
            logger.info(f"Init classifier weights: prior prob {self.prior_prob}")
            for layer in self.modules():
                if isinstance(layer, CONV_TYPES):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

            # Use prior in model initialization to improve stability
            bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
            for layer in self.conv_out.modules():
                if isinstance(layer, CONV_TYPES):
                    torch.nn.init.constant_(layer.bias, bias_value)
        else:
            logger.info("Init classifier weights: conv default")
  

class BCECLassifier(BaseClassifier):
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
                 weight: Optional[Tensor] = None,
                 reduction: str = "mean",
                 smoothing: float = 0.0,
                 loss_weight: float = 1.,
                 **kwargs
                 ):
        """
        Classifier Head with sigmoid based BCE loss computation and prio
        prob weight init
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                classifier
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            prior_prob: initialize final conv with given prior probability
            weight: weight in BCEWithLogitsLoss (see pytorch for more info)
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            smoothing:  label smoothing
            loss_weight: scalar to balance multiple losses
            kwargs: keyword arguments passed to first and internal convolutions
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

        self.loss = BCEWithLogitsLossOneHot(
            num_classes=num_classes,
            weight=weight,
            reduction=reduction,
            smoothing=smoothing,
            loss_weight=loss_weight,
            )
        self.logits_convert_fn = nn.Sigmoid()


class CEClassifier(BaseClassifier):
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
                weight: Optional[Tensor] = None,
                reduction: str = "mean",
                loss_weight: float = 1.,
                **kwargs
                ):
        """
        Classifier Head with sigmoid based BCE loss computation and prio
        prob weight init
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                classifier
            num_convs: number of convolutions
                input_conv -> num_convs -> output_convs
            add_norm: en-/disable normalization layers in internal layers
            prior_prob: initialize final conv with given prior probability
            weight: weight in cross entrpoy loss (see pytorch for more info)
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            kwargs: keyword arguments passed to first and internal convolutions
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes + 1, # add one channel for background
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            **kwargs,
            )

        self.loss = CrossEntropyLoss(
            weight=weight,
            reduction=reduction,
            loss_weight=loss_weight,
            )
        self.logits_convert_fn = nn.Softmax(dim=1)

    def box_logits_to_probs(self, box_logits: Tensor) -> Tensor:
        """
        Convert bounding box logits to probabilities

        Args:
            box_logits (Tensor): bounding box logits [N, C], C=number of classes

        Returns:
            Tensor: probabilities
        """
        return self.logits_convert_fn(box_logits)[:, 1:] # remove background predictions


class FocalClassifier(BaseClassifier):
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
                 gamma: float = 2,
                 alpha: float = -1,
                 class_frequencies: Optional[list] = None,  # 新增参数
                 reduction: str = "sum",
                 loss_weight: float = 1.,
                 **kwargs
                 ):
        """
        Enhanced Focal Classifier with class weighting support for class imbalance
        
        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            num_classes: number of foreground classes
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the classifier
            num_convs: number of convolutions
            add_norm: en-/disable normalization layers in internal layers
            prior_prob: initialize final conv with given prior probability
            gamma: focal loss gamma (2.0 recommended)
            alpha: focal loss alpha (0.25 recommended for class imbalance, -1 to disable)
            class_frequencies: List of class frequencies [70, 150, 140, 130, 120] for your data
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
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

        # 计算类别权重
        class_weights = None
        if class_frequencies is not None:
            total_samples = sum(class_frequencies)
            # 使用平方根平滑，避免权重过于极端
            class_weights = []
            for freq in class_frequencies:
                weight = (total_samples / freq) ** 0.5  # 平方根平滑
                class_weights.append(weight)
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            
            # 归一化并限制范围
            class_weights = class_weights / class_weights.mean()
            class_weights = torch.clamp(class_weights, min=0.5, max=2.0)
            
            logger.info(f"Enhanced Focal class weights: {class_weights.tolist()}")
        else:
            # 默认给class0轻微权重提升
            logger.info("Using default mild class weighting for medical detection")
            default_weights = [1.3] + [1.0] * (num_classes - 1)
            class_weights = torch.tensor(default_weights, dtype=torch.float32)
            logger.info(f"Default Focal class weights: {class_weights.tolist()}")

        self.loss = FocalLossWithLogits(
            gamma=gamma,
            alpha=alpha,
            class_weights=class_weights,  # 传递类别权重
            reduction=reduction,
            loss_weight=loss_weight,
            )
        self.logits_convert_fn = nn.Sigmoid()





class AsymmetricFocalClassifier(BaseClassifier):
    """
    分类器头 - 使用AsymmetricFocalLoss，专门优化医学影像检测
    
    特别适用于前列腺检测等极度类别不平衡的医学影像任务:
    - 大幅减少假阳性 (gamma_neg=4)
    - 保持高召回率 (gamma_pos=1) 
    - 优化小病灶检测
    """
    
    def __init__(self,
                 conv: nn.Module,
                 in_channels: int,
                 num_convs: int = 1,
                 add_norm: bool = True,
                 internal_channels: int = 128,
                 num_classes: int = 5,
                 anchors_per_pos: int = 27,
                 num_levels: int = 4,
                 prior_prob: float = 0.01,
                 gamma_neg: float = 4.0,        # 负样本focusing - 减少假阳性
                 gamma_pos: float = 1.0,        # 正样本focusing - 保持召回率
                 clip: float = 0.05,            # 概率裁剪 - 处理困难负样本
                 alpha: float = 0.25,           # 正负样本平衡
                 class_frequencies: Optional[list] = None,
                 reduction: str = "sum",
                 loss_weight: float = 2.0,      # 推荐更高的权重
                 **kwargs
                 ):
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

        # 计算类别权重
        class_weights = None
        if class_frequencies is not None:
            total_samples = sum(class_frequencies)
            class_weights = []
            for freq in class_frequencies:
                # 对class0 (最少样本) 使用更强的权重
                if freq == min(class_frequencies):
                    weight = (total_samples / freq) ** 0.7
                else:
                    weight = (total_samples / freq) ** 0.5
                class_weights.append(weight)
                
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            class_weights = class_weights / class_weights.mean()
            class_weights = torch.clamp(class_weights, min=0.3, max=3.0)
            
            logger.info(f"AsymmetricFocal class weights: {class_weights.tolist()}")
        else:
            logger.info("Using optimized class weighting for medical detection")
            default_weights = [2.0] + [1.0] * (num_classes - 1)
            class_weights = torch.tensor(default_weights, dtype=torch.float32)
            logger.info(f"Default AsymmetricFocal weights: {class_weights.tolist()}")

        from nndet.losses.classification import AsymmetricFocalLoss
        
        self.loss = AsymmetricFocalLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            alpha=alpha,
            class_weights=class_weights,
            reduction=reduction,
            loss_weight=loss_weight,
            )
        
        self.logits_convert_fn = nn.Sigmoid()

    def init_weights(self) -> None:
        logger.info(f"Init AsymmetricFocal classifier weights: prior prob {self.prior_prob}")
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in self.conv_out:
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.constant_(module.bias, bias_value)


ClassifierType = TypeVar('ClassifierType', bound=Classifier)
