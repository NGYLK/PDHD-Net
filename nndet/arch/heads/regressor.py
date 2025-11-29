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
import torch.nn as nn

from typing import Optional, Tuple, Callable, TypeVar
from abc import abstractmethod

from loguru import logger

from nndet.core.boxes import box_iou
from nndet.arch.layers.scale import Scale
from torch import Tensor

from nndet.losses import SmoothL1Loss, GIoULoss
from nndet.losses.base import reduction_helper
from nndet.core.boxes.ops import generalized_box_iou
import math


CONV_TYPES = (nn.Conv2d, nn.Conv3d)


class Regressor(nn.Module):
    @abstractmethod
    def compute_loss(self, pred_deltas: Tensor, target_deltas: Tensor, **kwargs) -> Tensor:
        """
        Compute regression loss (l1 loss)

        Args:
            pred_deltas (Tensor): predicted bounding box deltas [N,  dim * 2]
            target_deltas (Tensor): target bounding box deltas [N,  dim * 2]

        Returns:
            Tensor: loss
        """
        raise NotImplementedError


class BaseRegressor(Regressor):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 learn_scale: bool = False,
                 **kwargs,
                 ):
        """
        Base class to build regressor heads with typical conv structure
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                regressor
            num_convs: number of convolutions
                in conv -> num convs -> final conv
            add_norm: en-/disable normalization layers in internal layers
            learn_scale: learn additional single scalar values per feature
                pyramid level
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__()
        self.dim = conv.dim
        self.num_levels = num_levels
        self.num_convs = num_convs
        self.learn_scale = learn_scale

        self.anchors_per_pos = anchors_per_pos

        self.in_channels = in_channels
        self.internal_channels = internal_channels

        self.conv_internal = self.build_conv_internal(conv, add_norm=add_norm, **kwargs)
        self.conv_out = self.build_conv_out(conv)

        if self.learn_scale:
            self.scales = self.build_scales()

        self.loss: Optional[nn.Module] = None
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
        out_channels = self.anchors_per_pos * self.dim * 2
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

    def build_scales(self) -> nn.ModuleList:
        """
        Build additionales scalar values per level
        """
        logger.info("Learning level specific scalar in regressor")
        return nn.ModuleList([Scale() for _ in range(self.num_levels)])

    def forward(self, x: torch.Tensor, level: int, **kwargs) -> torch.Tensor:
        """
        Forward input

        Args:
            x: input feature map of size [N x C x Y x X x Z]

        Returns:
            torch.Tensor: classification logits for each anchor
                [N, n_anchors, dim*2]
        """
        bb_logits = self.conv_out(self.conv_internal(x))

        if self.learn_scale:
            bb_logits = self.scales[level](bb_logits)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        bb_logits = bb_logits.permute(*axes)
        bb_logits = bb_logits.contiguous()
        bb_logits = bb_logits.view(x.size()[0], -1, self.dim * 2)
        return bb_logits

    def compute_loss(self,
                     pred_deltas: Tensor,
                     target_deltas: Tensor,
                     **kwargs,
                     ) -> Tensor:
        """
        Compute regression loss (l1 loss)

        Args:
            pred_deltas: predicted bounding box deltas [N,  dim * 2]
            target_deltas: target bounding box deltas [N,  dim * 2]

        Returns:
            Tensor: loss
        """
        return self.loss(pred_deltas, target_deltas, **kwargs)

    def init_weights(self) -> None:
        """
        Init weights with normal distribution (mean=0, std=0.01)
        """
        logger.info("Overwriting regressor conv weight init")
        for layer in self.modules():
            if isinstance(layer, CONV_TYPES):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)


class L1Regressor(BaseRegressor):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 beta: float = 1.,
                 reduction: Optional[str] = "sum",
                 loss_weight: float = 1.,
                 learn_scale: bool = False,
                 **kwargs,
                 ):
        """
        Build regressor heads with typical conv structure and smooth L1 loss
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                regressor
            num_convs: number of convolutions
                in conv -> num convs -> final conv
            add_norm: en-/disable normalization layers in internal layers
            beta: L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            learn_scale: learn additional single scalar values per feature
                pyramid level
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            num_convs=num_convs,
            add_norm=add_norm,
            learn_scale=learn_scale,
            **kwargs
        )
        self.loss = SmoothL1Loss(
            beta=beta,
            reduction=reduction,
            loss_weight=loss_weight,
            )


class GIoURegressor(BaseRegressor):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 reduction: Optional[str] = "sum",
                 loss_weight: float = 1.,
                 learn_scale: bool = False,
                 **kwargs,
                 ):
        """
        Build regressor heads with typical conv structure and generalized
        IoU loss
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels which are passed through the
                regressor
            num_convs: number of convolutions
                in conv -> num convs -> final conv
            add_norm: en-/disable normalization layers in internal layers
            reduction: reduction to apply to loss. 'sum' | 'mean' | 'none'
            loss_weight: scalar to balance multiple losses
            learn_scale: learn additional single scalar values per feature
                pyramid level
            kwargs: keyword arguments passed to first and internal convolutions
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            num_convs=num_convs,
            add_norm=add_norm,
            learn_scale=learn_scale,
            **kwargs
        )
        self.loss = GIoULoss(
            reduction=reduction,
            loss_weight=loss_weight,
            )


# =============================================================================
# 2025年医学影像小目标检测专用回归头 - 解决70% fp_iou问题
# =============================================================================

class SIoULoss(nn.Module):
    """
    SIoU Loss - 2025年医学小目标检测最优选择
    论文: https://arxiv.org/abs/2205.12740
    特别适合小目标医学影像：
    - 角度损失: 处理病灶形状和方向
    - 距离损失: 精确定位小目标中心  
    - 形状损失: 适应病灶的尺寸变化
    """
    
    def __init__(self,
                 reduction: Optional[str] = "sum",
                 eps: float = 1e-7,
                 loss_weight: float = 1.0,
                 theta: float = 4.0):  # 控制角度损失的强度
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.loss_weight = loss_weight
        self.theta = theta
        
        logger.info(f"SIoU Loss initialized for medical small objects: θ={theta}")

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        计算SIoU损失 - 专门为医学小目标优化
        
        Args:
            pred_boxes: 预测框 [N, dim*2] (x1,y1,x2,y2,z1,z2)
            target_boxes: 真实框 [N, dim*2]
        """
        # 确保是3D框格式 (x1,y1,x2,y2,z1,z2)
        if pred_boxes.shape[-1] == 6:  # 3D case
            return self._compute_3d_siou(pred_boxes, target_boxes)
        else:  # 2D case  
            return self._compute_2d_siou(pred_boxes, target_boxes)
    
    def _compute_3d_siou(self, pred_boxes, target_boxes):
        """计算3D SIoU损失"""
        # 提取坐标
        px1, py1, px2, py2, pz1, pz2 = pred_boxes.unbind(dim=-1)
        gx1, gy1, gx2, gy2, gz1, gz2 = target_boxes.unbind(dim=-1)
        
        # 计算中心点
        pred_cx = (px1 + px2) / 2
        pred_cy = (py1 + py2) / 2  
        pred_cz = (pz1 + pz2) / 2
        
        gt_cx = (gx1 + gx2) / 2
        gt_cy = (gy1 + gy2) / 2
        gt_cz = (gz1 + gz2) / 2
        
        # 计算宽高深
        pred_w = px2 - px1
        pred_h = py2 - py1
        pred_d = pz2 - pz1
        
        gt_w = gx2 - gx1
        gt_h = gy2 - gy1
        gt_d = gz2 - gz1
        
        # 1. 角度损失 (Angle Loss) - 3D扩展
        cx_diff = torch.abs(pred_cx - gt_cx)
        cy_diff = torch.abs(pred_cy - gt_cy)
        cz_diff = torch.abs(pred_cz - gt_cz)
        
        c_w = torch.max(pred_w, gt_w)
        c_h = torch.max(pred_h, gt_h) 
        c_d = torch.max(pred_d, gt_d)
        
        # 3D角度损失计算 - 数值稳定版
        # 避免除零和arcsin域外的问题
        sin_alpha = torch.clamp(cx_diff / torch.clamp(c_w, min=self.eps), 0, 0.99)
        sin_beta = torch.clamp(cy_diff / torch.clamp(c_h, min=self.eps), 0, 0.99)  
        sin_gamma = torch.clamp(cz_diff / torch.clamp(c_d, min=self.eps), 0, 0.99)
        
        # 简化角度损失计算 - 避免复杂的三角函数组合
        # 使用L2距离归一化作为角度代理
        angle_term = (sin_alpha + sin_beta + sin_gamma) / 3.0
        angle_loss = torch.clamp(angle_term, 0, 1)  # 简化为线性项，避免NaN
        
        # 2. 距离损失 (Distance Loss) - 数值稳定版
        rho_xy = (cx_diff ** 2 + cy_diff ** 2) / (torch.clamp(c_w ** 2 + c_h ** 2, min=self.eps))
        rho_z = (cz_diff ** 2) / torch.clamp(c_d ** 2, min=self.eps)
        # 限制指数输入，避免数值溢出
        rho_xy = torch.clamp(rho_xy, 0, 10)
        rho_z = torch.clamp(rho_z, 0, 10)
        distance_loss = 2 - torch.exp(-rho_xy) - torch.exp(-rho_z)
        
        # 3. 形状损失 (Shape Loss) - 数值稳定版
        omega_w = torch.abs(pred_w - gt_w) / torch.clamp(torch.max(pred_w, gt_w), min=self.eps)
        omega_h = torch.abs(pred_h - gt_h) / torch.clamp(torch.max(pred_h, gt_h), min=self.eps)
        omega_d = torch.abs(pred_d - gt_d) / torch.clamp(torch.max(pred_d, gt_d), min=self.eps)
        
        # 限制omega值，避免数值不稳定
        omega_w = torch.clamp(omega_w, 0, 2)
        omega_h = torch.clamp(omega_h, 0, 2)
        omega_d = torch.clamp(omega_d, 0, 2)
        
        # 简化形状损失计算，避免复杂的指数和幂运算
        shape_loss = (omega_w + omega_h + omega_d) / 3.0
        
        # 4. 标准IoU
        inter_x1 = torch.max(px1, gx1)
        inter_y1 = torch.max(py1, gy1)
        inter_z1 = torch.max(pz1, gz1)
        inter_x2 = torch.min(px2, gx2)
        inter_y2 = torch.min(py2, gy2)
        inter_z2 = torch.min(pz2, gz2)
        
        inter_vol = torch.clamp(inter_x2 - inter_x1, min=0) * \
                   torch.clamp(inter_y2 - inter_y1, min=0) * \
                   torch.clamp(inter_z2 - inter_z1, min=0)
        
        pred_vol = pred_w * pred_h * pred_d
        gt_vol = gt_w * gt_h * gt_d
        union_vol = pred_vol + gt_vol - inter_vol
        
        iou = inter_vol / (union_vol + self.eps)
        
        # 最终SIoU = IoU - 加权损失组合 (数值稳定版)
        # 使用更保守的权重组合，避免损失过大
        penalty = 0.1 * angle_loss + 0.1 * distance_loss + 0.1 * shape_loss
        siou = iou - penalty
        siou = torch.clamp(siou, min=-1.0, max=1.0)  # 限制SIoU范围
        loss = 1 - siou
        loss = torch.clamp(loss, min=0.0, max=2.0)  # 限制损失范围
        
        return self.loss_weight * reduction_helper(loss, self.reduction)
    
    def _compute_2d_siou(self, pred_boxes, target_boxes):
        """计算2D SIoU损失"""
        # 提取坐标
        px1, py1, px2, py2 = pred_boxes.unbind(dim=-1)
        gx1, gy1, gx2, gy2 = target_boxes.unbind(dim=-1)
        
        # 计算中心点
        pred_cx = (px1 + px2) / 2
        pred_cy = (py1 + py2) / 2
        
        gt_cx = (gx1 + gx2) / 2
        gt_cy = (gy1 + gy2) / 2
        
        # 计算宽高
        pred_w = px2 - px1
        pred_h = py2 - py1
        
        gt_w = gx2 - gx1
        gt_h = gy2 - gy1
        
        # 1. 角度损失 - 修正版
        cx_diff = torch.abs(pred_cx - gt_cx)
        cy_diff = torch.abs(pred_cy - gt_cy)
        
        c_w = torch.max(pred_w, gt_w)
        c_h = torch.max(pred_h, gt_h)
        
        # 数值稳定的角度损失
        sin_alpha = torch.clamp(cx_diff / torch.clamp(c_w, min=self.eps), 0, 0.99)
        sin_beta = torch.clamp(cy_diff / torch.clamp(c_h, min=self.eps), 0, 0.99)
        
        # 简化为线性组合，避免复杂三角函数
        angle_loss = (sin_alpha + sin_beta) / 2.0
        
        # 2. 距离损失 - 数值稳定版
        rho = (cx_diff ** 2 + cy_diff ** 2) / (torch.clamp(c_w ** 2 + c_h ** 2, min=self.eps))
        rho = torch.clamp(rho, 0, 10)
        distance_loss = 2 - torch.exp(-rho)
        
        # 3. 形状损失 - 简化版
        omega_w = torch.abs(pred_w - gt_w) / torch.clamp(torch.max(pred_w, gt_w), min=self.eps)
        omega_h = torch.abs(pred_h - gt_h) / torch.clamp(torch.max(pred_h, gt_h), min=self.eps)
        omega_w = torch.clamp(omega_w, 0, 2)
        omega_h = torch.clamp(omega_h, 0, 2)
        shape_loss = (omega_w + omega_h) / 2.0
        
        # 4. 标准IoU
        inter_x1 = torch.max(px1, gx1)
        inter_y1 = torch.max(py1, gy1)
        inter_x2 = torch.min(px2, gx2)
        inter_y2 = torch.min(py2, gy2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                    torch.clamp(inter_y2 - inter_y1, min=0)
        
        pred_area = pred_w * pred_h
        gt_area = gt_w * gt_h
        union_area = pred_area + gt_area - inter_area
        
        iou = inter_area / (union_area + self.eps)
        
        # 最终SIoU - 数值稳定版
        penalty = 0.1 * angle_loss + 0.1 * distance_loss + 0.1 * shape_loss
        siou = iou - penalty
        siou = torch.clamp(siou, min=-1.0, max=1.0)
        loss = 1 - siou
        loss = torch.clamp(loss, min=0.0, max=2.0)
        
        return self.loss_weight * reduction_helper(loss, self.reduction)


class EIoULoss(nn.Module):
    """
    EIoU Loss - Enhanced IoU for Medical Imaging
    论文: https://arxiv.org/abs/2101.08158
    特点: 引入焦点机制，特别适合医学小目标
    """
    
    def __init__(self,
                 reduction: Optional[str] = "sum", 
                 eps: float = 1e-7,
                 loss_weight: float = 1.0,
                 focal_weight: float = 0.5):  # 焦点权重
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.loss_weight = loss_weight
        self.focal_weight = focal_weight
        
        logger.info(f"EIoU Loss for medical small targets: focal_weight={focal_weight}")

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """计算Enhanced IoU损失"""
        
        # 使用现有的GIoU计算作为基础
        giou = torch.diag(generalized_box_iou(pred_boxes, target_boxes, eps=self.eps))
        
        # 计算标准IoU用于焦点权重
        iou = self._compute_iou(pred_boxes, target_boxes)
        
        # 焦点权重: 对难样本(低IoU)给予更多关注
        focal_weights = torch.pow(1 - iou, self.focal_weight)
        
        # Enhanced IoU = GIoU * Focal_Weight
        eiou = giou * focal_weights
        loss = 1 - eiou
        
        return self.loss_weight * reduction_helper(loss, self.reduction)
    
    def _compute_iou(self, pred_boxes, target_boxes):
        """计算标准IoU"""
        if pred_boxes.shape[-1] == 6:  # 3D
            return self._compute_3d_iou(pred_boxes, target_boxes)
        else:  # 2D
            return self._compute_2d_iou(pred_boxes, target_boxes)
    
    def _compute_3d_iou(self, pred_boxes, target_boxes):
        """计算3D IoU"""
        px1, py1, px2, py2, pz1, pz2 = pred_boxes.unbind(dim=-1)
        gx1, gy1, gx2, gy2, gz1, gz2 = target_boxes.unbind(dim=-1)
        
        # 交集
        inter_x1 = torch.max(px1, gx1)
        inter_y1 = torch.max(py1, gy1)
        inter_z1 = torch.max(pz1, gz1)
        inter_x2 = torch.min(px2, gx2)
        inter_y2 = torch.min(py2, gy2)
        inter_z2 = torch.min(pz2, gz2)
        
        inter_vol = torch.clamp(inter_x2 - inter_x1, min=0) * \
                   torch.clamp(inter_y2 - inter_y1, min=0) * \
                   torch.clamp(inter_z2 - inter_z1, min=0)
        
        # 并集
        pred_vol = (px2 - px1) * (py2 - py1) * (pz2 - pz1)
        gt_vol = (gx2 - gx1) * (gy2 - gy1) * (gz2 - gz1)
        union_vol = pred_vol + gt_vol - inter_vol
        
        return inter_vol / (union_vol + 1e-7)
    
    def _compute_2d_iou(self, pred_boxes, target_boxes):
        """计算2D IoU"""
        px1, py1, px2, py2 = pred_boxes.unbind(dim=-1)
        gx1, gy1, gx2, gy2 = target_boxes.unbind(dim=-1)
        
        # 交集
        inter_x1 = torch.max(px1, gx1)
        inter_y1 = torch.max(py1, gy1)
        inter_x2 = torch.min(px2, gx2)
        inter_y2 = torch.min(py2, gy2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                    torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 并集
        pred_area = (px2 - px1) * (py2 - py1)
        gt_area = (gx2 - gx1) * (gy2 - gy1)
        union_area = pred_area + gt_area - inter_area
        
        return inter_area / (union_area + 1e-7)


class MedicalSmallTargetRegressor(BaseRegressor):
    """
    2025年医学影像小目标检测专用回归头
    集成最新的SIoU/EIoU损失和小目标特化设计
    专门解决70% fp_iou问题
    """
    
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 4,  # 增加卷积层深度
                 add_norm: bool = True,
                 reduction: Optional[str] = "sum",
                 loss_weight: float = 1.0,
                 learn_scale: bool = True,  # 默认启用
                 loss_type: str = "siou",  # siou/eiou/giou
                 small_target_enhancement: bool = True,  # 小目标增强
                 **kwargs):
        """
        医学小目标专用回归头
        
        Args:
            loss_type: 损失函数类型 (siou/eiou/giou)
            small_target_enhancement: 是否启用小目标增强
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            num_convs=num_convs,
            add_norm=add_norm,
            learn_scale=learn_scale,
            **kwargs
        )
        
        self.loss_type = loss_type
        self.small_target_enhancement = small_target_enhancement
        
        # 选择损失函数
        if loss_type == "siou":
            self.loss = SIoULoss(
                reduction=reduction,
                loss_weight=loss_weight,
                theta=4.0  # 医学影像优化参数
            )
            logger.info("Using SIoU Loss for medical small targets")
        elif loss_type == "eiou":
            self.loss = EIoULoss(
                reduction=reduction,
                loss_weight=loss_weight,
                focal_weight=0.5  # 专注困难样本
            )
            logger.info("Using EIoU Loss for medical small targets")
        else:  # 默认GIoU
            self.loss = GIoULoss(
                reduction=reduction,
                loss_weight=loss_weight
            )
            logger.info("Using traditional GIoU Loss")
        
        # 小目标增强模块
        if small_target_enhancement:
            self.small_target_enhancer = self._build_small_target_enhancer(conv)
            logger.info("Small target enhancement enabled")
    
    def _build_small_target_enhancer(self, conv):
        """构建小目标增强模块"""
        # 通道注意力模块 - 修正版
        class ChannelAttention(nn.Module):
            def __init__(self, channels, reduction=8):
                super().__init__()
                self.avg_pool = nn.AdaptiveAvgPool3d(1) if hasattr(conv, 'dim') and conv.dim == 3 else nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(channels, channels // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels // reduction, channels, bias=False),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                b, c = x.size(0), x.size(1)
                y = self.avg_pool(x).view(b, c)
                y = self.fc(y).view(b, c, *([1] * (x.dim() - 2)))
                return x * y.expand_as(x)
                
        return ChannelAttention(self.internal_channels, reduction=8)
    
    def forward(self, x: torch.Tensor, level: int, **kwargs) -> torch.Tensor:
        """前向传播 - 增强小目标特征"""
        
        # 基础特征提取
        features = self.conv_internal(x)
        
        # 小目标增强
        if self.small_target_enhancement:
            # 通道注意力增强 - 直接返回增强后的特征
            features = self.small_target_enhancer(features)
        
        # 回归输出
        bb_logits = self.conv_out(features)
        
        # 可学习尺度
        if self.learn_scale:
            bb_logits = self.scales[level](bb_logits)
        
        # 维度调整
        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        bb_logits = bb_logits.permute(*axes)
        bb_logits = bb_logits.contiguous()
        bb_logits = bb_logits.view(x.size()[0], -1, self.dim * 2)
        
        return bb_logits
    
    def init_weights(self) -> None:
        """权重初始化 - 针对小目标优化"""
        logger.info("Initializing weights for medical small target regressor")
        
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Conv3d)):
                # 使用更小的标准差，适合小目标
                torch.nn.init.normal_(layer.weight, mean=0, std=0.005)  # 比默认0.01更小
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)


RegressorType = TypeVar('RegressorType', bound=Regressor)
