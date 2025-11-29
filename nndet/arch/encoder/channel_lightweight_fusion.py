import torch
import torch.nn as nn
import math

class ChannelWiseLightFusion(nn.Module):
    """
    通道级轻量融合 - 内存友好且效果更好
    基于ECA-Net和2024年CVPR最新研究
    - 解决假融合问题
    - 真正的特征交互
    - 极低内存消耗
    """
    def __init__(self, cnn_channels, trans_channels, out_channels):
        super().__init__()
        
        # 通道对齐 (1x1卷积，内存友好)
        self.cnn_align = nn.Conv3d(cnn_channels, out_channels, 1, bias=False)
        self.trans_align = nn.Conv3d(trans_channels, out_channels, 1, bias=False)
        
        # 高效通道注意力 (ECA-Net变体)
        self.eca_cnn = EfficientChannelAttention(out_channels)
        self.eca_trans = EfficientChannelAttention(out_channels)
        
        # 轻量级空间融合 - 权重为1.0，不是0.1！
        self.spatial_fusion = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, 3, padding=1, 
                     groups=max(1, out_channels//4), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差连接投影
        if cnn_channels != out_channels:
            self.residual_proj = nn.Conv3d(cnn_channels, out_channels, 1, bias=False)
        else:
            self.residual_proj = nn.Identity()
        
    def forward(self, cnn_feat, trans_feat):
        """
        真正的轻量级融合 - 解决假融合问题
        """
        # 通道对齐
        cnn_aligned = self.cnn_align(cnn_feat)
        trans_aligned = self.trans_align(trans_feat)
        
        # 高效通道注意力增强
        cnn_enhanced = self.eca_cnn(cnn_aligned)
        trans_enhanced = self.eca_trans(trans_aligned)
        
        # 真正的空间级融合 (权重1.0，不是0.1)
        combined = torch.cat([cnn_enhanced, trans_enhanced], dim=1)
        spatial_fused = self.spatial_fusion(combined)
        
        # 残差连接
        residual = self.residual_proj(cnn_feat)
        output = spatial_fused + residual
        
        return output

class EfficientChannelAttention(nn.Module):
    """
    高效通道注意力 - 基于ECA-Net
    - 无需大矩阵运算
    - 自适应卷积核大小
    - 保留更多通道信息
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        
        # 自适应卷积核大小计算
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 全局平均池化
        y = self.avg_pool(x)  # [B, C, 1, 1, 1]
        
        # 1D卷积进行通道交互
        y = y.squeeze(-1).squeeze(-1).squeeze(-1)  # [B, C]
        y = self.conv(y.unsqueeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1)
        
        # Sigmoid激活
        y = self.sigmoid(y).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        return x * y

class CrossModalChannelFusion(nn.Module):
    """
    跨模态通道融合 - 进一步增强版本
    如果基础版本效果好，可以尝试这个
    """
    def __init__(self, cnn_channels, trans_channels, out_channels):
        super().__init__()
        
        # 通道对齐
        self.cnn_align = nn.Conv3d(cnn_channels, out_channels, 1, bias=False)
        self.trans_align = nn.Conv3d(trans_channels, out_channels, 1, bias=False)
        
        # 跨模态通道注意力
        self.cross_channel_attn = CrossModalChannelAttention(out_channels)
        
        # 空间融合
        self.spatial_fusion = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差连接
        if cnn_channels != out_channels:
            self.residual_proj = nn.Conv3d(cnn_channels, out_channels, 1, bias=False)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, cnn_feat, trans_feat):
        # 通道对齐
        cnn_aligned = self.cnn_align(cnn_feat)
        trans_aligned = self.trans_align(trans_feat)
        
        # 跨模态通道注意力
        cnn_enhanced, trans_enhanced = self.cross_channel_attn(cnn_aligned, trans_aligned)
        
        # 空间融合
        combined = torch.cat([cnn_enhanced, trans_enhanced], dim=1)
        spatial_fused = self.spatial_fusion(combined)
        
        # 残差连接
        residual = self.residual_proj(cnn_feat)
        output = spatial_fused + residual
        
        return output

class CrossModalChannelAttention(nn.Module):
    """
    跨模态通道注意力 - CNN和Transformer特征互相增强
    """
    def __init__(self, channels):
        super().__init__()
        
        # CNN -> Transformer 通道注意力
        self.cnn_to_trans_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels//8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//8, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Transformer -> CNN 通道注意力
        self.trans_to_cnn_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels//8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//8, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, cnn_feat, trans_feat):
        # CNN特征受Transformer影响
        trans_to_cnn_weight = self.trans_to_cnn_attn(trans_feat)
        cnn_enhanced = cnn_feat * trans_to_cnn_weight
        
        # Transformer特征受CNN影响
        cnn_to_trans_weight = self.cnn_to_trans_attn(cnn_feat)
        trans_enhanced = trans_feat * cnn_to_trans_weight
        
        return cnn_enhanced, trans_enhanced


