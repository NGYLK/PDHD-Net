import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalTrueFusion(nn.Module):
    """
    极简但真正有效的CNN-Transformer融合策略
    - 解决假融合问题
    - 极低内存占用
    - 保证真正的特征交互
    """
    def __init__(self, cnn_channels, transformer_channels, out_channels, num_heads=2):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.transformer_channels = transformer_channels
        self.out_channels = out_channels
        
        # 极简特征投影
        self.cnn_proj = nn.Conv3d(cnn_channels, out_channels, 1, bias=False)
        self.transformer_proj = nn.Conv3d(transformer_channels, out_channels, 1, bias=False)
        
        # 极简交叉融合 - 避免大矩阵乘法
        self.cross_fusion = SimpleCrossFusion(out_channels)
        
        # 残差连接
        if cnn_channels != out_channels:
            self.residual_proj = nn.Conv3d(cnn_channels, out_channels, 1, bias=False)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, cnn_feat, transformer_feat):
        """
        极简但真正的融合
        """
        # 特征投影
        cnn_proj = self.cnn_proj(cnn_feat)
        transformer_proj = self.transformer_proj(transformer_feat)
        
        # 真正的交叉融合
        fused = self.cross_fusion(cnn_proj, transformer_proj)
        
        # 残差连接
        residual = self.residual_proj(cnn_feat)
        output = fused + residual
        
        return output

class SimpleCrossFusion(nn.Module):
    """
    极简交叉融合 - 避免大矩阵运算
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 通道级交叉注意力 - 内存友好
        self.cnn_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels//4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//4, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.transformer_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels//4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//4, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间级交叉融合
        self.spatial_fusion = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 1, bias=False)
        )
        
        # 最终融合权重
        self.fusion_weight = nn.Sequential(
            nn.Conv3d(channels * 2, 2, 1, bias=False),
            nn.Softmax(dim=1)
        )
    
    def forward(self, cnn_feat, transformer_feat):
        """
        真正的交叉融合 - 内存友好版本
        """
        # Step 1: 通道级交叉注意力
        # CNN特征受Transformer特征影响
        transformer_to_cnn_gate = self.cnn_gate(transformer_feat)
        cnn_enhanced = cnn_feat * transformer_to_cnn_gate
        
        # Transformer特征受CNN特征影响
        cnn_to_transformer_gate = self.transformer_gate(cnn_feat)
        transformer_enhanced = transformer_feat * cnn_to_transformer_gate
        
        # Step 2: 空间级融合
        combined = torch.cat([cnn_enhanced, transformer_enhanced], dim=1)
        spatial_fused = self.spatial_fusion(combined)
        
        # Step 3: 自适应加权融合
        fusion_weights = self.fusion_weight(torch.cat([cnn_enhanced, transformer_enhanced], dim=1))
        w1, w2 = fusion_weights[:, 0:1], fusion_weights[:, 1:2]
        
        # 最终输出
        output = cnn_enhanced * w1 + transformer_enhanced * w2 + spatial_fused * 0.1
        
        return output
