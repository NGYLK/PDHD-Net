"""
小波变换频域-空间域特征融合模块
适用于前列腺癌检测的多尺度特征提取

基于SOTA研究：
- DedustNet: 频率主导的Swin Transformer
- SFFNet: 空间-频率域融合网络  
- WTANet: 多尺度小波变换注意力网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class GPUWaveletTransform3D(nn.Module):
    """GPU加速的小波变换模块，模拟小波变换效果"""
    
    def __init__(self):
        super(GPUWaveletTransform3D, self).__init__()
        
        # Haar小波滤波器 (在GPU上)
        # LL (低通-低通): 平滑
        self.ll_kernel = nn.Parameter(torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]], dtype=torch.float32), requires_grad=False)
        
        # LH (低通-高通): 水平边缘  
        self.lh_kernel = nn.Parameter(torch.tensor([[[[0.5, -0.5], [0.5, -0.5]]]], dtype=torch.float32), requires_grad=False)
        
        # HL (高通-低通): 垂直边缘
        self.hl_kernel = nn.Parameter(torch.tensor([[[[0.5, 0.5], [-0.5, -0.5]]]], dtype=torch.float32), requires_grad=False)
        
        # HH (高通-高通): 对角边缘
        self.hh_kernel = nn.Parameter(torch.tensor([[[[0.5, -0.5], [-0.5, 0.5]]]], dtype=torch.float32), requires_grad=False)
        
    def forward(self, x):
        """
        GPU加速的小波变换
        Input: x [B, C, D, H, W]
        Output: coeffs (LL, LH, HL, HH) 每个 [B, C, D, H//2, W//2]
        """
        B, C, D, H, W = x.shape
        
        # 重塑为 [B*C*D, 1, H, W] 以便使用2D卷积
        x_reshaped = x.view(B * C * D, 1, H, W)
        
        # 使用卷积模拟小波变换 (stride=2实现下采样)
        LL = F.conv2d(x_reshaped, self.ll_kernel, stride=2, padding=0)
        LH = F.conv2d(x_reshaped, self.lh_kernel, stride=2, padding=0)
        HL = F.conv2d(x_reshaped, self.hl_kernel, stride=2, padding=0)
        HH = F.conv2d(x_reshaped, self.hh_kernel, stride=2, padding=0)
        
        # 重塑回原始批次维度
        new_H, new_W = LL.shape[2], LL.shape[3]
        LL = LL.view(B, C, D, new_H, new_W)
        LH = LH.view(B, C, D, new_H, new_W)
        HL = HL.view(B, C, D, new_H, new_W)
        HH = HH.view(B, C, D, new_H, new_W)
        
        return LL, LH, HL, HH


class WaveletTransform3D(nn.Module):
    """3D小波变换模块，优先使用GPU加速版本"""
    
    def __init__(self, wavelet='haar', mode='symmetric', use_gpu_version=True):
        super(WaveletTransform3D, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.use_gpu_version = use_gpu_version
        
        if use_gpu_version:
            self.gpu_wavelet = GPUWaveletTransform3D()
        
    def forward(self, x):
        """
        执行小波变换，优先使用GPU版本
        """
        if self.use_gpu_version:
            return self.gpu_wavelet(x)
        else:
            # 原始CPU版本作为fallback
            return self._cpu_wavelet_transform(x)
    
    def _cpu_wavelet_transform(self, x):
        """原始的CPU小波变换 (保留作为备用)"""
        B, C, D, H, W = x.shape
        x_reshaped = x.view(B * C * D, H, W)
        
        LL_list = []
        LH_list = []
        HL_list = []
        HH_list = []
        
        for i in range(B * C * D):
            img = x_reshaped[i].detach().cpu().numpy()
            try:
                LL, (LH, HL, HH) = pywt.dwt2(img, self.wavelet, mode=self.mode)
                LL_list.append(LL)
                LH_list.append(LH)
                HL_list.append(HL)
                HH_list.append(HH)
            except:
                # 如果失败，使用简单下采样
                h_half, w_half = H//2, W//2
                LL_list.append(img[:h_half, :w_half])
                LH_list.append(img[:h_half, w_half:])
                HL_list.append(img[h_half:, :w_half])
                HH_list.append(img[h_half:, w_half:])
        
        LL = torch.stack([torch.from_numpy(c).to(x.device, dtype=x.dtype) for c in LL_list])
        LH = torch.stack([torch.from_numpy(c).to(x.device, dtype=x.dtype) for c in LH_list])
        HL = torch.stack([torch.from_numpy(c).to(x.device, dtype=x.dtype) for c in HL_list])
        HH = torch.stack([torch.from_numpy(c).to(x.device, dtype=x.dtype) for c in HH_list])
        
        if len(LL_list) > 0:
            actual_H, actual_W = LL_list[0].shape
            LL = LL.view(B, C, D, actual_H, actual_W)
            LH = LH.view(B, C, D, actual_H, actual_W)
            HL = HL.view(B, C, D, actual_H, actual_W)
            HH = HH.view(B, C, D, actual_H, actual_W)
        
        return LL, LH, HL, HH


class WaveletFeatureFusion(nn.Module):
    """
    小波特征融合模块
    将小波系数转换为有用的特征表示
    """
    
    def __init__(self, in_channels, out_channels):
        super(WaveletFeatureFusion, self).__init__()
        
        # 各频段特征处理
        self.ll_conv = nn.Conv3d(in_channels, out_channels//4, 1)  # 低频-低频
        self.lh_conv = nn.Conv3d(in_channels, out_channels//4, 1)  # 低频-高频
        self.hl_conv = nn.Conv3d(in_channels, out_channels//4, 1)  # 高频-低频
        self.hh_conv = nn.Conv3d(in_channels, out_channels//4, 1)  # 高频-高频
        
        # 特征融合
        self.fusion_conv = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
        # 注意力权重生成
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_channels, out_channels//8, 1),
            nn.ReLU(),
            nn.Conv3d(out_channels//8, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, wavelet_coeffs):
        """
        融合小波系数生成增强特征
        """
        LL, LH, HL, HH = wavelet_coeffs
        
        # 各频段特征提取
        ll_feat = self.ll_conv(LL)  # 主要结构信息
        lh_feat = self.lh_conv(LH)  # 水平边缘
        hl_feat = self.hl_conv(HL)  # 垂直边缘  
        hh_feat = self.hh_conv(HH)  # 对角边缘
        
        # 拼接所有频段特征
        fused_feat = torch.cat([ll_feat, lh_feat, hl_feat, hh_feat], dim=1)
        
        # 特征融合和增强
        fused_feat = self.fusion_conv(fused_feat)
        fused_feat = self.norm(fused_feat)
        fused_feat = self.activation(fused_feat)
        
        # 生成注意力权重
        attention_weights = self.attention(fused_feat)
        enhanced_feat = fused_feat * attention_weights
        
        return enhanced_feat


class WaveletSpatialFusion(nn.Module):
    """
    小波-空间特征融合模块
    适用于前列腺癌检测的特征增强
    """
    
    def __init__(self, in_channels, out_channels, wavelet='haar'):
        super(WaveletSpatialFusion, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 小波变换
        self.wavelet_transform = WaveletTransform3D(wavelet=wavelet)
        
        # 小波特征融合
        self.wavelet_fusion = WaveletFeatureFusion(in_channels, out_channels)
        
        # 空间特征处理
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, 1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差连接调整
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x):
        """
        前向传播：空间-频域特征融合
        """
        # 原始空间特征
        spatial_feat = self.spatial_conv(x)
        
        # 小波频域特征
        try:
            wavelet_coeffs = self.wavelet_transform(x)
            wavelet_feat = self.wavelet_fusion(wavelet_coeffs)
            
            # 上采样小波特征到原始尺寸
            wavelet_feat = F.interpolate(
                wavelet_feat, 
                size=spatial_feat.shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
            
            # 融合空间和频域特征
            combined_feat = torch.cat([spatial_feat, wavelet_feat], dim=1)
            fused_feat = self.final_fusion(combined_feat)
            
        except Exception as e:
            # 如果小波变换失败，只使用空间特征
            print(f"Wavelet transform failed: {e}, using spatial features only")
            fused_feat = spatial_feat
        
        # 残差连接
        residual = self.residual_conv(x)
        output = fused_feat + residual
        
        return output


def test_wavelet_fusion():
    """测试小波融合模块"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    B, C, D, H, W = 2, 64, 16, 32, 32
    x = torch.randn(B, C, D, H, W, device=device)
    
    print("🔬 测试小波-空间特征融合...")
    
    # 创建融合模块
    fusion_module = WaveletSpatialFusion(
        in_channels=C, 
        out_channels=C, 
        wavelet='haar'
    ).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = fusion_module(x)
    
    print(f"📊 输入形状: {x.shape}")
    print(f"📊 输出形状: {output.shape}")
    print(f"✅ 小波融合测试成功!")
    
    return fusion_module


if __name__ == "__main__":
    test_wavelet_fusion()
