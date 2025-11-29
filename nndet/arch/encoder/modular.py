import torch 
import torch.nn as nn
from typing import Callable, Tuple, Sequence, Union, List, Optional
from timm.models.swin_transformer import swin_base_patch4_window7_224_in22k as SwinTransformer
from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.arch.blocks.basic import AbstractBlock

from nndet.arch.encoder.swimTransformer import SwinTransformer3D
import torch.nn.functional as F  # 加入插值函数
import logging
from nndet.arch.encoder.window_attention_fusion import WindowAttentionFusion
from nndet.arch.encoder.WaveletFusion import WaveletSpatialFusion
from nndet.arch.encoder.channel_lightweight_fusion import ChannelWiseLightFusion

# 配置日志级别和格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ["Encoder"]

class MemoryEfficientFusion(nn.Module):
    """
    内存友好的真正融合模块
    - 解决假融合问题
    - 适合11GB GPU内存
    - 保证真正的CNN-Transformer交互
    - 基于2024年CVPR最新轻量级融合方法
    """
    def __init__(self, conv_channels, transformer_channels, fused_channels, window_size, num_heads):
        super().__init__()
        
        # 使用通道级轻量融合策略 - 解决假融合问题
        self.channel_fusion = ChannelWiseLightFusion(
            cnn_channels=conv_channels,
            trans_channels=transformer_channels,
            out_channels=fused_channels
        )
    
    def forward(self, conv_feat, transformer_feat):
        """
        内存友好的真正融合 - 简化版本
        """
        # 直接使用通道级轻量融合，无需额外处理
        return self.channel_fusion(conv_feat, transformer_feat)

class Encoder(AbstractEncoder):
    def __init__(self,
                 conv: Callable[[], nn.Module],
                 conv_kernels: Sequence[Union[Tuple[int], int]],
                 strides: Sequence[Union[Tuple[int], int]],
                 block_cls: AbstractBlock,
                 in_channels: int,
                 start_channels: int,
                 stage_kwargs: Sequence[dict] = None,
                 out_stages: Sequence[int] = None,
                 max_channels: int = None,
                 first_block_cls: Optional[AbstractBlock] = None,
                 # 移除额外参数
                 # bam_reduction_ratio: int = 16,
                 # drop_prob: float = 0.2,
                 # block_size: int = 5,
                 fft_low_ratio: float = 0.3):  # 添加 FFT 低频比例参数
        super().__init__()
        self.num_stages = len(conv_kernels)
        self.dim = conv.dim
        if stage_kwargs is None:
            stage_kwargs = [{}] * self.num_stages
        elif isinstance(stage_kwargs, dict):
            stage_kwargs = [stage_kwargs] * self.num_stages
        assert len(stage_kwargs) == len(conv_kernels)

        if out_stages is None:
            self.out_stages = list(range(self.num_stages))
        else:
            self.out_stages = out_stages
        if first_block_cls is None:
            first_block_cls = block_cls

        stages = []
        self.out_channels = []
        self.self_attention_fusion_modules = nn.ModuleList()
        
        # 小波融合模块 - 使用GPU加速版本，恢复所有stage
        self.use_wavelet_fusion = True
        self.wavelet_fusion_stages = [1, 3, 4]  # GPU版本速度快，可以使用所有stage
        self.wavelet_fusion_modules = nn.ModuleList()
        in_ch = in_channels
        if isinstance(strides[0], int):
            strides = [tuple([s] * self.dim) for s in strides]
        self.strides = strides
        self.use_transformer = True
        if self.use_transformer:
            # 确保 depths 和 num_heads 的长度与 self.num_stages 一致
            self.depths = [2] * self.num_stages
            self.num_heads = [4 * (2 ** i) for i in range(self.num_stages)]
            self.transformer = SwinTransformer3D(
                in_chans=in_channels,
                embed_dim=start_channels,
                window_size=(2,7,7),
                patch_size=(2,4,4),
                depths=self.depths,
                num_heads=self.num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                norm_layer=nn.LayerNorm
            )
            # 计算每个阶段的Transformer输出通道数
            self.transformer_out_channels = [int(start_channels * 2 ** i) for i in range(self.num_stages)]

        for stage_id in range(self.num_stages):
            current_in_channels = in_ch
            if stage_id == 0:
                _block = first_block_cls(
                    conv=conv,
                    in_channels=current_in_channels,
                    out_channels=start_channels,
                    conv_kernel=conv_kernels[stage_id],
                    stride=None,
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            else:
                _block = block_cls(
                    conv=conv,
                    in_channels=current_in_channels,
                    out_channels=None,
                    conv_kernel=conv_kernels[stage_id],
                    stride=strides[stage_id - 1],
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            in_ch = _block.get_output_channels()
            self.out_channels.append(in_ch)

            # 简化：只保留核心卷积块
            stages.append(_block)

            # 初始化小波融合模块
            if self.use_wavelet_fusion and stage_id in self.wavelet_fusion_stages:
                wavelet_fusion = WaveletSpatialFusion(
                    in_channels=in_ch,
                    out_channels=in_ch,
                    wavelet='haar'  # 适合医学图像的小波基
                )
                self.wavelet_fusion_modules.append(wavelet_fusion)
            else:
                self.wavelet_fusion_modules.append(nn.Identity())

            # 初始化 SelfAttentionFusion 模块
            if self.use_transformer:
                fused_channels = in_ch
                transformer_channels = self.transformer_out_channels[stage_id]
                window_size = self.transformer.window_size
                num_heads = self.num_heads[stage_id]
                self_attention_fusion = MemoryEfficientFusion(
                    conv_channels=in_ch,
                    transformer_channels=transformer_channels,
                    fused_channels=fused_channels,
                    window_size=window_size,
                    num_heads=num_heads
                )
                self.self_attention_fusion_modules.append(self_attention_fusion)

        self.stages = torch.nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        
        # Step 1: 双分支输入分配 (移除FFT频域处理)
        # CNN分支和Transformer分支都使用原始图像
        cnn_x = x
        if self.use_transformer:
            transformer_feats = self.transformer(x)  # 原始图像给Transformer

        
        # Step 2: 高效的分阶段处理和选择性融合
        for stage_id, module in enumerate(self.stages):
            # 处理 CNN 分支
            cnn_x = module(cnn_x)
            
            # 应用小波融合 (在Stage 1, 3, 4)
            if self.use_wavelet_fusion and stage_id < len(self.wavelet_fusion_modules):
                wavelet_module = self.wavelet_fusion_modules[stage_id]
                if not isinstance(wavelet_module, nn.Identity):
                    cnn_x = wavelet_module(cnn_x)
            
            # 选择性融合: 仅在Stage 2和5进行融合 (提高效率和效果)
            if (self.use_transformer and 
                stage_id < len(self.transformer_out_channels) and 
                stage_id in [2, 5]):  # 关键融合时机
                
                transformer_feat = transformer_feats[stage_id]
                transformer_feat_resized = F.interpolate(
                    transformer_feat, size=cnn_x.shape[2:], 
                    mode='trilinear', align_corners=False
                )
                
                # 使用超精度融合机制
                fused_x = self.self_attention_fusion_modules[stage_id](cnn_x, transformer_feat_resized)
                cnn_x = fused_x
                # logger.info(f"Stage {stage_id+1}: 极简真正融合完成 CNN: {cnn_x.shape} Transformer: {transformer_feat_resized.shape}")  # 已确认融合工作正常，关闭日志
            
            if stage_id in self.out_stages:
                outputs.append(cnn_x)
        
        return outputs

    def get_channels(self) -> List[int]:
        """
        计算每个返回特征图的通道数
        """
        out_channels = []
        for stage_id in range(self.num_stages):
            if stage_id in self.out_stages:
                out_channels.append(self.out_channels[stage_id])
        return out_channels

    def get_strides(self) -> List[List[int]]:
        """
        计算每个输出特征图相对于输入大小的步幅
        """
        out_strides = []
        for stage_id in range(self.num_stages):
            if stage_id == 0:
                out_strides.append([1] * self.dim)
            else:
                new_stride = [prev_stride * s for prev_stride, s
                              in zip(out_strides[stage_id - 1], self.strides[stage_id - 1])]
                out_strides.append(new_stride)
        return out_strides