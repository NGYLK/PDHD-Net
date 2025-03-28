import torch
import torch.nn as nn
<<<<<<< HEAD
import torch.nn.functional as F
=======
>>>>>>> 864d4cade90dccff407e62481e9e8e0b38c746b0

class FFT3D(nn.Module):
    def __init__(self, in_channels: int, low_ratio: float = 0.4):
        super(FFT3D, self).__init__()
        self.low_ratio = low_ratio
        self.register_buffer("low_mask", None)  # 存储低频掩码

    def get_low_mask(self, x: torch.Tensor) -> torch.Tensor:
        """生成与输入同形状的低频掩码（中心立方区域）"""
        B, C, D, H, W = x.shape
        
        # 计算各维度低频区域大小
        low_D = max(1, int(D * self.low_ratio))
        low_H = max(1, int(H * self.low_ratio))
        low_W = max(1, int(W * self.low_ratio))
        center_D, center_H, center_W = D // 2, H // 2, W // 2
        
        # 计算低频区域的起止索引
        start_D = max(0, center_D - low_D // 2)
        end_D = min(D, center_D + low_D // 2 + low_D % 2)
        start_H = max(0, center_H - low_H // 2)
        end_H = min(H, center_H + low_H // 2 + low_H % 2)
        start_W = max(0, center_W - low_W // 2)
        end_W = min(W, center_W + low_W // 2 + low_W % 2)
        
        # 创建3D低频掩码
        mask = torch.zeros((1, 1, D, H, W), dtype=torch.float32, device=x.device)
        mask[..., start_D:end_D, start_H:end_H, start_W:end_W] = 1.0
        
        self.low_mask = mask
        
        # 确认掩码是否包含 1
        if mask.sum().item() == 0:
            # 设置一个简单的默认掩码
            default_mask = torch.zeros_like(mask)
            default_mask[..., D//4:3*D//4, H//4:3*H//4, W//4:3*W//4] = 1.0
            self.low_mask = default_mask
        
        return self.low_mask

    def forward(self, x: torch.Tensor) -> tuple:
        x = x.float()  # FFT需要float32
        
        # Step 1: 3D FFT并执行频谱中心化
        x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-3, -2, -1))  # 低频移至中心
        
        # Step 2: 分离高低频
        low_mask = self.get_low_mask(x)  # (1,1,D,H,W)
        fft_low = x_fft_shifted * low_mask
        fft_high = x_fft_shifted * (1 - low_mask)
        
        # Step 3: 逆中心化后逆变换
        fft_low = torch.fft.ifftshift(fft_low, dim=(-3, -2, -1))
        fft_high = torch.fft.ifftshift(fft_high, dim=(-3, -2, -1))
        
        low_freq = torch.fft.ifftn(fft_low, dim=(-3, -2, -1)).real
        high_freq = torch.fft.ifftn(fft_high, dim=(-3, -2, -1)).real

        return low_freq, high_freq

def test_fft():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, D, H, W = 2, 1, 32, 32, 32
    x = torch.randn(B, C, D, H, W, device=device)
    
    # 使用较高的 low_ratio 以获取更多的低频能量
    fft_module = FFT3D(in_channels=C, low_ratio=0.7).to(device)
    low, high = fft_module(x)
    
    # 频域能量验证
    x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
    x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-3, -2, -1))
    mask = fft_module.low_mask  # 直接使用模块中存储的掩码
    
    fft_low = x_fft_shifted * mask
    fft_high = x_fft_shifted * (1 - mask)
    
    energy_low = (torch.abs(fft_low)**2).sum().item()
    energy_high = (torch.abs(fft_high)**2).sum().item()
    total_energy = energy_low + energy_high
    print(f"频域低频能量占比: {energy_low / total_energy:.4f}")
    
    # 空间域重建验证
    error = torch.norm(x - (low + high)) / torch.norm(x)
    print(f"重建误差: {error:.6f}")

if __name__ == "__main__":
    test_fft()