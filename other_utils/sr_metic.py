import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

if __name__ == '__main__':

    # 初始化指标计算器
    psnr = PeakSignalNoiseRatio(data_range=1.0)  # 假设图像范围为[0,1]
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    # 创建示例图像
    img1 = torch.rand(1, 3, 256, 256)
    img2 = img1 + 0.1 * torch.randn_like(img1)
    img2 = torch.clamp(img2, 0, 1)

    # 计算指标
    psnr_value = psnr(img1, img2)
    ssim_value = ssim(img1, img2)

    print(f"PSNR: {psnr_value:.4f} dB")  # 输出: PSNR: 20.0488 dB
    print(f"SSIM: {ssim_value:.4f}")     # 输出: SSIM: 0.8135
