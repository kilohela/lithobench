import sys
sys.path.append(".")
import torch
import matplotlib.pyplot as plt
from lithobench.dataset import *

if __name__ == "__main__":
    train_loader, _ = loadersILT("MetalSet", (2048, 2048), 1, 8)
    for x, y in train_loader:
        # [1, 1, 2048, 2048]
        # 只处理第一个batch
        fig, axes = plt.subplots(2, 3, figsize=(10, 10))

        # 显示原始图像x
        axes[0, 0].imshow(x[0, 0].cpu().numpy(), cmap='gray')
        axes[0, 0].set_title('Original Image X')
        axes[0, 0].axis('off')

        # 显示原始图像y
        axes[0, 1].imshow(y[0, 0].cpu().numpy(), cmap='gray')
        axes[0, 1].set_title('Original Image Y')
        axes[0, 1].axis('off')

        # 对图像x进行FFT并显示
        f_x = torch.fft.fft2(x[0, 0])
        f_x_shifted = torch.fft.fftshift(f_x)  # 零频移到中心
        magnitude_x = torch.abs(f_x_shifted)
        log_magnitude_x = torch.log(magnitude_x + 1e-8)  # 加一个小常数避免log(0)

        axes[1, 0].imshow(log_magnitude_x.cpu().numpy(), cmap='viridis')
        axes[1, 0].set_title('X freq (log scale)')
        axes[1, 0].axis('off')

        # 对图像y进行FFT并显示
        f_y = torch.fft.fft2(y[0, 0])
        f_y_shifted = torch.fft.fftshift(f_y)  # 零频移到中心
        magnitude_y = torch.abs(f_y_shifted)
        log_magnitude_y = torch.log(magnitude_y + 1e-8)  # 加一个小常数避免log(0)

        axes[1, 1].imshow(log_magnitude_y.cpu().numpy(), cmap='viridis')
        axes[1, 1].set_title('Y freq (log scale)')
        axes[1, 1].axis('off')

        f_y_lowpass_shifted = torch.zeros_like(f_y_shifted)
        center = f_y_shifted.shape[-1] // 2
        low_pass_radius = 36
        f_y_lowpass_shifted[center - low_pass_radius:center + low_pass_radius, center - low_pass_radius:center + low_pass_radius] = f_y_shifted[center - low_pass_radius:center + low_pass_radius, center - low_pass_radius:center + low_pass_radius]

        f_y_lowpass = torch.fft.ifftshift(f_y_lowpass_shifted)
        y_lowpass = np.abs(torch.fft.ifft2(f_y_lowpass))

        # axes[0, 2].imshow(y_lowpass.cpu().numpy(), cmap='gray')
        # axes[0, 2].set_title('Lowpass Filtered Y')
        # axes[0, 2].axis('off')

        magnitude_y_lowpass = torch.abs(f_y_lowpass_shifted)
        log_magnitude_y_lowpass = torch.log(magnitude_y_lowpass + 1e-8)
        axes[1, 2].imshow(log_magnitude_y_lowpass.cpu().numpy(), cmap='viridis')
        axes[1, 2].set_title('Y lowpass freq (log scale)')
        axes[1, 2].axis('off')

        y_lowpass_binary = (y_lowpass > 0.5).float()
        axes[0, 2].imshow(y_lowpass_binary.cpu().numpy(), cmap='gray')
        axes[0, 2].set_title('Binary Y lowpass')
        axes[0, 2].axis('off')


        plt.tight_layout()
        plt.show()
        break  # 只处理第一个batch中的图像