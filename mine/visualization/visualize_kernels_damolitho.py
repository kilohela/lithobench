
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lithobench.litho.damolitho import Generator

def visualize_kernels(checkpoint_path, output_dir):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(in_ch=1, out_ch=1).to(device)

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Please ensure the checkpoint file is not corrupted and is compatible with the model architecture.")
        return

    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            weights = module.weight.data.cpu().numpy()
            
            out_channels, in_channels, kh, kw = weights.shape
            
            num_out_kernels_to_show = min(out_channels, 4)
            num_in_kernels_to_show = min(in_channels, 4)

            fig, axes = plt.subplots(num_in_kernels_to_show, num_out_kernels_to_show, figsize=(num_out_kernels_to_show * 2.5, num_in_kernels_to_show * 2.5))
            fig.suptitle(f'Layer: {name} ({kh}x{kw})')

            if num_in_kernels_to_show == 1 and num_out_kernels_to_show == 1:
                axes = np.array([[axes]])
            elif num_in_kernels_to_show == 1:
                 axes = np.array([axes])
            elif num_out_kernels_to_show == 1:
                 axes = np.array([[ax] for ax in axes])

            for i in range(num_in_kernels_to_show):
                for j in range(num_out_kernels_to_show):
                    kernel = weights[j, i, :, :]
                    ax = axes[i, j]
                    ax.imshow(kernel, cmap='viridis', interpolation='nearest')
                    ax.set_title(f'In {i} -> Out {j}', fontsize=8)
                    ax.axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(output_dir, f"{name.replace('.', '_')}.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved visualization for layer {name} to {save_path}")

if __name__ == '__main__':
    checkpoint_file = "saved/MetalSet_DAMOLitho/netG.pth"
    output_directory = "./mine/visualization/kernel_visualizations_damolitho"
    visualize_kernels(checkpoint_file, output_directory)
