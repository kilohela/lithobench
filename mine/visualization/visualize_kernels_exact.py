
import torch
import matplotlib.pyplot as plt
import os
import sys

import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pylitho.exact import Kernel

def visualize_kernels(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kernel_configs = {
        "focus": {"defocus": False, "conjuncture": False, "combo": False},
        "defocus": {"defocus": True, "conjuncture": False, "combo": False},
        "CT_focus": {"defocus": False, "conjuncture": True, "combo": False},
        "CT_defocus": {"defocus": True, "conjuncture": True, "combo": False},
        "combo_focus": {"defocus": False, "conjuncture": False, "combo": True},
        "combo_defocus": {"defocus": True, "conjuncture": False, "combo": True},
        "combo_CT_focus": {"defocus": False, "conjuncture": True, "combo": True},
        "combo_CT_defocus": {"defocus": True, "conjuncture": True, "combo": True},
    }

    for name, config in kernel_configs.items():
        try:
            print(f"Loading kernel: {name}")
            kernel_loader = Kernel(basedir="./kernel", 
                                   defocus=config["defocus"], 
                                   conjuncture=config["conjuncture"], 
                                   combo=config["combo"], 
                                   device=device)
            
            kernels = kernel_loader.kernels.cpu().numpy()
            
            # kernels shape: (num_kernels, height, width)
            num_kernels = kernels.shape[0]

            # Limit number of kernels to visualize
            num_kernels_to_show = min(num_kernels, 8)

            fig, axes = plt.subplots(1, num_kernels_to_show, figsize=(num_kernels_to_show * 2, 2))
            fig.suptitle(f'Kernel: {name} - First {num_kernels_to_show} Kernels')
            
            if num_kernels_to_show == 1:
                axes = [axes] # make it iterable

            for i in range(num_kernels_to_show):
                kernel_image = np.abs(kernels[i, :, :])
                ax = axes[i]
                im = ax.imshow(kernel_image, cmap='viridis', interpolation='nearest')
                ax.set_title(f'Kernel {i}')
                ax.axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(output_dir, f"{name}.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved visualization for kernel '{name}' to {save_path}")

        except Exception as e:
            print(f"Could not load or visualize kernel '{name}'. Error: {e}")

if __name__ == '__main__':
    output_directory = "./mine/visualization/kernel_visualizations_exact"
    visualize_kernels(output_directory)
