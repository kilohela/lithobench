
import torch
import matplotlib.pyplot as plt
import os
import sys

# Add project root to sys.path to allow importing from mine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mine.unet_backbone import UNET

def visualize_kernels(checkpoint_path, output_dir):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNET(in_channels=1, out_channels=1, features=[64, 128, 256, 512]).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # The checkpoint might be a dictionary with a 'model' key or it might be the state dict itself
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weights = module.weight.data.cpu().numpy()
            
            # weights shape: (out_channels, in_channels, kernel_height, kernel_width)
            out_channels = weights.shape[0]
            
            # Limit number of kernels to visualize to a reasonable number, e.g., 8
            num_kernels_to_show = min(out_channels, 8)

            fig, axes = plt.subplots(1, num_kernels_to_show, figsize=(num_kernels_to_show * 2, 2))
            fig.suptitle(f'Layer: {name} - First {num_kernels_to_show} Kernels (Input Channel 0)')

            if num_kernels_to_show == 1:
                axes = [axes] # make it iterable

            for i in range(num_kernels_to_show):
                # We visualize the kernel for the first input channel
                kernel = weights[i, 0, :, :]
                ax = axes[i]
                im = ax.imshow(kernel, cmap='viridis', interpolation='nearest')
                ax.set_title(f'Kernel {i}')
                ax.axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(output_dir, f"{name.replace('.', '_')}.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved visualization for layer {name} to {save_path}")

if __name__ == '__main__':
    checkpoint_file = "./mine/checkpoints/unet_backbone_best.pth"
    output_directory = "./mine/visualization/kernel_visualizations"
    visualize_kernels(checkpoint_file, output_directory)
