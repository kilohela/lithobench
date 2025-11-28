import matplotlib.pyplot as plt
import torch
import argparse
import os

def plot_loss_curve(checkpoint_path, output_path):
    """
    Reads a training checkpoint and plots the training and validation loss curves.

    Args:
        checkpoint_path (str): The path to the checkpoint file.
        output_path (str): The path to save the generated plot image.
    """
    try:
        checkpoint = torch.load(checkpoint_path)
        logger = checkpoint['logger']
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        return
    
    num_steps = len(logger['train_loss'])
    val_steps_interval = logger['val_steps_interval']

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_steps + 1), logger['train_loss'], label='Train Loss')
    plt.plot(range(val_steps_interval, num_steps + 1, val_steps_interval), logger['val_loss'], label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve of {os.path.basename(checkpoint_path)}')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Loss curve saved to {output_path}")

def parseArgs(): 
    parser = argparse.ArgumentParser(description="Read a checkpoint file and plot the loss curve.")
    parser.add_argument("--checkpoint", "-c", required=True, type=str, help="Path to the checkpoint file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    output_path = os.path.join('mine', 'outputs', f'{os.path.basename(args.checkpoint)}_loss.png')
    plot_loss_curve(args.checkpoint, output_path)
