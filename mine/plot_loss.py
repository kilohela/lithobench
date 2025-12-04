import matplotlib.pyplot as plt
import torch
import argparse
import os

'''
This script is used to plot the loss curves from a training checkpoint. 
It reads the checkpoint file and extracts the training and validation loss values. 
It then plots these values on a graph and saves the graph to a file.
An optional baseline loss curve can also be plotted for comparison.
'''

def plot_loss_curve(checkpoint_path, baseline_path, output_path):
    """
    Reads a training checkpoint and plots the training and validation loss curves.

    Args:
        checkpoint_path (str): The path to the checkpoint file.
        output_path (str): The path to save the generated plot image.
    """
    try:
        checkpoint = torch.load(checkpoint_path)
        logger = checkpoint['logger']
        baseline = torch.load(baseline_path)['logger'] if baseline_path is not None else None
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        return
    
    num_steps = len(logger['train_loss'])
    val_steps_interval = logger['val_steps_interval']

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_steps + 1), logger['train_loss'], label='Train Loss')
    plt.plot(range(val_steps_interval, num_steps + 1, val_steps_interval), logger['val_loss'], label='Validation Loss')
    if baseline is not None:
        try:
            baseline_steps = len(baseline['train_loss'])
            baseline_interval = baseline['val_steps_interval']
            plt.plot(range(1, baseline_steps + 1), baseline['train_loss'], label='Baseline Train Loss')
            plt.plot(range(baseline_interval, baseline_steps + 1, baseline_interval), baseline['val_loss'], label='Baseline Validation Loss')
        except ValueError:
            print(baseline_steps)
            print(baseline_interval)
            print(len(baseline['val_loss']))
            
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
    parser.add_argument("--output", "-o", required=False, type=str, help="Path to save the plot image")
    parser.add_argument("--baseline", "-b", required=False, type=str, help="baseline checkpoint path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    baseline_path = args.baseline
    output_path = os.path.join('mine', 'outputs', f'{os.path.basename(args.checkpoint)}_loss.png') if args.output is None else args.output
    plot_loss_curve(args.checkpoint, args.baseline, output_path)
