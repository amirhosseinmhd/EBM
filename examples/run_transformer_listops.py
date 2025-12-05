"""
Example script to train a standard transformer on ListOps task.

This demonstrates how to use the transformer model for the ListOps
hierarchical reasoning task from Long Range Arena.

Usage:
    python examples/run_transformer_listops.py
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_data_generator import create_synthetic_data
from model.train_transformer import train_transformer, test_transformer
from config.model_configs import get_default_config


def main():
    """Main function to train transformer on ListOps."""

    print("=" * 80)
    print("Standard Transformer on ListOps Task")
    print("=" * 80)

    # Configuration
    task = 'listops'
    vec_size = 10  # max_seq_len = vec_size * 50 = 500
    size_train = 10000
    size_val = 2000
    size_test = 2000

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Generate ListOps data
    print("\n" + "=" * 80)
    print("Generating ListOps Datasets")
    print("=" * 80)

    train_dataset, val_dataset, test_dataset, input_dim, output_dim = create_synthetic_data(
        task=task,
        size_train=size_train,
        size_val=size_val,
        size_test=size_test,
        vec_size=vec_size,
        device=device
    )

    print(f"\n✓ Datasets created:")
    print(f"  Training: {size_train} samples")
    print(f"  Validation: {size_val} samples")
    print(f"  Test: {size_test} samples")
    print(f"  Max sequence length: {input_dim}")
    print(f"  Number of classes: {output_dim}")

    # Get default config and customize if needed
    config = get_default_config('transformer')

    # You can customize the configuration here:
    # config['model_size'] = 'medium'  # Larger model
    # config['num_epochs'] = 200       # More epochs
    # config['batch_size'] = 64        # Larger batches
    # config['lr'] = 5e-5              # Different learning rate

    print("\n" + "=" * 80)
    print("Model Configuration")
    print("=" * 80)
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Train the transformer
    print("\n" + "=" * 80)
    print("Training Standard Transformer")
    print("=" * 80)

    model = train_transformer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        **config
    )

    # Final test evaluation
    print("\n" + "=" * 80)
    print("Final Test Evaluation")
    print("=" * 80)

    test_acc, test_loss = test_transformer(
        model=model,
        test_dataset=test_dataset,
        batch_size=config['batch_size'],
        device=device
    )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")
    print("=" * 80)

    # Optional: Save the model
    # save_path = 'transformer_listops.pt'
    # torch.save(model.state_dict(), save_path)
    # print(f"\nModel saved to {save_path}")

    return model, test_acc, test_loss


if __name__ == "__main__":
    model, accuracy, loss = main()
