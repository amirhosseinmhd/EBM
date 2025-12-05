"""
Training script for Standard Transformer on ListOps task.

This script provides training and evaluation functions compatible with
the existing codebase structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
from typing import Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer_standard import StandardTransformer, create_standard_transformer


def train_transformer(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    test_dataset: TensorDataset,
    vocab_size: int = 18,
    num_classes: int = 10,
    model_size: str = 'small',
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    max_seq_len: int = 512,
    device: Optional[str] = None,
    patience: int = 10,
    print_every: int = 100
) -> StandardTransformer:
    """
    Train a standard transformer on ListOps dataset.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        vocab_size: Size of vocabulary (default: 18 for ListOps)
        num_classes: Number of classes (default: 10 for ListOps)
        model_size: Model size ('tiny', 'small', 'medium', 'large')
        batch_size: Batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        warmup_steps: Number of warmup steps for learning rate
        max_seq_len: Maximum sequence length
        device: Device to train on (None = auto-detect)
        patience: Early stopping patience
        print_every: Print training stats every N steps

    Returns:
        Trained transformer model
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Create model
    model = create_standard_transformer(
        vocab_size=vocab_size,
        num_classes=num_classes,
        model_size=model_size
    )
    model = model.to(device)

    # Print model info
    total_params = model.get_num_params(exclude_embeddings=False)
    non_emb_params = model.get_num_params(exclude_embeddings=True)

    print("\n" + "=" * 70)
    print("Standard Transformer Model")
    print("=" * 70)
    print(f"Model size: {model_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Non-embedding parameters: {non_emb_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print("=" * 70)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\nTraining Standard Transformer on ListOps")
    print("=" * 70)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    global_step = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)
            loss = criterion(logits, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            train_correct += (predicted == y).sum().item()
            train_total += y.size(0)

            global_step += 1

            # Print progress
            if global_step % print_every == 0:
                avg_loss = train_loss / (batch_idx + 1)
                avg_acc = 100.0 * train_correct / train_total
                current_lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step:5d} | "
                      f"Epoch {epoch:3d} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Acc: {avg_acc:.2f}% | "
                      f"LR: {current_lr:.6f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item()
                _, predicted = torch.max(logits, dim=1)
                val_correct += (predicted == y).sum().item()
                val_total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # Test evaluation
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                test_loss += loss.item()
                _, predicted = torch.max(logits, dim=1)
                test_correct += (predicted == y).sum().item()
                test_total += y.size(0)

        test_loss /= len(test_loader)
        test_acc = 100.0 * test_correct / test_total

        print(f"\nEpoch {epoch:3d} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        # Validation
        val_correct = 0
        val_total = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, predicted = torch.max(logits, dim=1)
            val_correct += (predicted == y).sum().item()
            val_total += y.size(0)
        val_acc = 100.0 * val_correct / val_total

        # Test
        test_correct = 0
        test_total = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, predicted = torch.max(logits, dim=1)
            test_correct += (predicted == y).sum().item()
            test_total += y.size(0)
        test_acc = 100.0 * test_correct / test_total

    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print("=" * 70)

    return model


def test_transformer(
    model: StandardTransformer,
    test_dataset: TensorDataset,
    batch_size: int = 32,
    device: Optional[str] = None
) -> Tuple[float, float]:
    """
    Test a trained transformer model.

    Args:
        model: Trained transformer model
        test_dataset: Test dataset
        batch_size: Batch size
        device: Device to test on (None = auto-detect)

    Returns:
        Tuple of (accuracy, loss)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            test_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            test_correct += (predicted == y).sum().item()
            test_total += y.size(0)

    test_loss /= len(test_loader)
    test_acc = 100.0 * test_correct / test_total

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Loss: {test_loss:.4f}")

    return test_acc, test_loss


if __name__ == "__main__":
    # Example usage with synthetic data
    from data.listops_generator import (
        generate_listops_data,
        create_listops_tensors,
        create_vocab
    )

    print("=" * 70)
    print("Training Standard Transformer on ListOps")
    print("=" * 70)

    # Generate ListOps data
    print("\nGenerating ListOps datasets...")
    vocab = create_vocab()
    max_seq_len = 512

    # Generate samples
    train_samples = generate_listops_data(
        num_samples=10000,
        max_depth=10,
        max_args=10,
        min_length=50,
        max_length=500,
        seed=42
    )

    val_samples = generate_listops_data(
        num_samples=2000,
        max_depth=10,
        max_args=10,
        min_length=50,
        max_length=500,
        seed=43
    )

    test_samples = generate_listops_data(
        num_samples=2000,
        max_depth=10,
        max_args=10,
        min_length=50,
        max_length=500,
        seed=44
    )

    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train = create_listops_tensors(train_samples, vocab, max_seq_len, device)
    X_val, y_val = create_listops_tensors(val_samples, vocab, max_seq_len, device)
    X_test, y_test = create_listops_tensors(test_samples, vocab, max_seq_len, device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    print(f"✓ Datasets created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Max seq length: {max_seq_len}")

    # Train model
    model = train_transformer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        vocab_size=len(vocab),
        num_classes=10,
        model_size='small',
        batch_size=32,
        num_epochs=100,
        lr=1e-4,
        device=device
    )

    # Test model
    test_acc, test_loss = test_transformer(model, test_dataset, device=device)

    print("\n✓ Training complete!")
