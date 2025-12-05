import torch
# import jax.numpy as jnp
from torch.utils.data import TensorDataset, DataLoader
from data.listops_generator import (
    generate_listops_data,
    create_listops_tensors,
    create_vocab
)


def create_synthetic_data(task, size_train=1000, size_val=256, size_test=256, vec_size=10, device='cpu'):
    """
    Create synthetic datasets for different tasks.
    Args:
        task: Task type ('add', 'multiply', 'listops', etc.)
        size_train: Number of training samples
        size_val: Number of validation samples
        size_test: Number of test samples
        vec_size: Size of each vector (for add/multiply) or max sequence length multiplier (for listops)
        device: Device to create tensors on ('cpu', 'cuda', etc.)

    Returns:
        train_dataset: TensorDataset for training
        val_dataset: TensorDataset for validation
        test_dataset: TensorDataset for testing
        input_dim: Dimension of input
        output_dim: Dimension of output
    """

    if task == 'add':
        return _create_addition_data(size_train, size_val, size_test, vec_size, device)
    elif task == 'multiply':
        return _create_multiplication_data(size_train, size_val, size_test, vec_size, device)
    elif task == 'listops':
        return _create_listops_data(size_train, size_val, size_test, vec_size, device)
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks: 'add', 'multiply', 'listops'")


def _create_addition_data(size_train, size_val, size_test, vec_size, device):
    """Create synthetic data for vector addition task."""
    
    # Training set: range [-1, 1]
    v1_train = torch.randn(size_train, vec_size, device=device) * 2 - 1
    v2_train = torch.randn(size_train, vec_size, device=device) * 2 - 1
    x_train = torch.cat([v1_train, v2_train], dim=1)
    y_train = v1_train + v2_train
    
    # Validation set: same distribution as training
    v1_val = torch.randn(size_val, vec_size, device=device) * 2 - 1
    v2_val = torch.randn(size_val, vec_size, device=device) * 2 - 1
    x_val = torch.cat([v1_val, v2_val], dim=1)
    y_val = v1_val + v2_val
    
    # Test set: harder problem - larger values, range [-2, 2]
    v1_test = torch.randn(size_test, vec_size, device=device) * 4 - 2
    v2_test = torch.randn(size_test, vec_size, device=device) * 4 - 2
    x_test = torch.cat([v1_test, v2_test], dim=1)
    y_test = v1_test + v2_test
    
    # Create TensorDatasets
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    
    input_dim = vec_size * 2
    output_dim = vec_size
    
    return train_dataset, val_dataset, test_dataset, input_dim, output_dim


def _create_multiplication_data(size_train, size_val, size_test, vec_size, device):
    """Create synthetic data for element-wise vector multiplication task."""
    
    # Training set: range [-1, 1]
    v1_train = torch.randn(size_train, vec_size, device=device) * 2 - 1
    v2_train = torch.randn(size_train, vec_size, device=device) * 2 - 1
    x_train = torch.cat([v1_train, v2_train], dim=1)
    y_train = v1_train * v2_train
    
    # Validation set: same distrib—ution as training
    v1_val = torch.randn(size_val, vec_size, device=device) * 2 - 1
    v2_val = torch.randn(size_val, vec_size, device=device) * 2 - 1
    x_val = torch.cat([v1_val, v2_val], dim=1)
    y_val = v1_val * v2_val
    
    # Test set: harder problem - larger values, range [-2, 2]
    v1_test = torch.randn(size_test, vec_size, device=device) * 4 - 2
    v2_test = torch.randn(size_test, vec_size, device=device) * 4 - 2
    x_test = torch.cat([v1_test, v2_test], dim=1)
    y_test = v1_test * v2_test
    
    # Create TensorDatasets
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    
    input_dim = vec_size * 2
    output_dim = vec_size
    
    return train_dataset, val_dataset, test_dataset, input_dim, output_dim


def _create_listops_data(size_train, size_val, size_test, vec_size, device):
    """
    Create synthetic data for ListOps hierarchical reasoning task.

    Args:
        size_train: Number of training samples
        size_val: Number of validation samples
        size_test: Number of test samples
        vec_size: Not used directly, but determines max_seq_len (vec_size * 50)
        device: Device to create tensors on

    Returns:
        train_dataset: TensorDataset for training
        val_dataset: TensorDataset for validation
        test_dataset: TensorDataset for testing
        input_dim: Maximum sequence length
        output_dim: Number of output classes (10 for values 0-9)
    """
    # Create vocabulary
    vocab = create_vocab()
    vocab_size = len(vocab)

    # Set max sequence length based on vec_size
    # Default: vec_size=10 -> max_seq_len=500
    max_seq_len = vec_size * 50

    # Generate samples with different seeds for reproducibility
    print(f"Generating {size_train} training samples...")
    train_samples = generate_listops_data(
        num_samples=size_train,
        max_depth=10,
        max_args=10,
        min_length=50,
        max_length=max_seq_len,
        seed=42
    )

    print(f"Generating {size_val} validation samples...")
    val_samples = generate_listops_data(
        num_samples=size_val,
        max_depth=10,
        max_args=10,
        min_length=50,
        max_length=max_seq_len,
        seed=43
    )

    print(f"Generating {size_test} test samples...")
    test_samples = generate_listops_data(
        num_samples=size_test,
        max_depth=10,
        max_args=10,
        min_length=50,
        max_length=max_seq_len,
        seed=44
    )

    # Convert to tensors
    x_train, y_train = create_listops_tensors(train_samples, vocab, max_seq_len, device)
    x_val, y_val = create_listops_tensors(val_samples, vocab, max_seq_len, device)
    x_test, y_test = create_listops_tensors(test_samples, vocab, max_seq_len, device)

    # Create TensorDatasets
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    input_dim = max_seq_len
    output_dim = 10  # 10 possible output classes (0-9)

    print(f"ListOps dataset created:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"  Input shape: {x_train.shape}")
    print(f"  Output classes: {output_dim}")

    return train_dataset, val_dataset, test_dataset, input_dim, output_dim


def get_dataloader(dataset, batch_size, shuffle=True):
    """Helper function to create DataLoader from dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def convert_dataset_to_jax(dataset: TensorDataset):
    x_torch, y_torch = dataset.tensors
    x_jax = torch_to_jax(x_torch)
    y_jax = torch_to_jax(y_torch)
    return x_jax, y_jax


# def torch_to_jax(tensor: torch.Tensor) -> jnp.ndarray:
#     return jnp.array(tensor.detach().cpu().numpy())
if __name__ == "__main__":
    # Example usage
    train_data, val_data, test_data, input_dim, output_dim = create_synthetic_data(
        task='listops',
        size_train=1,
        size_val=1,
        size_test=1,
        vec_size=10
    )
    print(next(iter(train_data)))