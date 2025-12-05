# Synthetic Data Generation and Model Training

This module provides a clean interface for generating synthetic datasets and training models on various tasks.

## Structure

```
├── data/
│   └── synthetic_data_generator.py  # Data generation for different tasks
├── model/
│   ├── ebt_s1_toy.py               # Energy-Based Transformer model
│   └── ffn_toy.py                  # Feedforward Neural Network model
└── run_main.py                     # Main runner script
```

## Quick Start

### Using the Main Runner (Recommended)

The simplest way to run experiments:

```python
python run_main.py
```

This will:
1. Generate synthetic datasets for the addition task
2. Train both EBT and FFN models
3. Compare their performance on out-of-distribution test data

### Custom Configuration

```python
from run_main import run_main

# Run both models with custom settings
results = run_main(
    task='add',              # Task type: 'add' or 'multiply'
    vec_size=10,             # Size of each vector
    size_train=10000,        # Training samples
    size_val=256,            # Validation samples
    size_test=256,           # Test samples (OOD)
    model_type='both',       # 'ebt', 'ffn', or 'both'
    ebt_config={
        'n_optimization_steps': 10,
        'step_size': 1,
        'latent_dim': 128,
        'batch_size': 64,
        'num_epochs': 10000,
        'lr': 1e-3
    },
    ffn_config={
        'latent_dim': 128,
        'batch_size': 128,
        'num_epochs': 50000,
        'lr': 1e-4
    }
)

# Access trained models
ebt_model = results['ebt']['model']
ffn_model = results['ffn']['model']
```

## Data Generation

### Using the Data Generator Directly

```python
from data.synthetic_data_generator import create_synthetic_data
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create datasets
train_dataset, val_dataset, test_dataset, input_dim, output_dim = create_synthetic_data(
    task='add',           # 'add', 'multiply', or 'listops'
    size_train=10000,
    size_val=256,
    size_test=256,
    vec_size=10,
    device=device
)

# Access data
x_train, y_train = train_dataset.tensors
x_val, y_val = val_dataset.tensors
x_test, y_test = test_dataset.tensors
```

### Supported Tasks

#### Addition (`task='add'`)
- Input: Two vectors `v1` and `v2` concatenated
- Output: Element-wise sum `v1 + v2`
- Training distribution: Values in range `[-1, 1]`
- Test distribution (OOD): Values in range `[-2, 2]`

#### Multiplication (`task='multiply'`)
- Input: Two vectors `v1` and `v2` concatenated
- Output: Element-wise product `v1 * v2`
- Training distribution: Values in range `[-1, 1]`
- Test distribution (OOD): Values in range `[-2, 2]`

#### ListOps (`task='listops'`)
Hierarchical reasoning task from Google Research's [Long-Range Arena](https://github.com/google-research/long-range-arena).

- **Input**: Tokenized nested expressions with operations
- **Output**: Single value `[0-9]`
- **Challenge**: Model must parse hierarchical structure and apply operations correctly

**Operations:**
- `[MIN` - Minimum of arguments
- `[MAX` - Maximum of arguments
- `[MED` - Median of arguments
- `[SM` - Sum modulo 10

**Format:** Binary tree representation (from original implementation)
```
Expression: ( ( ( ( [SM 2 ) 6 ) 5 ) ] )
Meaning: SM(2, 6, 5) = (2+6+5) % 10 = 3
```

The nested `( ( ( (` comes from left-associative pairing:
- `([SM, 2)` → `((prev), 6)` → `((prev), 5)` → `((prev), ])`

**Example with nesting:**
```
Expression: ( ( ( [MAX 2 ) ( ( ( [MIN 9 ) 4 ) ] ) ) ] )
Meaning: MAX(2, MIN(9, 4))

Step 1: MIN(9, 4) = 4
Step 2: MAX(2, 4) = 4
Result: 4 ✓
```

**Usage:**
```python
train_ds, val_ds, test_ds, input_dim, output_dim = create_synthetic_data(
    task='listops',
    size_train=10000,
    size_val=2000,
    size_test=2000,
    vec_size=10,      # max_seq_len = vec_size * 50 = 500 tokens
    device='cuda'
)
# Input shape: (batch, 500) - tokenized sequences
# Output shape: (batch,) - values 0-9
# Vocabulary: 18 tokens
```

## Running Individual Models

### Energy-Based Transformer (EBT)

```python
from model.ebt_s1_toy import train_energy_model, test_energy_model
from data.synthetic_data_generator import create_synthetic_data
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate data
train_dataset, val_dataset, test_dataset, input_dim, output_dim = create_synthetic_data(
    task='add',
    size_train=10000,
    size_val=256,
    size_test=256,
    vec_size=10,
    device=device
)

# Train model
energy_fn = train_energy_model(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    input_dim=input_dim,
    output_dim=output_dim,
    n_optimization_steps=10,
    step_size=1
)

# Test model
test_mse = test_energy_model(energy_fn, n_optimization_steps=10, step_size=1)
```

### Feedforward Network (FFN)

```python
from model.ffn_toy import train_feedforward
from data.synthetic_data_generator import create_synthetic_data
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate data
train_dataset, val_dataset, test_dataset, input_dim, output_dim = create_synthetic_data(
    task='add',
    size_train=10000,
    size_val=256,
    size_test=256,
    vec_size=10,
    device=device
)

# Train model
model = train_feedforward(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    input_dim=input_dim,
    output_dim=output_dim
)
```

## Adding New Tasks

To add a new task (e.g., subtraction), modify `data/synthetic_data_generator.py`:

```python
def _create_subtraction_data(size_train, size_val, size_test, vec_size, device):
    """Create synthetic data for vector subtraction task."""
    
    # Training set
    v1_train = torch.randn(size_train, vec_size, device=device) * 2 - 1
    v2_train = torch.randn(size_train, vec_size, device=device) * 2 - 1
    x_train = torch.cat([v1_train, v2_train], dim=1)
    y_train = v1_train - v2_train  # Subtraction
    
    # Validation set
    v1_val = torch.randn(size_val, vec_size, device=device) * 2 - 1
    v2_val = torch.randn(size_val, vec_size, device=device) * 2 - 1
    x_val = torch.cat([v1_val, v2_val], dim=1)
    y_val = v1_val - v2_val
    
    # Test set (OOD)
    v1_test = torch.randn(size_test, vec_size, device=device) * 4 - 2
    v2_test = torch.randn(size_test, vec_size, device=device) * 4 - 2
    x_test = torch.cat([v1_test, v2_test], dim=1)
    y_test = v1_test - v2_test
    
    # Create TensorDatasets
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    
    input_dim = vec_size * 2
    output_dim = vec_size
    
    return train_dataset, val_dataset, test_dataset, input_dim, output_dim
```

Then add it to the `create_synthetic_data` function:

```python
def create_synthetic_data(task, ...):
    if task == 'add':
        return _create_addition_data(...)
    elif task == 'multiply':
        return _create_multiplication_data(...)
    elif task == 'subtract':
        return _create_subtraction_data(...)
    else:
        raise ValueError(f"Unknown task: {task}")
```

## Key Features

1. **Separation of Concerns**: Data generation is completely separated from model training
2. **Flexible Configuration**: Easy to customize hyperparameters for each model
3. **Task-Agnostic**: Add new tasks by simply implementing the data generation function
4. **Clean Interface**: Simple function calls with clear return values
5. **Unified Training**: Both models use the same datasets for fair comparison
6. **Out-of-Distribution Testing**: Automatic OOD evaluation on harder test distributions

## Example Output

```
================================================================================
Running experiments on task: ADD
Device: cuda
GPU: NVIDIA GeForce RTX 3090
================================================================================

[1/3] Generating synthetic datasets...
✓ Dataset created: train=10000, val=256, test=256
✓ Input dim: 20, Output dim: 10

================================================================================
[2/3] Training Energy-Based Transformer (EBT) Model
================================================================================
Training Energy-Based Model (First-Order Approximation)
============================================================
Only tracking gradients through LAST optimization step
Total optimization steps: 10
Steps with θ-gradients: 1 (the last step only)
============================================================

Epoch    0 | Train Loss: 0.123456 | Val Loss: 0.234567 | Test Loss: 0.345678 | Energy: 1.2345 → 0.5678
...

✓ EBT Model Training Complete | Test MSE: 0.012345

================================================================================
[3/3] Training Feedforward Network (FFN) Model
================================================================================
...

✓ FFN Model Training Complete | Test MSE: 0.023456

================================================================================
EXPERIMENT SUMMARY
================================================================================
Task: ADD
Vector Size: 10
Training Samples: 10000
Validation Samples: 256
Test Samples: 256 (Out-of-Distribution)
--------------------------------------------------------------------------------
EBT Model Test MSE: 0.012345
FFN Model Test MSE: 0.023456
--------------------------------------------------------------------------------
Best Model: EBT (Difference: 0.011111)
================================================================================
```
