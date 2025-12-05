"""
ListOps synthetic data generation adapted from Google Research Long-Range Arena.
Original: https://github.com/google-research/long-range-arena

Generates hierarchical reasoning tasks with nested operations.

Example: ( ( ( ( [SM 2 ) 6 ) 5 ) ] )
Meaning: SM(2, 6, 5) = (2+6+5) % 10 = 3

Format Note: The nested parentheses ( ( ( ( come from the binary tree representation
used in the original implementation. Each operation is built left-associatively:
  ([SM, 2) → ((prev), 6) → ((prev), 5) → ((prev), ])

Operations:
  [MIN - Minimum of arguments
  [MAX - Maximum of arguments
  [MED - Median of arguments
  [SM  - Sum modulo 10

All results are in range [0-9].
"""

import random
import numpy as np
import torch
from torch.utils.data import TensorDataset


# Operators and constants
MIN, MAX, MED = '[MIN', '[MAX', '[MED'
SUM_MOD = '[SM'
END = ']'
OPERATORS = [MIN, MAX, MED, SUM_MOD]
VALUES = range(10)
VALUE_P = 0.25  # Probability of generating a value vs operator


def generate_tree(depth, max_depth, max_args):
    """
    Generate tree-like equations recursively.

    Args:
        depth: Current depth in the tree
        max_depth: Maximum allowed depth
        max_args: Maximum number of arguments per operator

    Returns:
        tree: Tuple representing the tree structure
        length: Number of tokens in the tree
    """
    if depth < max_depth:
        r = random.random()
    else:
        r = 1  # Force value generation at max depth

    if r > VALUE_P:
        # Generate a leaf node (value)
        value = random.choice(VALUES)
        return value, 1
    else:
        # Generate an operator node with multiple children
        num_values = random.randint(2, max_args)
        values = []
        length = 2  # Count for operator and END tokens

        for _ in range(num_values):
            sub_t, sub_l = generate_tree(depth + 1, max_depth, max_args)
            values.append(sub_t)
            length += sub_l

        # Build tree structure
        op = random.choice(OPERATORS)
        t = (op, values[0])
        for value in values[1:]:
            t = (t, value)
        t = (t, END)

    return t, length


def to_string(t, parens=True):
    """
    Convert tree structure to string representation.

    Args:
        t: Tree structure (tuple, int, or str)
        parens: Whether to include parentheses

    Returns:
        String representation of the tree
    """
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        if parens:
            return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'


def to_value(t):
    """
    Evaluate the tree structure to compute the output.

    Args:
        t: Tree structure

    Returns:
        Computed value (0-9)
    """
    if not isinstance(t, tuple):
        return t

    l = to_value(t[0])
    r = to_value(t[1])

    if l in OPERATORS:
        # Start of an operation, accumulate arguments
        return (l, [r])
    elif r == END:
        # End of operation, apply the operator
        if l[0] == MIN:
            return min(l[1])
        elif l[0] == MAX:
            return max(l[1])
        elif l[0] == MED:
            return int(np.median(l[1]))
        elif l[0] == SUM_MOD:
            return np.sum(l[1]) % 10
    elif isinstance(l, tuple):
        # Accumulate more arguments
        return (l[0], l[1] + [r])


def create_vocab():
    """
    Create vocabulary for tokenization.

    Returns:
        Dictionary mapping tokens to IDs
    """
    vocab = {'[PAD]': 0}

    # Add operators
    for op in OPERATORS:
        vocab[op] = len(vocab)

    # Add special tokens
    vocab[END] = len(vocab)
    vocab['('] = len(vocab)
    vocab[')'] = len(vocab)

    # Add values
    for val in VALUES:
        vocab[str(val)] = len(vocab)

    return vocab


def tokenize_sequence(seq_str, vocab, max_len):
    """
    Convert string sequence to token IDs.

    Args:
        seq_str: String representation of the sequence
        vocab: Vocabulary dictionary
        max_len: Maximum sequence length

    Returns:
        List of token IDs (padded or truncated to max_len)
    """
    tokens = seq_str.split()
    token_ids = [vocab.get(t, vocab['[PAD]']) for t in tokens]

    # Pad or truncate
    if len(token_ids) < max_len:
        token_ids += [vocab['[PAD]']] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]

    return token_ids


def generate_listops_data(num_samples, max_depth=10, max_args=10,
                          min_length=50, max_length=500, seed=None):
    """
    Generate ListOps dataset.

    Args:
        num_samples: Number of samples to generate
        max_depth: Maximum tree depth
        max_args: Maximum arguments per operator
        min_length: Minimum sequence length (in tokens)
        max_length: Maximum sequence length (in tokens)
        seed: Random seed for reproducibility

    Returns:
        List of (string, value) pairs
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    data = set()
    attempts = 0
    max_attempts = num_samples * 100  # Prevent infinite loops

    while len(data) < num_samples and attempts < max_attempts:
        tree, length = generate_tree(1, max_depth, max_args)
        if min_length <= length <= max_length:
            data.add(tree)
        attempts += 1

    if len(data) < num_samples:
        print(f"Warning: Only generated {len(data)} samples out of {num_samples} requested")

    # Convert to list of (string, value) pairs
    samples = [[to_string(tree), to_value(tree)] for tree in data]
    return samples


def create_listops_tensors(samples, vocab, max_seq_len, device='cpu'):
    """
    Convert ListOps samples to PyTorch tensors.

    Args:
        samples: List of (string, value) pairs
        vocab: Vocabulary dictionary
        max_seq_len: Maximum sequence length
        device: PyTorch device

    Returns:
        X_tensor: Input sequences (tokenized)
        y_tensor: Target values
    """
    X = []
    y = []

    for seq_str, target in samples:
        token_ids = tokenize_sequence(seq_str, vocab, max_seq_len)
        X.append(token_ids)
        y.append(target)

    X_tensor = torch.tensor(X, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    return X_tensor, y_tensor
