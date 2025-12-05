"""
Configuration file for model hyperparameters.
"""


def get_default_config(model_type):
    """
    Get default hyperparameter configuration for each model type.

    Args:
        model_type: 'ebt', 'ffn', 'pc', or 'transformer'

    Returns:
        Dictionary of default hyperparameters
    """
    configs = {
        'ebt': {
            'n_optimization_steps': 10,
            'step_size': 1,
            'latent_dim': 128,
            'batch_size': 64,
            'num_epochs': 100,
            'lr': 1e-3
        },
        'ffn': {
            'latent_dim': 128,
            'batch_size': 64,
            'num_epochs': 100,
            'lr': 1e-4
        },
        'pc': {
            'latent_dim': 128,
            'batch_size': 64,
            'num_epochs': 100,
            'lr': 1e-3,
            'n_inference_steps': 20,
            'infer_tau': 25,
        },
        'transformer': {
            'vocab_size': 18,
            'num_classes': 10,
            'model_size': 'small',  # 'tiny', 'small', 'medium', 'large'
            'batch_size': 32,
            'num_epochs': 100,
            'lr': 1e-4,
            'weight_decay': 0.01,
            'warmup_steps': 1000,
            'max_seq_len': 512,
            'patience': 10,
            'print_every': 100
        }
    }
    return configs.get(model_type, {})
