"""
Configuration file for model hyperparameters.
"""


def get_default_config(model_type):
    """
    Get default hyperparameter configuration for each model type.
    
    Args:
        model_type: 'ebt', 'ffn', or 'pc'
    
    Returns:
        Dictionary of default hyperparameters
    """
    configs = {
        'ebt': {
            'n_optimization_steps': 10,
            'step_size': 1,
            'latent_dim': 128,
            'batch_size': 64,
            'num_epochs': 10000,
            'lr': 1e-3
        },
        'ffn': {
            'latent_dim': 128,
            'batch_size': 64,
            'num_epochs': 50000,
            'lr': 1e-4
        },
        'pc': {
            'latent_dim': 128,
            'batch_size': 64,
            'num_epochs': 100,
            'lr': 1e-3,
            'n_inference_steps': 20,
            'inference_lr': 0.1
        }
    }
    return configs.get(model_type, {})
