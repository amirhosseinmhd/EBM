"""
Main runner script for comparing EBT and Feedforward models on synthetic tasks.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.synthetic_data_generator import create_synthetic_data
from model.ebt_s1_toy import train_energy_model, test_energy_model
from model.ffn_toy import train_feedforward
from model.pc_add import train_predictive_coding


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
            'batch_size': 128,
            'num_epochs': 50000,
            'lr': 1e-4
        },
        'pc': {
            'latent_dim': 128,
            'batch_size': 64,
            'num_epochs': 10000,
            'lr': 1e-3,
            'n_inference_steps': 20,
            'inference_lr': 0.1
        }
    }
    return configs.get(model_type, {})


def run_main(model_type, task='add', vec_size=10, size_train=10000, 
             size_val=256, size_test=256, config=None):
    """
    Train and evaluate a single model on a synthetic task.
    
    Args:
        model_type: Which model to run ('ebt', 'ffn', or 'pc')
        task: Task type ('add', 'multiply', etc.)
        vec_size: Size of each vector
        size_train: Number of training samples
        size_val: Number of validation samples
        size_test: Number of test samples
        config: Dictionary of hyperparameters (uses defaults if None)
    
    Returns:
        Dictionary containing trained model and test MSE
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print(f"Training {model_type.upper()} Model on {task.upper()} Task")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    # Generate synthetic data
    print("\nGenerating synthetic datasets...")
    train_dataset, val_dataset, test_dataset, input_dim, output_dim = create_synthetic_data(
        task=task,
        size_train=size_train,
        size_val=size_val,
        size_test=size_test,
        vec_size=vec_size,
        device=device
    )
    print(f"✓ Dataset created: train={size_train}, val={size_val}, test={size_test}")
    print(f"✓ Input dim: {input_dim}, Output dim: {output_dim}\n")
    
    # Get model configuration
    model_config = get_default_config(model_type)
    if config:
        model_config.update(config)
    
    # Train the specified model
    if model_type == 'ebt':
        print("Training Energy-Based Transformer (EBT)...")
        model = train_energy_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            input_dim=input_dim,
            output_dim=output_dim,
            **model_config
        )
        test_mse = test_energy_model(
            model,
            n_optimization_steps=model_config['n_optimization_steps'],
            step_size=model_config['step_size']
        )
        
    elif model_type == 'ffn':
        print("Training Feedforward Network (FFN)...")
        model = train_feedforward(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            input_dim=input_dim,
            output_dim=output_dim,
            **model_config
        )
        model.eval()
        with torch.no_grad():
            x_test, y_test = test_dataset.tensors
            y_pred_test = model(x_test)
            test_mse = ((y_pred_test - y_test) ** 2).mean().item()
            
    elif model_type == 'pc':
        print("Training Predictive Coding (PC)...")
        model = train_predictive_coding(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            input_dim=input_dim,
            output_dim=output_dim,
            **model_config
        )
        model.eval()
        with torch.no_grad():
            x_test, y_test = test_dataset.tensors
            y_pred_test, _ = model(
                x_test,
                n_inference_steps=model_config['n_inference_steps'],
                inference_lr=model_config['inference_lr']
            )
            test_mse = ((y_pred_test - y_test) ** 2).mean().item()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'ebt', 'ffn', or 'pc'")
    
    print(f"\n✓ Training Complete | Test MSE: {test_mse:.6f}")
    print("=" * 80)
    
    return {
        'model': model,
        'test_mse': test_mse,
        'config': model_config
    }


def compare_models(models_to_run, task='add', vec_size=10, size_train=10000,
                   size_val=256, size_test=256, configs=None):
    """
    Wrapper function to train and compare multiple models.
    
    Args:
        models_to_run: List of model types to run, e.g., ['ebt', 'ffn', 'pc']
        task: Task type ('add', 'multiply', etc.)
        vec_size: Size of each vector
        size_train: Number of training samples
        size_val: Number of validation samples
        size_test: Number of test samples
        configs: Dictionary mapping model names to their configs, e.g.,
                {'ebt': {'num_epochs': 5000}, 'pc': {'n_inference_steps': 30}}
    
    Returns:
        Dictionary containing results for all models
    """
    if configs is None:
        configs = {}
    
    results = {}
    
    # Run each model
    for model_type in models_to_run:
        model_config = configs.get(model_type, None)
        result = run_main(
            model_type=model_type,
            task=task,
            vec_size=vec_size,
            size_train=size_train,
            size_val=size_val,
            size_test=size_test,
            config=model_config
        )
        results[model_type] = result
        print()  # Add spacing between models
    
    # Print comparison summary
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Task: {task.upper()}")
        print(f"Vector Size: {vec_size}")
        print(f"Training Samples: {size_train}")
        print(f"Validation Samples: {size_val}")
        print(f"Test Samples: {size_test} (Out-of-Distribution)")
        print("-" * 80)
        
        for model_name, data in results.items():
            print(f"{model_name.upper()} Model Test MSE: {data['test_mse']:.6f}")
        
        print("-" * 80)
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['test_mse'])
        best_name = best_model[0].upper()
        best_mse = best_model[1]['test_mse']
        print(f"Best Model: {best_name} (MSE: {best_mse:.6f})")
        
        # Show comparisons
        for name, data in results.items():
            if name != best_model[0]:
                diff = data['test_mse'] - best_mse
                ratio = (data['test_mse'] / best_mse - 1) * 100
                print(f"  {name.upper()} is {diff:.6f} worse than {best_name} ({ratio:+.2f}%)")
        
        print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Example 1: Run a single model with default settings
    result = run_main(
        model_type='ebt',
        task='add',
        vec_size=10
    )
    
    # Example 2: Run a single model with custom config
    # result = run_main(
    #     model_type='pc',
    #     task='add',
    #     vec_size=10,
    #     config={
    #         'n_inference_steps': 30,
    #         'inference_lr': 0.15,
    #         'num_epochs': 5000
    #     }
    # )
    
    # Example 3: Compare multiple models using the wrapper
    # results = compare_models(
    #     models_to_run=['ebt', 'pc', 'ffn'],
    #     task='add',
    #     vec_size=10
    # )
    
    # Example 4: Compare models with custom configs
    # results = compare_models(
    #     models_to_run=['ebt', 'pc'],
    #     task='add',
    #     vec_size=10,
    #     configs={
    #         'ebt': {'num_epochs': 5000, 'lr': 5e-4},
    #         'pc': {'n_inference_steps': 30, 'num_epochs': 5000}
    #     }
    # )
    
    # Example 5: Run on different task
    # result = run_main(
    #     model_type='ffn',
    #     task='multiply',
    #     vec_size=10
    # )
