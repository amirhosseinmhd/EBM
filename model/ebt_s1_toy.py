import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.synthetic_data_generator import create_synthetic_data


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super(EncoderLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class EnergyFunction(torch.nn.Module):
    """Energy function E_θ(x, ŷ) - returns scalar energy value"""
    def __init__(self, input_dim, output_dim, latent_dim=128):
        super(EnergyFunction, self).__init__()
        combined_dim = input_dim + output_dim
        self.encoder = EncoderLayer(combined_dim, latent_dim)
        self.activation = torch.nn.SiLU()
        self.linear = torch.nn.Linear(latent_dim, 1)

    def forward(self, x, y_hat):
        """
        Args:
            x: context/input
            y_hat: candidate solution
        Returns:
            energy: scalar energy value (lower is better)
        """
        combined = torch.cat([x, y_hat], dim=-1)
        h = self.encoder(combined)
        h = self.activation(h)
        energy = self.linear(h)
        return energy


def optimize_energy(energy_fn, x, y_init, n_steps=5, step_size=0.1, track_theta_grad=True):
    """
    Simplified Energy Optimization with First-Order Approximation
    
    Only the LAST step maintains gradients w.r.t. θ (when track_theta_grad=True).
    Steps 0 to N-2 are treated as "constants" (detached from θ gradients).
    
    Args:
        track_theta_grad: If True, track gradients w.r.t. θ on last step (training mode)
                         If False, don't track any θ gradients (eval mode)
    
    This implements: ŷ_{i+1} = ŷ_i - α * g_θ(ŷ_i)
    where g_θ(ŷ) = ∇_ŷ E_θ(x, ŷ)
    """
    y_hat = y_init.clone()
    
    # Steps 0 to N-2: Update y_hat but DON'T track gradients w.r.t. θ
    for step in range(n_steps - 1):
        # Enable gradients for y_hat so we can compute ∇_ŷ E_θ
        y_hat = y_hat.detach().requires_grad_(True)
        
        # Compute energy
        energy = energy_fn(x, y_hat).sum()
        
        # Compute gradient w.r.t. y_hat (not w.r.t. θ)
        grad_y = torch.autograd.grad(energy, y_hat, create_graph=False)[0]
        
        # Update y_hat (this breaks the gradient connection to θ)
        y_hat = y_hat.detach() - step_size * grad_y
    
    # Last step (N-1): Optionally track gradients w.r.t. θ
    y_hat = y_hat.detach().requires_grad_(True)
    
    # Compute energy with create_graph=track_theta_grad
    energy = energy_fn(x, y_hat).sum()
    grad_y = torch.autograd.grad(energy, y_hat, create_graph=track_theta_grad)[0]
    
    # Final update
    y_final = y_hat - step_size * grad_y
    
    return y_final


def train_energy_model(train_dataset, val_dataset, test_dataset, input_dim, output_dim, 
                       n_optimization_steps=10, step_size=1, latent_dim=128, 
                       batch_size=64, num_epochs=10000, lr=1e-3):
    """Training loop for energy-based model"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Move validation and test datasets to device
    x_val, y_val = val_dataset.tensors
    x_test, y_test = test_dataset.tensors
    
    # Model and optimizer
    energy_fn = EnergyFunction(input_dim, output_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(energy_fn.parameters(), lr=lr)
    
    print("\nTraining Energy-Based Model (First-Order Approximation)")
    print("=" * 60)
    print(f"Only tracking gradients through LAST optimization step")
    print(f"Total optimization steps: {n_optimization_steps}")
    print(f"Steps with θ-gradients: 1 (the last step only)")
    print("=" * 60)
    
    epoch = 0
    while epoch < num_epochs:
        energy_fn.train()
        
        for x, y_true in train_loader:
            # Initialize from random guess
            y_init = torch.randn(x.size(0), output_dim, device=device) * 2 - 1
            
            # Optimize through energy minimization (WITH θ gradients for training)
            y_final = optimize_energy(energy_fn, x, y_init, n_optimization_steps, step_size, track_theta_grad=True)
            
            # Supervised loss: ||ŷ_N - y*||²
            loss = (y_final - y_true).pow(2).mean()
            
            # Backprop: only through the last optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Evaluate on validation and test sets
            if epoch % 250 == 0:
                energy_fn.eval()
                
                # Training batch info
                with torch.no_grad():
                    energy_init = energy_fn(x, y_init).mean()
                    energy_final = energy_fn(x, y_final.detach()).mean()
                
                # Validation loss (NO θ gradients needed)
                val_size = len(val_dataset)
                y_init_val = torch.randn(val_size, output_dim, device=device) * 2 - 1
                y_pred_val = optimize_energy(energy_fn, x_val, y_init_val, n_optimization_steps, step_size, track_theta_grad=False)
                with torch.no_grad():
                    val_loss = (y_pred_val.detach() - y_val).pow(2).mean()
                
                # Test loss (NO θ gradients needed)
                test_size = len(test_dataset)
                y_init_test = torch.randn(test_size, output_dim, device=device) * 2 - 1
                y_pred_test = optimize_energy(energy_fn, x_test, y_init_test, n_optimization_steps, step_size, track_theta_grad=False)
                with torch.no_grad():
                    test_loss = (y_pred_test.detach() - y_test).pow(2).mean()
                
                print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | Test Loss: {test_loss.item():.6f} | "
                      f"Energy: {energy_init.item():.4f} → {energy_final.item():.4f}")
                energy_fn.train()
            
            epoch += 1
            if epoch >= num_epochs:
                break
        
        if epoch >= num_epochs:
            break
    
    print("\n" + "=" * 60)
    print("Training complete!")
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    energy_fn.eval()
    
    # Validation set
    val_size = len(val_dataset)
    y_init_val = torch.randn(val_size, output_dim, device=device) * 2 - 1
    y_pred_val = optimize_energy(energy_fn, x_val, y_init_val, n_optimization_steps, step_size, track_theta_grad=False)
    with torch.no_grad():
        val_loss_final = (y_pred_val.detach() - y_val).pow(2).mean()
    print(f"Final Validation MSE: {val_loss_final.item():.6f}")
    
    # Test set
    test_size = len(test_dataset)
    y_init_test = torch.randn(test_size, output_dim, device=device) * 2 - 1
    y_pred_test = optimize_energy(energy_fn, x_test, y_init_test, n_optimization_steps, step_size, track_theta_grad=False)
    with torch.no_grad():
        test_loss_final = (y_pred_test.detach() - y_test).pow(2).mean()
    print(f"Final Test MSE: {test_loss_final.item():.6f}")

    return energy_fn


def test_energy_model(energy_fn, n_optimization_steps=50, step_size=0.1):
    """Test-time inference with energy minimization"""
    print("\nTesting with iterative energy minimization...")
    print("=" * 60)
    
    # Get device from model
    device = next(energy_fn.parameters()).device
    
    # Generate test data
    vec_size = 10
    n_test = 100
    v1_test = torch.randn(n_test, vec_size, device=device) * 4 - 2
    v2_test = torch.randn(n_test, vec_size, device=device) * 4 - 2
    x_test = torch.cat([v1_test, v2_test], dim=1)
    y_true_test = v1_test + v2_test
    
    energy_fn.eval()
    
    # Initialize from random guess
    y_hat = torch.randn(n_test, vec_size, device=device) * 2 - 1
    
    print("Iterative refinement (test time):")
    for step in range(n_optimization_steps):
        # Enable gradients for y_hat
        y_hat = y_hat.detach().requires_grad_(True)
        
        # Compute energy and gradient
        energy = energy_fn(x_test, y_hat).sum()
        grad_y = torch.autograd.grad(energy, y_hat)[0]
        
        # Update
        y_hat = (y_hat - step_size * grad_y).detach()
        
        if step % 10 == 0:
            with torch.no_grad():
                mse = (y_hat - y_true_test).pow(2).mean()
                energy_val = energy_fn(x_test, y_hat).mean()
            print(f"Step {step:3d} | Energy: {energy_val.item():.6f} | MSE: {mse.item():.6f}")
    
    # Final evaluation
    with torch.no_grad():
        final_mse = (y_hat - y_true_test).pow(2).mean()
    print(f"\nFinal Test MSE: {final_mse.item():.6f}")
    
    # Show example
    print(f"\nExample (first 5 elements):")
    print(f"v1:        {v1_test[0][:5].cpu().numpy()}")
    print(f"v2:        {v2_test[0][:5].cpu().numpy()}")
    print(f"True sum:  {y_true_test[0][:5].cpu().numpy()}")
    print(f"Predicted: {y_hat[0][:5].cpu().numpy()}")
    
    return final_mse.item()


if __name__ == "__main__":
    # Configuration
    task = 'multiply'  # 'addition' or 'multiplication'
    vec_size = 10
    size_train = 10000
    size_val = 256
    size_test = 256
    n_optimization_steps = 20
    step_size = 100
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate synthetic data
    print("Generating synthetic datasets...")
    train_dataset, val_dataset, test_dataset, input_dim, output_dim = create_synthetic_data(
        task=task,
        size_train=size_train,
        size_val=size_val,
        size_test=size_test,
        vec_size=vec_size,
        device=device
    )
    print(f"Dataset created: train={size_train}, val={size_val}, test={size_test}")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}\n")
    
    # Train energy-based model
    energy_fn = train_energy_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        input_dim=input_dim,
        output_dim=output_dim,
        n_optimization_steps=n_optimization_steps,
        step_size=step_size
    )
    
    # Optional: Test with more iterative refinement steps
    test_mse = test_energy_model(energy_fn, n_optimization_steps=n_optimization_steps, step_size=step_size)
    print("\n" + "=" * 60)
    print(f"Extended Test MSE: {test_mse:.6f}")
    print("=" * 60)