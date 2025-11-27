"""
Predictive Coding Network for synthetic tasks - CORRECTED VERSION

This implements a hierarchical predictive coding network with:
- Top-down predictions
- Bottom-up error propagation
- Iterative inference through prediction error minimization
- Proper gradient descent for state updates
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.synthetic_data_generator import create_synthetic_data


class PredictiveLayer(nn.Module):
    """
    Single layer in the predictive coding hierarchy.
    Each layer maintains its own state and computes prediction errors.
    """
    def __init__(self, input_dim, output_dim, activation='silu'):
        super(PredictiveLayer, self).__init__()
        # Top-down prediction weights (from higher layer to this layer)
        self.prediction_weights = nn.Linear(output_dim, input_dim)
        
        # Activation function
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
        
    def predict_down(self, higher_state):
        """Generate top-down prediction from higher layer."""
        return self.activation(self.prediction_weights(higher_state))
    



class PredictiveCodingNetwork(nn.Module):
    """
    Hierarchical Predictive Coding Network.
    
    Architecture:
    - Input x -> state1 (input_dim -> latent_dim)
    - state1 -> state2 (latent_dim -> latent_dim)
    - state2 -> state3 (latent_dim -> latent_dim)
    - state3 -> Output (latent_dim -> output_dim)
    
    Energy function:
    E = 0.5 * (||e0||² + ||e1||² + ||e2||² + ||e_output||²)
    
    where:
    - e0 = x - prediction_of_x_from_state1
    - e1 = state1 - prediction_of_state1_from_state2
    - e2 = state2 - prediction_of_state2_from_state3
    - e_output = y_target - output_from_state3
    """
    def __init__(self, input_dim, output_dim, latent_dim=128, activation='silu'):
        super(PredictiveCodingNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        # Define the hierarchical layers
        # Each layer predicts the layer BELOW it (top-down)
        self.layer1 = PredictiveLayer(input_dim, latent_dim, activation)
        self.layer2 = PredictiveLayer(latent_dim, latent_dim, activation)
        self.layer3 = PredictiveLayer(latent_dim, latent_dim, activation)
        
        # Output layer: state3 -> output
        self.output_layer = nn.Linear(latent_dim, output_dim)
        
        # Store activation for derivative computation
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
    
    def initialize_states(self, batch_size, device):
        """Initialize hidden states for all layers."""
        state1 = torch.randn(batch_size, self.latent_dim, device=device, requires_grad=False)
        state2 = torch.randn(batch_size, self.latent_dim, device=device, requires_grad=False)
        state3 = torch.randn(batch_size, self.latent_dim, device=device, requires_grad=False)
        return state1, state2, state3
    
    def compute_errors(self, x, y_target, state1, state2, state3):
        """
        Compute prediction errors at all layers.
        
        Returns:
            errors: (error0, error1, error2, error_output)
            predictions: (pred0, pred1, pred2, output)
        """
        # Top-down predictions
        pred2 = self.layer3.predict_down(state3)  # state3 predicts state2
        pred1 = self.layer2.predict_down(state2)  # state2 predicts state1
        pred0 = self.layer1.predict_down(state1)  # state1 predicts input x
        
        # Prediction errors
        error2 = state2 - pred2
        error1 = state1 - pred1
        error0 = x - pred0
        
        # Output
        output = self.output_layer(state3)
        if y_target is not None:
            error_output = y_target - output
        else:
            error_output = None
        
        return (error0, error1, error2, error_output), (pred0, pred1, pred2, output)
    
    def activation_derivative(self, x):
        """
        Compute derivative of activation function.
        For common activations, we can compute this efficiently.
        """
        if isinstance(self.activation, nn.Tanh):
            return 1 - torch.tanh(x).pow(2)
        elif isinstance(self.activation, nn.ReLU):
            return (x > 0).float()
        elif isinstance(self.activation, nn.SiLU):
            # SiLU: f(x) = x * sigmoid(x)
            # f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            sigmoid_x = torch.sigmoid(x)
            return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
        else:
            return torch.ones_like(x)
    
    def inference_step(self, x, y_target, state1, state2, state3, inference_lr=0.1):
        """
        Single step of predictive coding inference using GRADIENT DESCENT.
        
        Updates states to minimize the total prediction error energy.
        
        Energy: E = 0.5 * (||e0||² + ||e1||² + ||e2||² + ||e_output||²)
        
        Update rule: state_i = state_i - lr * ∂E/∂state_i
        
        Args:
            x: Input data
            y_target: Target output (can be None during unsupervised inference)
            state1, state2, state3: Current states
            inference_lr: Learning rate for inference
        """
        # Compute all errors and predictions
        (error0, error1, error2, error_output), (pred0, pred1, pred2, output) = \
            self.compute_errors(x, y_target, state1, state2, state3)
        
        # Compute pre-activations for derivative
        z1 = self.layer1.prediction_weights(state1)
        z2 = self.layer2.prediction_weights(state2)
        z3 = self.layer3.prediction_weights(state3)
        
        # Activation derivatives
        deriv0 = self.activation_derivative(z1)
        deriv1 = self.activation_derivative(z2)
        deriv2 = self.activation_derivative(z3)
        
        # Gradient of energy with respect to each state
        # ∂E/∂state_i = error_i - (φ'(z_{i-1}) ⊙ error_{i-1}) W_i
        #
        # For batches:
        # - error_i has shape (batch, dim_i)
        # - W_i has shape (dim_{i-1}, dim_i) in the Linear layer's weight matrix
        # - (deriv_{i-1} * error_{i-1}) @ W_i gives shape (batch, dim_i)
        
        # For state1:
        # - Contributes to error1 (being predicted by state2)
        # - Predicts input via error0
        grad_state1 = error1 - (deriv0 * error0) @ self.layer1.prediction_weights.weight
        
        # For state2:
        # - Contributes to error2 (being predicted by state3)
        # - Predicts state1 via error1
        grad_state2 = error2 - (deriv1 * error1) @ self.layer2.prediction_weights.weight
        
        # For state3:
        # - Predicts state2 via error2
        # - Produces output (if y_target provided, also gets supervised signal)
        grad_state3 = -(deriv2 * error2) @ self.layer3.prediction_weights.weight
        
        # Add supervised signal to top layer if target is provided
        if y_target is not None and error_output is not None:
            # Gradient from output layer: W_out^T @ error_output
            # output_layer.weight has shape (output_dim, latent_dim)
            # We need (batch, latent_dim), so: error_output @ output_layer.weight
            grad_state3 = grad_state3 + error_output @ self.output_layer.weight
        
        # GRADIENT DESCENT: minimize energy
        state1_new = state1 - inference_lr * grad_state1
        state2_new = state2 - inference_lr * grad_state2
        state3_new = state3 - inference_lr * grad_state3
        
        # Compute total energy
        total_energy = 0.5 * (
            error0.pow(2).sum(dim=1).mean() + 
            error1.pow(2).sum(dim=1).mean() + 
            error2.pow(2).sum(dim=1).mean()
        )
        if error_output is not None:
            total_energy += 0.5 * error_output.pow(2).sum(dim=1).mean()
        
        return state1_new, state2_new, state3_new, total_energy
    
    def forward(self, x, y_target=None, n_inference_steps=20, inference_lr=0.1):
        """
        Forward pass with iterative inference.
        
        Args:
            x: Input tensor (batch, input_dim)
            y_target: Target output for supervised inference (batch, output_dim)
            n_inference_steps: Number of inference iterations
            inference_lr: Learning rate for inference
        
        Returns:
            output: Predicted output (batch, output_dim)
            final_energy: Final prediction error energy
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize states
        state1, state2, state3 = self.initialize_states(batch_size, device)
        
        # Iterative inference
        for step in range(n_inference_steps):
            state1, state2, state3, total_energy = self.inference_step(
                x, y_target, state1, state2, state3, inference_lr
            )
        
        # Generate output from top layer
        output = self.output_layer(state3)
        
        return output, total_energy
    
    def forward_with_tracking(self, x, y_target=None, n_inference_steps=20, inference_lr=0.1):
        """
        Forward pass with energy tracking (for analysis).
        
        Returns:
            output: Predicted output
            energies: List of total energy at each step
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize states
        state1, state2, state3 = self.initialize_states(batch_size, device)
        
        energies = []
        
        # Iterative inference
        for step in range(n_inference_steps):
            state1, state2, state3, total_energy = self.inference_step(
                x, y_target, state1, state2, state3, inference_lr
            )
            energies.append(total_energy.item())
        
        # Generate output from top layer
        output = self.output_layer(state3)
        
        return output, energies


def train_predictive_coding(train_dataset, val_dataset, test_dataset, input_dim, output_dim,
                            latent_dim=128, batch_size=64, num_epochs=10000, lr=1e-3,
                            n_inference_steps=20, inference_lr=0.1):
    """
    Training loop for predictive coding network.
    
    Two approaches:
    1. Inference with target (supervised): Include y_target during inference
    2. Inference without target, then supervised loss: Run inference freely, then backprop
    
    Here we use approach 1: supervised inference
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Move datasets to device
    x_val, y_val = val_dataset.tensors
    x_test, y_test = test_dataset.tensors
    
    # Model and optimizer
    model = PredictiveCodingNetwork(input_dim, output_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print("\nTraining Predictive Coding Network (CORRECTED)")
    print("=" * 60)
    print(f"Architecture: {input_dim} -> {latent_dim} -> {latent_dim} -> {latent_dim} -> {output_dim}")
    print(f"Inference steps per forward pass: {n_inference_steps}")
    print(f"Inference learning rate: {inference_lr}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        
        # Sample random batch from training dataset
        indices = torch.randint(0, len(train_dataset), (batch_size,))
        x_batch = train_dataset.tensors[0][indices]
        y_batch = train_dataset.tensors[1][indices]
        
        # Forward pass with inference (INCLUDING target for supervised inference)
        y_pred, energy = model(x_batch, y_target=y_batch, 
                              n_inference_steps=n_inference_steps, 
                              inference_lr=inference_lr)
        
        # Supervised loss on output
        loss = criterion(y_pred, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate on validation and test sets
        if epoch % 250 == 0:
            model.eval()
            with torch.no_grad():
                # Validation loss
                y_pred_val, _ = model(x_val, y_target=None, 
                                     n_inference_steps=n_inference_steps,
                                     inference_lr=inference_lr)
                val_loss = criterion(y_pred_val, y_val)
                
                # Test loss
                y_pred_test, _ = model(x_test, y_target=None,
                                      n_inference_steps=n_inference_steps,
                                      inference_lr=inference_lr)
                test_loss = criterion(y_pred_test, y_test)
            
            print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | "
                  f"Test Loss: {test_loss.item():.6f} | Energy: {energy.item():.6f}")
            model.train()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    model.eval()
    with torch.no_grad():
        # Validation set
        y_pred_val, _ = model(x_val, y_target=None,
                             n_inference_steps=n_inference_steps,
                             inference_lr=inference_lr)
        val_loss_final = criterion(y_pred_val, y_val)
        print(f"Final Validation MSE: {val_loss_final.item():.6f}")
        
        # Test set
        y_pred_test, energies = model.forward_with_tracking(
            x_test, y_target=None, n_inference_steps=n_inference_steps, 
            inference_lr=inference_lr
        )
        test_loss_final = criterion(y_pred_test, y_test)
        print(f"Final Test MSE: {test_loss_final.item():.6f}")
        
        # Show inference convergence
        print(f"\n--- Inference Convergence (Test Set) ---")
        print(f"Initial energy: {energies[0]:.6f}")
        print(f"Final energy: {energies[-1]:.6f}")
        print(f"Reduction: {(energies[0] - energies[-1]) / energies[0] * 100:.1f}%")
        
        # Show some examples from test set
        print("\n--- Sample Predictions from Test Set ---")
        for i in range(min(5, len(y_test))):
            pred_str = ', '.join([f"{v:.4f}" for v in y_pred_test[i].cpu().numpy()[:5]])
            true_str = ', '.join([f"{v:.4f}" for v in y_test[i].cpu().numpy()[:5]])
            print(f"Pred: [{pred_str}...], True: [{true_str}...]")
    
    return model


def test_predictive_coding(model, n_inference_steps=50, inference_lr=0.1):
    """Test-time inference with more iterations."""
    print("\nTesting with extended inference...")
    print("=" * 60)
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Generate test data
    vec_size = 10
    n_test = 100
    v1_test = torch.randn(n_test, vec_size, device=device) * 4 - 2
    v2_test = torch.randn(n_test, vec_size, device=device) * 4 - 2
    x_test = torch.cat([v1_test, v2_test], dim=1)
    y_true_test = v1_test + v2_test
    
    model.eval()
    
    with torch.no_grad():
        # Track inference process
        y_pred, energies = model.forward_with_tracking(
            x_test, y_target=None, n_inference_steps=n_inference_steps, 
            inference_lr=inference_lr
        )
        
        print("Inference convergence:")
        for step in [0, 10, 20, 30, 40, n_inference_steps-1]:
            if step < len(energies):
                mse = ((y_pred - y_true_test).pow(2).mean()).item()
                print(f"Step {step:3d} | Energy: {energies[step]:.6f} | MSE: {mse:.6f}")
        
        # Final evaluation
        final_mse = ((y_pred - y_true_test).pow(2).mean()).item()
        print(f"\nFinal Test MSE: {final_mse:.6f}")
        
        # Show example
        print(f"\nExample (first 5 elements):")
        print(f"v1:        {v1_test[0][:5].cpu().numpy()}")
        print(f"v2:        {v2_test[0][:5].cpu().numpy()}")
        print(f"True sum:  {y_true_test[0][:5].cpu().numpy()}")
        print(f"Predicted: {y_pred[0][:5].cpu().numpy()}")
    
    return final_mse


if __name__ == "__main__":
    # Configuration
    task = 'add'
    vec_size = 10
    size_train = 10000
    size_val = 256
    size_test = 256
    
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
    
    # Train predictive coding network
    model = train_predictive_coding(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=128,
        n_inference_steps=20,
        inference_lr=0.05,
        lr=1e-3
    )
    
    # Optional: Test with more inference steps
    test_mse = test_predictive_coding(model, n_inference_steps=50, inference_lr=0.1)
    print("\n" + "=" * 60)
    print(f"Extended Test MSE: {test_mse:.6f}")
    print("=" * 60)