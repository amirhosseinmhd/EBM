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
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class FFNAdd(torch.nn.Module):
    def __init__(self, input_dim, latent_dim=128, output_dim=10):
        super(FFNAdd, self).__init__()
        self.encoder = EncoderLayer(input_dim, latent_dim)
        self.activation = torch.nn.ReLU()
        self.linear = torch.nn.Linear(latent_dim, output_dim)  

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        y_hat = self.linear(x)
        return y_hat


# Simple training function
def train_feedforward(train_dataset, val_dataset, test_dataset, input_dim, output_dim,
                     latent_dim=128, batch_size=128, num_epochs=50000, lr=1e-4):
    """Training loop for feedforward network"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Move validation and test datasets to device
    x_val, y_val = val_dataset.tensors
    x_test, y_test = test_dataset.tensors
    
    # Model
    model = FFNAdd(input_dim, latent_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print("\nTraining Feedforward Network")
    print("=" * 60)
    
    # Training loop
    epoch = 0
    while epoch < num_epochs:
        model.train()
        
        for x, y_true in train_loader:
            # Forward pass
            y_pred = model(x)
            loss = criterion(y_pred, y_true)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Evaluate on validation and test sets
            if epoch % 250 == 0:
                model.eval()
                with torch.no_grad():
                    # Validation loss
                    y_pred_val = model(x_val)
                    val_loss = criterion(y_pred_val, y_val)
                    
                    # Test loss
                    y_pred_test = model(x_test)
                    test_loss = criterion(y_pred_test, y_test)
                
                print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | Test Loss: {test_loss.item():.6f}")
                model.train()
            
            epoch += 1
            if epoch >= num_epochs:
                break
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    model.eval()
    with torch.no_grad():
        # Validation set
        y_pred_val = model(x_val)
        val_loss_final = criterion(y_pred_val, y_val)
        print(f"Final Validation MSE: {val_loss_final.item():.6f}")
        
        # Test set
        y_pred_test = model(x_test)
        test_loss_final = criterion(y_pred_test, y_test)
        print(f"Final Test MSE: {test_loss_final.item():.6f}")
        
        # Show some examples from test set
        print("\n--- Sample Predictions from Test Set ---")
        for i in range(5):
            print(f"Pred: {y_pred_test[i].cpu().numpy()}, True: {y_test[i].cpu().numpy()}")
    
    return model


if __name__ == "__main__":
    # Configuration
    task = 'add'
    vec_size = 100
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
    
    # Train feedforward network
    model = train_feedforward(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        input_dim=input_dim,
        output_dim=output_dim
    )