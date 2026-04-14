import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Import unified architecture
from model_ae import AnomalyAE

def train_model():
    print("Initializing PyTorch Autoencoder training...")
    
    # Setup paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_CSV = BASE_DIR / "results" / "master_node_features.csv"
    
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = MODELS_DIR / "anomaly_ae_model.pth"
    SCALER_PATH = MODELS_DIR / "scaler.joblib"

    # Load and prepare data
    print("Loading master feature dataset...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # Drop ID column to isolate numerical features
    features_df = df.drop(columns=['Node_ID']).fillna(0)
    input_dim = features_df.shape[1]
    print(f"Detected {input_dim} features for training.")

    print("Scaling data...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features_df)
    
    # Save the scaler for the explainer script
    joblib.dump(scaler, SCALER_PATH)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.FloatTensor(X_scaled)
    dataset = TensorDataset(X_tensor, X_tensor) 
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize model & training parameters
    model = AnomalyAE(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    print(f"Starting training loop for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_features, _ in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.6f}")

    # Save outputs
    torch.save(model.state_dict(), MODEL_PATH)
    print("Training complete.")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Model weights saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()