import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Add parent directory to path so we can import Phase 1 tools
sys.path.append(str(Path(__file__).resolve().parent.parent))
from processing import load_and_clean_data
from network_logic import build_transfer_network

# Import the new GNN architecture
from phase_2_anomaly_detection.model_ae import AnomalyGAE

def train_gnn_model():
    print("Initializing PyTorch Geometric (GNN) training...")

    # Setup paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_CSV = BASE_DIR / "results" / "master_node_features.csv"
    RAW_DATA_CSV = BASE_DIR / "data" / "raw_network_transactions.csv"
    
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = MODELS_DIR / "anomaly_ae_model.pth"
    SCALER_PATH = MODELS_DIR / "scaler.joblib"

    # --- 1. LOAD FEATURES ---
    print("Loading master feature dataset...")
    df_features = pd.read_csv(INPUT_CSV, dtype={'Node_ID': str}, low_memory=False)
    
    # CRITICAL GNN STEP: PyTorch Geometric requires nodes to be indexed from 0 to N-1.
    # We create a dictionary mapping your string Node_IDs to strict integers.
    node_to_idx = {node_id: idx for idx, node_id in enumerate(df_features['Node_ID'])}
    
    features_only = df_features.drop(columns=['Node_ID']).fillna(0)
    input_dim = features_only.shape[1]

    print("Scaling data...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features_only)
    joblib.dump(scaler, SCALER_PATH)
    
    x_tensor = torch.FloatTensor(X_scaled)

    # --- 2. EXTRACT GRAPH EDGES ---
    print("Loading raw transactions to extract network edges...")
    df_raw = load_and_clean_data(str(RAW_DATA_CSV))
    G = build_transfer_network(df_raw)
    
    print("Mapping edges to PyTorch Geometric format...")
    source_indices = []
    target_indices = []
    
    for u, v in G.edges():
        if str(u) in node_to_idx and str(v) in node_to_idx:
            source_indices.append(node_to_idx[str(u)])
            target_indices.append(node_to_idx[str(v)])
            
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    print(f"Graph constructed with {x_tensor.size(0)} nodes and {edge_index.size(1)} edges.")

    # --- 3. INITIALIZE MODEL & TRAINING ---
    model = AnomalyGAE(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Full-batch training for GNNs usually needs a few more epochs
    epochs = 100 
    print(f"Starting full-batch training loop for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass now takes BOTH features and edges!
        outputs = model(x_tensor, edge_index)
        loss = criterion(outputs, x_tensor)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    # Save outputs
    torch.save(model.state_dict(), MODEL_PATH)
    print("Training complete.")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Model weights saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_gnn_model()