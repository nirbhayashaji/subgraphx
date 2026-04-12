import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from model_ae import AnomalyAE

def detect_anomalies(metrics_df: pd.DataFrame, output_dir: Path, epochs=100, lr=0.005):
    features = [
        'in_degree', 'betweenesscentrality', 'eigencentrality', 
        'degree_imbalance_ratio', 'pageranks', 'all_degree', 
        'dangerous_quantity_difference_ratio', 'proportion_individual_suppliers',
        'disposed_quantity_ratio'
    ]
    
    data = metrics_df[features].fillna(0).values
    scaler = StandardScaler()
    X = torch.FloatTensor(scaler.fit_transform(data))
    
    model = AnomalyAE(input_dim=len(features))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
  
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, X)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  > Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            
    model.eval()
    with torch.no_grad():
        reconstructed = model(X)
        errors = torch.mean((X - reconstructed)**2, dim=1).numpy()
        
    metrics_df['anomaly_score'] = errors
    
    joblib.dump(scaler, output_dir / "scaler.joblib")
    torch.save(model.state_dict(), output_dir / "model_ae.pth")
    
    return metrics_df.sort_values('anomaly_score', ascending=False), model, scaler, features