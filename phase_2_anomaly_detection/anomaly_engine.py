import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor # Used here as a simple Autoencoder

def run_anomaly_ensemble(data_path: Path | str, output_path: Path | str):
    print(" Initializing Anomaly Detection Ensemble...")
    
    # 1. Load Data
    # Adding low_memory=False to silence the DtypeWarning you saw earlier!
    df = pd.read_csv(data_path, low_memory=False)
    nodes = df['Node_ID'].values
    features = df.drop(columns=['Node_ID']).fillna(0)
    
    # 2. Scale Features (Crucial for distance-based models and NNs)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)
    
    # DataFrame to hold our scores
    results_df = pd.DataFrame({'Node_ID': nodes})
    
    print(" Training Models & Scoring Nodes...")

    # --- MODEL 1: Isolation Forest ---
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X_scaled)
    # Invert scores so HIGHER = More Anomalous
    results_df['Score_IF'] = -iso.decision_function(X_scaled) 

    # --- MODEL 2: Local Outlier Factor (LOF) ---
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof.fit_predict(X_scaled)
    # Invert scores so HIGHER = More Anomalous
    results_df['Score_LOF'] = -lof.negative_outlier_factor_

    # --- MODEL 3: K-Means Clustering ---
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    # Anomaly score is the distance to the assigned cluster center
    distances = kmeans.transform(X_scaled)
    results_df['Score_KMeans'] = [distances[i, label] for i, label in enumerate(kmeans.labels_)]

    # --- MODEL 4: Simple Autoencoder (MLP) ---
    autoencoder = MLPRegressor(hidden_layer_sizes=(8, 4, 8), max_iter=500, random_state=42)
    autoencoder.fit(X_scaled, X_scaled) # Train to predict itself
    reconstructed = autoencoder.predict(X_scaled)
    # Anomaly score is the Mean Squared Error of reconstruction
    results_df['Score_AE'] = np.mean(np.square(X_scaled - reconstructed), axis=1)

    print(" Applying Borda Count Rank Aggregation...")

    # --- BORDA COUNT IMPLEMENTATION ---
    rank_cols = []
    for model in ['IF', 'LOF', 'KMeans', 'AE']:
        score_col = f'Score_{model}'
        rank_col = f'Rank_{model}'
        results_df[rank_col] = results_df[score_col].rank(method='min')
        rank_cols.append(rank_col)
        
    # The Borda Score is the sum of the ranks across all 4 models
    results_df['Borda_Score'] = results_df[rank_cols].sum(axis=1)
    
    # Sort by the final Borda Score
    results_df = results_df.sort_values(by='Borda_Score', ascending=False).reset_index(drop=True)
    
    # Save the results
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    
    print(f" Detection complete! Results saved to: {out_path}")
    print("\n Top 5 Most Anomalous Nodes (Consensus):")
    print(results_df[['Node_ID', 'Borda_Score']].head(5))

if __name__ == "__main__":
    # Dynamically find the root project folder so paths always work!
    BASE_DIR = Path(__file__).resolve().parent.parent 
    
    INPUT_CSV = BASE_DIR / "results" / "master_node_features.csv"
    OUTPUT_CSV = BASE_DIR / "results" / "anomaly_ensemble_results.csv"
    
    run_anomaly_ensemble(INPUT_CSV, OUTPUT_CSV)