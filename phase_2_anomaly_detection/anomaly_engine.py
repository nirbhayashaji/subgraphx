import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor

def run_anomaly_ensemble(data_path: Path | str, output_path: Path | str, target_nodes: list | None = None):
    print("Initializing Anomaly Detection Ensemble...")
    
    df = pd.read_csv(data_path, low_memory=False)
    nodes = df['Node_ID'].values
    features = df.drop(columns=['Node_ID']).fillna(0)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)
    
    results_df = pd.DataFrame({'Node_ID': nodes})
    
    print("Training Models & Scoring Nodes.")
    
    # 1. Isolation Forest (Matched to R: seed=19, ntrees=100)
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=19)
    iso.fit(X_scaled)
    results_df['Score_IF'] = -iso.decision_function(X_scaled) 

    # 2. Local Outlier Factor (Matched to R: minPts max search space ~30)
    # Note: We keep this at 100 to naturally solve the unique/duplicate issue 
    # the R script bypassed using the `unique()` function.
    lof = LocalOutlierFactor(n_neighbors=100, contamination=0.01)
    lof.fit_predict(X_scaled)
    results_df['Score_LOF'] = -lof.negative_outlier_factor_

    # 3. K-Means (Matched to R: centers=2, seed=19)
    kmeans = KMeans(n_clusters=2, random_state=19, n_init=10)
    kmeans.fit(X_scaled)
    distances = kmeans.transform(X_scaled)
    results_df['Score_KMeans'] = [distances[i, label] for i, label in enumerate(kmeans.labels_)]

    # 4. MLP Autoencoder (Unique to the new architecture, seed synced to 19)
    autoencoder = MLPRegressor(hidden_layer_sizes=(8, 4, 8), max_iter=500, random_state=19)
    autoencoder.fit(X_scaled, X_scaled)
    reconstructed = autoencoder.predict(X_scaled)
    results_df['Score_AE'] = np.mean(np.square(X_scaled - reconstructed), axis=1)

    print("Applying Borda Count Rank Aggregation...")
    rank_cols = []
    for model in ['IF', 'LOF', 'KMeans', 'AE']:
        score_col = f'Score_{model}'
        rank_col = f'Rank_{model}'
        results_df[rank_col] = results_df[score_col].rank(method='min', ascending=False)
        rank_cols.append(rank_col)
        
    results_df['Borda_Score'] = results_df[rank_cols].sum(axis=1)
    results_df = results_df.sort_values(by='Borda_Score', ascending=True).reset_index(drop=True)
    
    # Calculate the definitive Overall Rank (1 is the most anomalous)
    results_df['Overall_Rank'] = results_df.index + 1
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    
    print(f"Detection complete! Results saved to: {out_path}")
    
    # --- NEW: TOP 5 AI DISCOVERIES TRACKING & PLOTTING ---
    print("\n--- Top 5 Most Anomalous Nodes (Mathematical Consensus) ---")
    top5_df = results_df.head(5)
    print(top5_df[['Node_ID', 'Overall_Rank', 'Rank_IF', 'Rank_LOF', 'Rank_KMeans', 'Rank_AE', 'Borda_Score']])
    
    models = ['IF', 'LOF', 'KMeans', 'AE', 'Overall (Borda)']
    plt.figure(figsize=(10, 6))
    for index, row in top5_df.iterrows():
        ranks = [row['Rank_IF'], row['Rank_LOF'], row['Rank_KMeans'], row['Rank_AE'], row['Overall_Rank']]
        plt.plot(models, ranks, marker='o', linewidth=2, markersize=8, label=f"Node {row['Node_ID']}")

    plt.gca().invert_yaxis() # Invert so Rank 1 is at the top
    plt.title("Model Consensus Trajectory for Top 5 Anomalous Nodes\n(Higher on graph = More Anomalous)", fontsize=14, fontweight='bold')
    plt.ylabel("Anomaly Rank (out of Total Nodes)", fontsize=12)
    plt.xlabel("Detection Algorithm", fontsize=12)
    plt.legend(title="Top 5 Node IDs")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path_top5 = out_path.parent / "plots" / "top_5_nodes_rank_comparison.png"
    plot_path_top5.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path_top5, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nTop 5 comparative ranking plot saved to: {plot_path_top5}")

    # --- BUSINESS SUSPECT TRACKING & PLOTTING ---
    if target_nodes:
        print("\n--- Business Suspect Nodes: Ranking Analysis ---")
        suspect_df = results_df[results_df['Node_ID'].astype(str).isin(target_nodes)]
        
        if not suspect_df.empty:
            print(suspect_df[['Node_ID', 'Overall_Rank', 'Rank_IF', 'Rank_LOF', 'Rank_KMeans', 'Rank_AE']])
            
            plt.figure(figsize=(10, 6))
            for index, row in suspect_df.iterrows():
                ranks = [row['Rank_IF'], row['Rank_LOF'], row['Rank_KMeans'], row['Rank_AE'], row['Overall_Rank']]
                plt.plot(models, ranks, marker='o', linewidth=2, markersize=8, label=f"Node {row['Node_ID']}")

            plt.gca().invert_yaxis() # Invert so Rank 1 is at the top
            plt.title("Model Consensus Trajectory for Business Suspect Nodes\n(Higher on graph = More Anomalous)", fontsize=14, fontweight='bold')
            plt.ylabel("Anomaly Rank (out of Total Nodes)", fontsize=12)
            plt.xlabel("Detection Algorithm", fontsize=12)
            plt.legend(title="Suspect IDs")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plot_path_biz = out_path.parent / "plots" / "suspect_nodes_rank_comparison.png"
            plt.savefig(plot_path_biz, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Business Suspect comparative ranking plot saved to: {plot_path_biz}")
        else:
            print("Warning: None of the target nodes were found in the dataset.")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent 
    INPUT_CSV = BASE_DIR / "results" / "master_node_features.csv"
    OUTPUT_CSV = BASE_DIR / "results" / "anomaly_ensemble_results.csv"
    
    TARGETS = ["513902872", "509903851", "516096826"]
    run_anomaly_ensemble(INPUT_CSV, OUTPUT_CSV, target_nodes=TARGETS)