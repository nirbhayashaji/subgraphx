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
    print("Initializing SubgraphX Anomaly Detection Ensemble...")
    
    df = pd.read_csv(data_path, low_memory=False)
    nodes = df['Node_ID'].values
    features = df.drop(columns=['Node_ID']).fillna(0)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)
    
    results_df = pd.DataFrame({'Node_ID': nodes})
    
    print("Training Models & Scoring Nodes...")
    
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=19)
    iso.fit(X_scaled)
    results_df['Score_IF'] = -iso.decision_function(X_scaled) 

    # Increased k to 500 to see past duplicate feature clusters in a 160k+ node network.
    # Added n_jobs=-1 to use all CPU cores and speed up the calculation.
    lof = LocalOutlierFactor(n_neighbors=500, contamination=0.01, n_jobs=-1)
    lof.fit_predict(X_scaled)
    results_df['Score_LOF'] = -lof.negative_outlier_factor_

    kmeans = KMeans(n_clusters=2, random_state=19, n_init=10)
    kmeans.fit(X_scaled)
    distances = kmeans.transform(X_scaled)
    results_df['Score_KMeans'] = [distances[i, label] for i, label in enumerate(kmeans.labels_)]

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
    results_df['Overall_Rank'] = results_df.index + 1
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    
    print(f"Detection complete! Results saved to: {out_path}")
    
    # --- RADAR CHART LOGIC FOR TOP 5 ---
    top5_df = results_df.head(5)
    
    # Convert Ranks to "Anomaly Percentiles" (100% = Most Anomalous) for the Radar Chart
    total_nodes = len(results_df)
    models = ['IF', 'LOF', 'KMeans', 'AE', 'Overall']
    
    angles = [n / float(len(models)) * 2 * np.pi for n in range(len(models))]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for index, row in top5_df.iterrows():
        pcts = [
            (1 - row['Rank_IF']/total_nodes)*100, 
            (1 - row['Rank_LOF']/total_nodes)*100, 
            (1 - row['Rank_KMeans']/total_nodes)*100, 
            (1 - row['Rank_AE']/total_nodes)*100,
            (1 - row['Overall_Rank']/total_nodes)*100
        ]
        pcts += pcts[:1] # Close the polygon
        ax.plot(angles, pcts, linewidth=2, linestyle='solid', label=f"Node {row['Node_ID']}")
        ax.fill(angles, pcts, alpha=0.1)

    plt.xticks(angles[:-1], models, fontweight='bold', size=11)
    ax.set_ylim(0, 100)
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100% (High Anomaly)'])
    
    plt.title("Model Consensus Radar: Top 5 Anomalous Nodes", size=15, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plot_path_top5 = out_path.parent / "plots" / "top_5_nodes_radar.png"
    plot_path_top5.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path_top5, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nTop 5 comparative Radar Chart saved to: {plot_path_top5}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent 
    INPUT_CSV = BASE_DIR / "results" / "master_node_features.csv"
    OUTPUT_CSV = BASE_DIR / "results" / "anomaly_ensemble_results.csv"
    
    TARGETS = ["513902872", "509903851", "516096826"]
    run_anomaly_ensemble(INPUT_CSV, OUTPUT_CSV, target_nodes=TARGETS)