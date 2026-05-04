import sys
import torch
import joblib
import pandas as pd
import networkx as nx
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from processing import load_and_clean_data
from network_logic import build_transfer_network
from phase_2_anomaly_detection.model_ae import AnomalyGAE 
from phase_2_anomaly_detection.visualization_logic import (
    draw_shapley_evidence_map, 
    draw_business_suspect_comparison
)

def calculate_marginal_contribution(model, scaler, metrics_df, features, G, target_node, edge_index):
    """GNN Message Passing to calculate influence."""
    model.eval()
    ego_graph = nx.ego_graph(G, target_node, radius=1, undirected=True)
    neighbors = list(ego_graph.nodes())
    if target_node in neighbors:
        neighbors.remove(target_node)
    
    def get_score_for_target(df_state):
        data = df_state[features].fillna(0)
        X = torch.FloatTensor(scaler.transform(data))
        with torch.no_grad():
            reconstructed = model(X, edge_index)
            errors = torch.mean((X - reconstructed)**2, dim=1)
        try:
            target_idx = df_state.index.get_loc(df_state[df_state['Node_ID'] == target_node].index[0])
            return errors[target_idx].item()
        except:
            return 0.0

    base_state_df = metrics_df.copy()
    f_Si_Gi = get_score_for_target(base_state_df)
    baseline_features = metrics_df[features].median().values
    
    contributions = {}
    for neighbor in neighbors:
        masked_df = metrics_df.copy()
        idx = masked_df[masked_df['Node_ID'] == neighbor].index
        masked_df.loc[idx, features] = baseline_features
        f_Si = get_score_for_target(masked_df)
        contributions[neighbor] = f_Si_Gi - f_Si
        
    influential_nodes = {node: score for node, score in contributions.items() if score > 0.0}
    return influential_nodes, f_Si_Gi

def explain_and_visualize(target_node: str):
    target_node = str(target_node)
    print(f"\nInitializing GNN investigation on Node {target_node}...")

    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_CSV = BASE_DIR / "data" / "raw_network_transactions.csv"
    FEATURES_CSV = BASE_DIR / "results" / "master_node_features.csv"
    RESULTS_CSV = BASE_DIR / "results" / "anomaly_ensemble_results.csv" # <-- Added this
    MODEL_PATH = BASE_DIR / "models" / "anomaly_ae_model.pth"
    SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"
    PLOT_DIR = BASE_DIR / "results" / "plots"

    df_raw = load_and_clean_data(str(INPUT_CSV))
    G = build_transfer_network(df_raw)
    metrics_df = pd.read_csv(FEATURES_CSV, dtype={'Node_ID': str}, low_memory=False)
    df_results = pd.read_csv(RESULTS_CSV, dtype={'Node_ID': str}, low_memory=False) # <-- Added this

    features = [col for col in metrics_df.columns if col != 'Node_ID']

    node_to_idx = {node_id: idx for idx, node_id in enumerate(metrics_df['Node_ID'])}
    source_indices = [node_to_idx[str(u)] for u, v in G.edges() if str(u) in node_to_idx and str(v) in node_to_idx]
    target_indices = [node_to_idx[str(v)] for u, v in G.edges() if str(u) in node_to_idx and str(v) in node_to_idx]
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)

    scaler = joblib.load(SCALER_PATH)
    model = AnomalyGAE(input_dim=len(features))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() 

    influential_nodes, base_score = calculate_marginal_contribution(
        model, scaler, metrics_df, features, G, target_node, edge_index
    )
    
    guilty_nodes = list(influential_nodes.keys()) + [target_node]
    crime_scene = G.subgraph(guilty_nodes)
    
    # <-- Corrected function call to pass the DataFrames
    draw_shapley_evidence_map(target_node, crime_scene, influential_nodes, df_results, metrics_df, PLOT_DIR)

def print_node_profile(node_id: str, base_dir: Path):
    results_csv = base_dir / "results" / "anomaly_ensemble_results.csv"
    features_csv = base_dir / "results" / "master_node_features.csv"
    
    df_results = pd.read_csv(results_csv, dtype={'Node_ID': str}, low_memory=False)
    df_features = pd.read_csv(features_csv, dtype={'Node_ID': str}, low_memory=False)
    
    if node_id not in df_results['Node_ID'].values:
        print(f"\nError: Node ID '{node_id}' not found in results.")
        return False
        
    node_result = df_results[df_results['Node_ID'] == node_id].iloc[0]
    
    print("\n" + "="*70)
    print(f"  DEEP DIVE: NODE {node_id}")
    print("="*70)
    print(f"OVERALL RANK: {int(node_result['Overall_Rank'])} | Borda Score: {node_result['Borda_Score']:.2f}")
    
    print("\n--- Model Rankings ---")
    print(f" Isolation Forest : Rank {int(node_result['Rank_IF'])}")
    print(f" Local Outlier    : Rank {int(node_result['Rank_LOF'])}")
    print(f" K-Means          : Rank {int(node_result['Rank_KMeans'])}")
    print(f" GNN Autoencoder  : Rank {int(node_result['Rank_AE'])}")
        
    print("\n--- Percentile Analysis ---")
    node_idx = df_features[df_features['Node_ID'] == node_id].index[0]
    node_features = df_features.iloc[node_idx]
    
    metrics = ['in_degree', 'out_degree', 'betweenesscentrality', 'pageranks']
    for m in metrics:
        if m in node_features.index:
            pct = (df_features[m] <= node_features[m]).mean() * 100
            val = node_features[m]
            print(f"{m.replace('_', ' ').title().ljust(25)}: {str(val)[:10].ljust(12)} | [Top {100-pct:.2f}%]")
    
    return True

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    RESULTS_DIR = BASE_DIR / "results"
    PLOT_DIR = RESULTS_DIR / "plots"
    
    print("\n=======================================================")
    print("      SUBGRAPHX EXPLAINABILITY MODULE (PHASE 3)      ")
    print("=======================================================")
    print("What would you like to investigate?")
    print("  [1] The Top 3 Mathematical Anomalies (AI Consensus)")
    print("  [2] The 3 Specific Business Suspect Nodes")
    print("  [3] A Specific Custom Node ID (Deep Dive)")
    print("=======================================================")
    
    choice = input("Enter 1, 2, or 3: ").strip()

    biz_nodes = ["513902872", "509903851", "516096826"]
    df_res = pd.read_csv(RESULTS_DIR / "anomaly_ensemble_results.csv", dtype={'Node_ID': str})
    
    # Always generate/update the ranking radar chart
    draw_business_suspect_comparison(biz_nodes, df_res, PLOT_DIR)
    
    nodes_to_explain = []
    if choice == '1':
        nodes_to_explain = df_res['Node_ID'].head(3).tolist()
    elif choice == '2':
        nodes_to_explain = biz_nodes
    elif choice == '3':
        custom_id = input("\nEnter exact Node ID: ").strip()
        if print_node_profile(custom_id, BASE_DIR):
            nodes_to_explain = [custom_id]

    for node in nodes_to_explain:
        explain_and_visualize(node)