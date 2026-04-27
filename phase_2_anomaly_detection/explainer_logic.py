import sys
import torch
import joblib
import pandas as pd
import networkx as nx
from pathlib import Path

# Add the parent directory to the path to import Phase 1 scripts
sys.path.append(str(Path(__file__).resolve().parent.parent))
from processing import load_and_clean_data
from network_logic import build_transfer_network

# Import the GNN Architecture and our new Visualization Logic
from phase_2_anomaly_detection.model_ae import AnomalyGAE 
from phase_2_anomaly_detection.visualization_logic import draw_shapley_evidence_map

def calculate_marginal_contribution(model, scaler, metrics_df, features, G, target_node, edge_index):
    """
    Applies Shapley value logic to find which neighbors contribute most to the target's anomaly score.
    """
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
            
        target_idx = df_state.index.get_loc(df_state[df_state['Node_ID'] == target_node].index[0])
        return errors[target_idx].item()

    contributions = {}
    
    base_state_df = metrics_df.copy()
    f_Si_Gi = get_score_for_target(base_state_df)
    baseline_features = metrics_df[features].median().values
    
    for neighbor in neighbors:
        masked_df = metrics_df.copy()
        idx = masked_df[masked_df['Node_ID'] == neighbor].index
        masked_df.loc[idx, features] = baseline_features
        
        f_Si = get_score_for_target(masked_df)
        m = f_Si_Gi - f_Si
        contributions[neighbor] = m
        
    influential_nodes = {node: score for node, score in contributions.items() if score > 0.0}
    return influential_nodes, f_Si_Gi

def explain_and_visualize(target_node: str):
    target_node = str(target_node)
    print(f"\nInitializing GNN Shapley investigation on Node {target_node}...")

    # Setup paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_CSV = BASE_DIR / "data" / "raw_network_transactions.csv"
    FEATURES_CSV = BASE_DIR / "results" / "master_node_features.csv"
    MODEL_PATH = BASE_DIR / "models" / "anomaly_ae_model.pth"
    SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"
    PLOT_DIR = BASE_DIR / "results" / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading network, features, and models...")
    df_raw = load_and_clean_data(str(INPUT_CSV))
    G = build_transfer_network(df_raw)
    metrics_df = pd.read_csv(FEATURES_CSV, dtype={'Node_ID': str}, low_memory=False)
    
    if target_node not in G.nodes() or target_node not in metrics_df['Node_ID'].values:
        print(f"Error: Node {target_node} not found in the dataset.")
        return

    features = [col for col in metrics_df.columns if col != 'Node_ID']

    print("Constructing edge index for GNN message passing...")
    node_to_idx = {node_id: idx for idx, node_id in enumerate(metrics_df['Node_ID'])}
    source_indices = []
    target_indices = []
    for u, v in G.edges():
        if str(u) in node_to_idx and str(v) in node_to_idx:
            source_indices.append(node_to_idx[str(u)])
            target_indices.append(node_to_idx[str(v)])
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)

    scaler = joblib.load(SCALER_PATH)

    model = AnomalyGAE(input_dim=len(features))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() 

    print("Calculating marginal contributions (Shapley Values) via Message Passing...")
    influential_nodes, base_score = calculate_marginal_contribution(
        model, scaler, metrics_df, features, G, target_node, edge_index
    )
    
    print(f"Target Base Score: {base_score:.4f}")
    print(f"Found {len(influential_nodes)} mathematically responsible trading partners.")
    
    # --- HAND OFF TO THE NEW VISUALIZATION SCRIPT ---
    guilty_nodes = list(influential_nodes.keys()) + [target_node]
    crime_scene = G.subgraph(guilty_nodes)
    draw_shapley_evidence_map(target_node, crime_scene, influential_nodes, PLOT_DIR)

def print_node_profile(node_id: str, base_dir: Path):
    """Prints a deep-dive contextual profile for a custom node."""
    results_csv = base_dir / "results" / "anomaly_ensemble_results.csv"
    features_csv = base_dir / "results" / "master_node_features.csv"
    
    try:
        df_results = pd.read_csv(results_csv, dtype={'Node_ID': str}, low_memory=False)
        df_features = pd.read_csv(features_csv, dtype={'Node_ID': str}, low_memory=False)
    except FileNotFoundError:
        print("Error: Required CSV files not found. Run the anomaly engine first.")
        return False
        
    if node_id not in df_results['Node_ID'].values:
        print(f"Error: Node {node_id} not found in the network.")
        return False
        
    node_result = df_results[df_results['Node_ID'] == node_id].iloc[0]
    
    biz_nodes = ["513902872", "509903851", "516096826"]
    biz_results = df_results[df_results['Node_ID'].isin(biz_nodes)]
    
    print("\n" + "="*60)
    print(f" 📊 DEEP DIVE PROFILE: NODE {node_id}")
    print("="*60)
    print(f"OVERALL RANK: {int(node_result['Overall_Rank'])} (out of {len(df_results)} nodes)")
    print(f"Borda Score:  {node_result['Borda_Score']}")
    
    print("\n--- Contextual Ranking Comparison ---")
    for _, biz in biz_results.iterrows():
        print(f"🏢 Business Suspect Rank (Node {biz['Node_ID']}): {int(biz['Overall_Rank'])}")
        
    print("\n--- Key Network Metrics ---")
    node_features = df_features[df_features['Node_ID'] == node_id].iloc[0]
    
    metrics_to_show = [
        'in_degree', 'out_degree', 'betweenesscentrality', 'pageranks',
        'degree_imbalance_ratio', 'weighted_degree_imbalance_ratio', 
        'flagged_quantity_difference_ratio', 'terminal_quantity_ratio'
    ]
    
    for metric in metrics_to_show:
        if metric in node_features.index:
            display_name = metric.replace('_', ' ').title()
            val = node_features[metric]
            print(f"{display_name.ljust(35)}: {val:.4f}" if isinstance(val, float) else f"{display_name.ljust(35)}: {val}")
    print("="*60 + "\n")
    return True
    """Prints a deep-dive contextual profile for a custom node."""
    results_csv = base_dir / "results" / "anomaly_ensemble_results.csv"
    features_csv = base_dir / "results" / "master_node_features.csv"
    
    try:
        df_results = pd.read_csv(results_csv, dtype={'Node_ID': str}, low_memory=False)
        df_features = pd.read_csv(features_csv, dtype={'Node_ID': str}, low_memory=False)
    except FileNotFoundError:
        print("Error: Required CSV files not found. Run the anomaly engine first.")
        return False
        
    if node_id not in df_results['Node_ID'].values:
        print(f"Error: Node {node_id} not found in the network.")
        return False
        
    node_result = df_results[df_results['Node_ID'] == node_id].iloc[0]
    top1_result = df_results.iloc[0] 
    
    biz_nodes = ["513902872", "509903851", "516096826"]
    biz_results = df_results[df_results['Node_ID'].isin(biz_nodes)]
    
    print("\n" + "="*60)
    print(f"  DEEP DIVE PROFILE: NODE {node_id}")
    print("="*60)
    print(f"OVERALL RANK: {int(node_result['Overall_Rank'])} (out of {len(df_results)} nodes)")
    print(f"Borda Score:  {node_result['Borda_Score']}")
    
    print("\n--- Contextual Ranking Comparison ---")
    print(f" #1 AI Anomaly Rank:  1 (Node {top1_result['Node_ID']})")
    for _, biz in biz_results.iterrows():
        print(f" Business Suspect Rank (Node {biz['Node_ID']}): {int(biz['Overall_Rank'])}")
        
    print("\n--- Key Network Metrics ---")
    node_features = df_features[df_features['Node_ID'] == node_id].iloc[0]
    
    metrics_to_show = [
        'in_degree', 'out_degree', 'betweenesscentrality', 'pageranks',
        'degree_imbalance_ratio', 'weighted_degree_imbalance_ratio', 
        'flagged_quantity_difference_ratio', 'terminal_quantity_ratio'
    ]
    
    for metric in metrics_to_show:
        if metric in node_features.index:
            display_name = metric.replace('_', ' ').title()
            val = node_features[metric]
            print(f"{display_name.ljust(35)}: {val:.4f}" if isinstance(val, float) else f"{display_name.ljust(35)}: {val}")
    print("="*60 + "\n")
    return True

if __name__ == "__main__":
    print("\n=======================================================")
    print("      SUBGRAPHX EXPLAINABILITY MODULE (PHASE 3)      ")
    print("=======================================================")
    print("What would you like to investigate?")
    print("  [1] The Top 3 Mathematical Anomalies (AI Consensus)")
    print("  [2] The 3 Specific Business Suspect Nodes")
    print("  [3] A Specific Custom Node ID (Deep Dive)")
    print("=======================================================")
    
    choice = input("Enter 1, 2, or 3: ").strip()
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    nodes_to_explain = []
    
    if choice == '1':
        results_csv = BASE_DIR / "results" / "anomaly_ensemble_results.csv"
        try:
            df_results = pd.read_csv(results_csv)
            nodes_to_explain = df_results['Node_ID'].head(3).astype(str).tolist()
            print(f"\nPulling Top 3 anomalies from ranking: {nodes_to_explain}")
        except FileNotFoundError:
            print("Error: anomaly_ensemble_results.csv not found. Run main.py first.")
            sys.exit()
            
    elif choice == '2':
        nodes_to_explain = ["513902872", "509903851", "516096826"]
        print(f"\nQueueing specific business suspects: {nodes_to_explain}")
        
    elif choice == '3':
        custom_node = input("\nEnter the exact Node ID to investigate: ").strip()
        success = print_node_profile(custom_node, BASE_DIR)
        if success:
            nodes_to_explain = [custom_node]
        else:
            sys.exit()
            
    else:
        print("Invalid choice. Exiting.")
        sys.exit()

    for node in nodes_to_explain:
        print(f"\n-------------------------------------------------------")
        explain_and_visualize(node)