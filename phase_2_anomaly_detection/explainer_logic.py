import sys
import torch
import joblib
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# Add the parent directory to the path to import Phase 1 scripts
sys.path.append(str(Path(__file__).resolve().parent.parent))
from processing import load_and_clean_data
from network_logic import build_transfer_network

# Import the new GNN Architecture
from phase_2_anomaly_detection.model_ae import AnomalyGAE 

def calculate_marginal_contribution(model, scaler, metrics_df, features, G, target_node, edge_index):
    """
    Applies Shapley value logic to find which neighbors contribute most to the target's anomaly score.
    Now utilizes GNN Message Passing!
    """
    model.eval()
    
    # Get the local neighborhood (Catching both INCOMING and OUTGOING partners)
    ego_graph = nx.ego_graph(G, target_node, radius=1, undirected=True)
    neighbors = list(ego_graph.nodes())
    if target_node in neighbors:
        neighbors.remove(target_node)
    
    def get_score_for_target(df_state):
        data = df_state[features].fillna(0)
        X = torch.FloatTensor(scaler.transform(data))
        with torch.no_grad():
            # GNN FORWARD PASS: Now requires both features and network structure (edge_index)
            reconstructed = model(X, edge_index)
            errors = torch.mean((X - reconstructed)**2, dim=1)
            
        target_idx = df_state.index.get_loc(df_state[df_state['Node_ID'] == target_node].index[0])
        return errors[target_idx].item()

    contributions = {}
    
    # Base Score: Full neighborhood present
    base_state_df = metrics_df.copy()
    f_Si_Gi = get_score_for_target(base_state_df)
    
    # Calculate median baseline for masking
    baseline_features = metrics_df[features].median().values
    
    # Calculate marginal contribution for each neighbor
    for neighbor in neighbors:
        masked_df = metrics_df.copy()
        
        # Masking: Set neighbor's features to the median
        idx = masked_df[masked_df['Node_ID'] == neighbor].index
        masked_df.loc[idx, features] = baseline_features
        
        f_Si = get_score_for_target(masked_df)
        
        m = f_Si_Gi - f_Si
        contributions[neighbor] = m
        
    # Filter for nodes that drove the anomaly score up
    influential_nodes = {node: score for node, score in contributions.items() if score > 0.0}
    
    return influential_nodes, f_Si_Gi

def explain_and_visualize(target_node: str):
    target_node = str(target_node)
    print(f"Initializing GNN Shapley investigation on Node {target_node}...")

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

    # --- BUILD THE EDGE INDEX FOR THE GNN ---
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

    # Load PyTorch Geometric Model
    model = AnomalyGAE(input_dim=len(features))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() 

    print("Calculating marginal contributions (Shapley Values) via Message Passing...")
    influential_nodes, base_score = calculate_marginal_contribution(
        model, scaler, metrics_df, features, G, target_node, edge_index
    )
    
    print(f"Target Base Score: {base_score:.4f}")
    print(f"Found {len(influential_nodes)} mathematically responsible trading partners.")

    print("Drawing the evidence map...")
    
    guilty_nodes = list(influential_nodes.keys()) + [target_node]
    crime_scene = G.subgraph(guilty_nodes)

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(crime_scene, k=0.5, seed=42)

    # Target Node styling
    nx.draw_networkx_nodes(crime_scene, pos, nodelist=[target_node], node_color='#ff4d4d', node_size=2000, edgecolors='black')
    
    # Partner styling based on Shapley Value
    if influential_nodes:
        max_shap = max(influential_nodes.values())
        if max_shap > 0:
            partner_sizes = [(influential_nodes[n] / max_shap) * 1000 + 300 for n in crime_scene.nodes() if n != target_node]
            
            cmap = cm.get_cmap('YlOrRd') 
            partner_colors = [cmap(influential_nodes[n] / max_shap) for n in crime_scene.nodes() if n != target_node]
            
            nx.draw_networkx_nodes(crime_scene, pos, nodelist=list(influential_nodes.keys()), 
                                   node_color=partner_colors, node_size=partner_sizes, edgecolors='black')

    nx.draw_networkx_edges(crime_scene, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray', width=2, alpha=0.7)

    labels = {n: f"{n}\n(Shapley: {influential_nodes.get(n, 0):.4f})" if n != target_node else f"TARGET\n{n}" for n in crime_scene.nodes()}
    nx.draw_networkx_labels(crime_scene, pos, labels=labels, font_size=8, font_weight="bold")

    plt.title(f"GNN Explainability: Node {target_node}\n(Marginal Contribution to Anomaly Score)", fontsize=16, fontweight='bold')
    plt.axis('off')

    plot_path = PLOT_DIR / f"subgraphx_shapley_{target_node}.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Investigation complete. Visual evidence saved to: {plot_path}")

if __name__ == "__main__":
    print("\n=======================================================")
    print("      SUBGRAPHX EXPLAINABILITY MODULE (PHASE 3)      ")
    print("=======================================================")
    print("What would you like to investigate?")
    print("  [1] The Top 3 Mathematical Anomalies (AI Consensus)")
    print("  [2] The 3 Specific Business Suspect Nodes")
    print("=======================================================")
    
    choice = input("Enter 1 or 2: ").strip()
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    nodes_to_explain = []
    
    if choice == '1':
        # Automatically pull the Top 3 from the anomaly rankings
        results_csv = BASE_DIR / "results" / "anomaly_ensemble_results.csv"
        try:
            df_results = pd.read_csv(results_csv)
            nodes_to_explain = df_results['Node_ID'].head(3).astype(str).tolist()
            print(f"\nPulling Top 3 anomalies from ranking: {nodes_to_explain}")
        except FileNotFoundError:
            print("Error: anomaly_ensemble_results.csv not found. Run main.py first.")
            sys.exit()
            
    elif choice == '2':
        # Use the hardcoded business suspects
        nodes_to_explain = ["513902872", "509903851", "516096826"]
        print(f"\nQueueing specific business suspects: {nodes_to_explain}")
        
    else:
        print("Invalid choice. Exiting.")
        sys.exit()

    # Loop through the chosen list and generate the GNN Shapley evidence maps!
    for node in nodes_to_explain:
        print(f"\n-------------------------------------------------------")
        explain_and_visualize(node)