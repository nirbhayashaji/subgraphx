import torch
import pandas as pd
import networkx as nx

def calculate_marginal_contribution(model, scaler, metrics_df, features, G, target_node):
    """
    Applies Shapley value logic: m(Si, Gi) = f(Si U Gi) - f(Si)
    Finds which neighbors contribute most to the target's anomaly score.
    """
    model.eval()
    
    # 1. Get the local neighborhood (Ego Graph)
    ego_graph = nx.ego_graph(G, target_node, radius=1)
    neighbors = list(ego_graph.nodes())
    neighbors.remove(target_node)
    
    # Helper to calculate score for the target node given a specific feature state
    def get_score_for_target(df_state):
        data = df_state[features].fillna(0).values
        X = torch.FloatTensor(scaler.transform(data))
        with torch.no_grad():
            reconstructed = model(X)
            errors = torch.mean((X - reconstructed)**2, dim=1)
        # Find index of target_node
        target_idx = df_state.index.get_loc(df_state[df_state['node'] == target_node].index[0])
        return errors[target_idx].item()

    contributions = {}
    
    # 2. Base Score: f(Si U Gi) -> Full neighborhood present
    base_state_df = metrics_df.copy()
    f_Si_Gi = get_score_for_target(base_state_df)
    
    # 3. Calculate Marginal Contribution for each neighbor
    # f(Si) -> What happens to the target's score if we "mask" a neighbor?
    for neighbor in neighbors:
        masked_df = metrics_df.copy()
        
        # Masking: Setting neighbor's features to 0 (removing them from the coalition)
        idx = masked_df[masked_df['node'] == neighbor].index
        masked_df.loc[idx, features] = 0 
        
        f_Si = get_score_for_target(masked_df)
        
        # Marginal Contribution: f(Si U Gi) - f(Si)
        # If removing the neighbor lowers the anomaly score significantly, they are highly influential
        m = f_Si_Gi - f_Si
        contributions[neighbor] = m
        
    # Sort neighbors by their contribution to the anomaly
    sorted_contributions = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
    
    # Return the nodes that contribute positively to the anomaly score
    influential_nodes = [node for node, score in sorted_contributions if score > 0.0]
    
    return influential_nodes, sorted_contributions