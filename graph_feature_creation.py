import time
import logging
import pandas as pd
import networkx as nx
from pathlib import Path

def create_graph_features(G: nx.Graph | nx.DiGraph, target_nodes, output_filepath: Path | str, log_filepath: Path | str = "results/analysis.log"):
    """
    Calculates node-level network metrics from the graph and exports them to a CSV.
    Logs graph information to both the console and a specified log file.
    """
    log_path = Path(log_filepath)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("Pipeline")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("Performing detailed network analysis...")
    logger.info("Graph Information:")
    logger.info(G)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    logger.info(f"- Total Nodes: {num_nodes}")
    logger.info(f"- Total Edges: {num_edges}")
    logger.info(f"- Is Directed: {G.is_directed()}")
    logger.info(f"- Is Multigraph: {G.is_multigraph()}")
    
    logger.info(f"- Network Density: {nx.density(G):.6f}")
    
    try:
        logger.info(f"- Transitivity: {nx.transitivity(G):.6f}")
    except Exception as e:
        logger.warning(f"- Could not calculate Transitivity: {e}")

    try:
        assortativity = nx.degree_assortativity_coefficient(G)
        logger.info(f"- Degree Assortativity: {assortativity:.4f}")
    except Exception as e:
        logger.warning(f"- Could not calculate Assortativity: {e}")

    if isinstance(G, nx.DiGraph):
        try:
            logger.info(f"- Reciprocity: {nx.reciprocity(G):.6f}")
        except Exception as e:
            pass 
            
        try:
            weak_comps = nx.number_weakly_connected_components(G)
            strong_comps = nx.number_strongly_connected_components(G)
            logger.info(f"- Weakly Connected Components: {weak_comps}")
            logger.info(f"- Strongly Connected Components: {strong_comps}")
        except Exception as e:
            logger.warning(f"- Could not calculate components: {e}")
    else:
        try:
            comps = nx.number_connected_components(G)
            logger.info(f"- Connected Components: {comps}")
        except Exception as e:
            logger.warning(f"- Could not calculate components: {e}")

    if num_nodes > 0:
        degrees = [deg for node, deg in G.degree()]
        avg_degree = sum(degrees) / num_nodes
        logger.info(f"- Average Degree: {avg_degree:.2f}")
        logger.info(f"- Maximum Degree: {max(degrees)}")
        
    missing_targets = [node for node in target_nodes if node not in G.nodes()]
    if missing_targets:
        logger.warning(f"- WARNING: These target nodes are missing from the graph: {missing_targets}")
    else:
        logger.info("- All target nodes are present in the graph.")
    
    # --- 1. BASIC DEGREE CALCULATION ---
    t_start_degree = time.perf_counter()
    analysis_data = []
    
    for node in G.nodes():
        node_info = {'Node_ID': node}
        
        if isinstance(G, nx.DiGraph):
            node_info['In_Degree'] = G.in_degree(node)
            node_info['Out_Degree'] = G.out_degree(node)
            node_info['Total_Degree'] = G.degree(node)
        else:
            node_info['Degree'] = G.degree(node)             
        
        analysis_data.append(node_info)
        
    df_analysis = pd.DataFrame(analysis_data)
    logger.info(f"[Timing] Basic Degrees calculated in {time.perf_counter() - t_start_degree:.4f}s")
    
    # --- 2. ADVANCED METRICS CALCULATION ---
    logger.info("Calculating advanced network centralities and structural metrics...")
    sample_size = None if num_nodes < 5000 else 1000
    
    try:
        t0 = time.perf_counter()
        if sample_size:
            logger.info(f"Graph is very large. Using k={sample_size} approximation for Betweenness...")
        
        df_analysis['Betweenness'] = df_analysis['Node_ID'].map(
            nx.betweenness_centrality(G, k=sample_size, seed=42)
        )
        logger.info(f"[Timing] Betweenness calculated in {time.perf_counter() - t0:.4f}s")
        
        t0 = time.perf_counter()
        if num_nodes < 5000:
            df_analysis['Closeness'] = df_analysis['Node_ID'].map(nx.closeness_centrality(G))
            logger.info(f"[Timing] Closeness calculated in {time.perf_counter() - t0:.4f}s")
        else:
            logger.warning(f"Skipping Closeness Centrality: Graph is too large ({num_nodes} nodes).")
            df_analysis['Closeness'] = 0 
        
        t0 = time.perf_counter()
        clustering_dict: dict = nx.clustering(G)  # type: ignore
        df_analysis['Clustering_Coeff'] = df_analysis['Node_ID'].map(clustering_dict)
        logger.info(f"[Timing] Clustering Coeff calculated in {time.perf_counter() - t0:.4f}s")
        
        t0 = time.perf_counter()
        G_un = G.to_undirected() if G.is_directed() else G
        triangles_dict: dict = nx.triangles(G_un)  # type: ignore
        df_analysis['Triangles'] = df_analysis['Node_ID'].map(triangles_dict)
        logger.info(f"[Timing] Triangles calculated in {time.perf_counter() - t0:.4f}s")
        
    except Exception as e:
        logger.error(f"Error calculating standard structural metrics: {e}")

    # --- 3. FILE EXPORT ---
    generated_features = df_analysis.columns.tolist()
    logger.info(f"Node-level graph structural features calculated ({len(generated_features)-1} total): {', '.join(generated_features)}")

    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_analysis.to_csv(output_path, index=False)
    logger.info(f"Network analysis saved successfully to: {output_path}")
    
    return df_analysis