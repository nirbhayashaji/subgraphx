import time
import logging
import pandas as pd
import networkx as nx
from pathlib import Path

def create_transfer_features(G: nx.DiGraph, df: pd.DataFrame, output_filepath: Path | str, log_filepath: Path | str = "results/pipeline_execution.log"):
    """
    Calculates domain-specific waste network metrics for each node and exports them to a CSV.
    """
    # --- LOGGING SETUP ---
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

    COL_SENDER = 'produtor_nif'             
    COL_RECEIVER = 'destinatario_nif'       
    COL_QUANTITY = 'quantidade_recebida'    
    COL_DANGEROUS = 'ler_recebido_perigosidade' 
    COL_SENDER_TYPE = 'prod_nif_class'      
    
    VAL_DANGEROUS = 'S' 
    VAL_INDIVIDUAL = 'Particular' 

    logger.info("Extracting domain-specific transfer features...")
    
    # --- PRE-CALCULATE AGGREGATIONS FOR SPEED ---
    t_agg_start = time.perf_counter()
    
    w_out_series = df.groupby(COL_SENDER)[COL_QUANTITY].sum()
    w_in_series = df.groupby(COL_RECEIVER)[COL_QUANTITY].sum()
    
    df_dangerous = df[df[COL_DANGEROUS] == VAL_DANGEROUS]
    q_out_series = df_dangerous.groupby(COL_SENDER)[COL_QUANTITY].sum()
    q_in_series = df_dangerous.groupby(COL_RECEIVER)[COL_QUANTITY].sum()
    
    total_suppliers_series = df.groupby(COL_RECEIVER)[COL_SENDER].nunique()
    df_individuals = df[df[COL_SENDER_TYPE] == VAL_INDIVIDUAL]
    indiv_suppliers_series = df_individuals.groupby(COL_RECEIVER)[COL_SENDER].nunique()
    
    logger.info(f"[Timing] Pandas Aggregations completed in {time.perf_counter() - t_agg_start:.4f}s")

    # --- CALCULATE METRICS COLUMN BY COLUMN ---
    # Initialize DataFrame with nodes
    df_waste = pd.DataFrame({'Node_NIF': list(G.nodes())})
    
    # 1. Degree Imbalance Ratio
    t0 = time.perf_counter()
    def calc_deg_imb(node):
        d_in = G.in_degree(node) if isinstance(G, nx.DiGraph) else G.degree(node)
        d_out = G.out_degree(node) if isinstance(G, nx.DiGraph) else G.degree(node)
        return (d_in - d_out) / (d_in + d_out) if (d_in + d_out) > 0 else 0
    df_waste['Degree_Imbalance'] = df_waste['Node_NIF'].apply(calc_deg_imb)
    logger.info(f"[Timing] Feature 'Degree_Imbalance' calculated in {time.perf_counter() - t0:.4f}s")
    
    # 2. Weighted Degree Imbalance Ratio
    t0 = time.perf_counter()
    def calc_wtd_imb(node):
        w_in = w_in_series.get(node, 0)
        w_out = w_out_series.get(node, 0)
        return (w_in - w_out) / (w_in + w_out) if (w_in + w_out) > 0 else 0
    df_waste['Weighted_Degree_Imbalance'] = df_waste['Node_NIF'].apply(calc_wtd_imb)
    logger.info(f"[Timing] Feature 'Weighted_Degree_Imbalance' calculated in {time.perf_counter() - t0:.4f}s")
    
    # 3. Dangerous Quantity Difference Ratio
    t0 = time.perf_counter()
    def calc_dang_imb(node):
        q_in = q_in_series.get(node, 0)
        q_out = q_out_series.get(node, 0)
        return (q_in - q_out) / (q_in + q_out) if (q_in + q_out) > 0 else 0
    df_waste['Dangerous_Qty_Imbalance'] = df_waste['Node_NIF'].apply(calc_dang_imb)
    logger.info(f"[Timing] Feature 'Dangerous_Qty_Imbalance' calculated in {time.perf_counter() - t0:.4f}s")
    
    # 4. Proportion of Individual Suppliers
    t0 = time.perf_counter()
    def calc_indiv_prop(node):
        tot_sup = total_suppliers_series.get(node, 0)
        ind_sup = indiv_suppliers_series.get(node, 0)
        return ind_sup / tot_sup if tot_sup > 0 else 0
    df_waste['Individual_Supplier_Prop'] = df_waste['Node_NIF'].apply(calc_indiv_prop)
    logger.info(f"[Timing] Feature 'Individual_Supplier_Prop' calculated in {time.perf_counter() - t0:.4f}s")
    
    # --- FILE EXPORT ---
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_waste.to_csv(output_path, index=False)
    
    generated_features = df_waste.columns.tolist()
    logger.info(f"Transfer domain features calculated ({len(generated_features)-1} total): {', '.join(generated_features)}")
    logger.info(f"Transfer analysis saved successfully to: {output_path}")
    
    return df_waste