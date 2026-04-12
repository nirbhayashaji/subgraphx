import sys
import time
import logging
import pandas as pd
from pathlib import Path
from processing import load_and_clean_data
from network_logic import build_waste_network

from graph_feature_creation import create_graph_features
from transfer_feature_creation import create_transfer_features

def setup_pipeline_logger(log_filepath: Path | str):
    """Sets up a logger that writes to both console and file."""
    logger = logging.getLogger("Pipeline")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

def main():
    pipeline_start_time = time.perf_counter()
    
    # --- 1. SETUP ---
    BASE_DIR = Path(__file__).resolve().parent 
    INPUT_FILE = BASE_DIR / "data" / "view_egar_guias_202308221708_cleanedReducedNIFClassification_2025-02-08_15-34-52.csv"
    
    # PRE-CALCULATED FEATURE FILES
    GEPHI_FILE = BASE_DIR / "data" / "egar_unique_node_metrics_gephi_shazia.csv"
    TRANSFER_FILE = BASE_DIR / "data" / "unique_transfers_metrices_2025-02-11_21-33-51.csv"
    
    # 🚨 GITHUB FIX: Ensure all dynamic directories exist! 🚨
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR = RESULTS_DIR / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    LOGS_DIR = BASE_DIR / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_filepath = LOGS_DIR / "pipeline_execution.log"
    logger = setup_pipeline_logger(log_filepath)
    logger.info("#########################################################################################")
    sme_nodes = ["513902872", "509903851", "516096826"]

    analysis_output_path = RESULTS_DIR / "data_analysis.csv"
    waste_output_path = RESULTS_DIR / "waste_analysis.csv"
    master_features_path = RESULTS_DIR / "waste_transfer_graph_features_node_features.csv"
    global_summary_path = RESULTS_DIR / "global_graph_features_summary.csv"

    # --- 2. SUPER FAST-TRACK CHECK (PRE-CALCULATED GEPHI + TRANSFER METRICS) ---
    if GEPHI_FILE.exists() and TRANSFER_FILE.exists():
        logger.info(f"Found pre-calculated Gephi metrics: {GEPHI_FILE.name}")
        logger.info(f"Found pre-calculated Transfer metrics: {TRANSFER_FILE.name}")
        
        use_fast_track = input("\nBoth pre-calculated feature files were found.\nDo you want to fast-track the pipeline and merge these directly without recalculating? (yes/no): ").strip().lower()
        
        if use_fast_track == 'yes':
            logger.info("Super Fast-Track Initiated. Bypassing ALL feature generation...")
            
            logger.info("Loading Raw Data to verify Graph Structure...")
            t_data = time.perf_counter()
            df = load_and_clean_data(str(INPUT_FILE))
            logger.info(f"[Timing] Data Loading took {time.perf_counter() - t_data:.2f}s")
            
            logger.info("Building Directed Graph (Structure Only)...")
            G = build_waste_network(df)
            num_nodes, num_edges = G.number_of_nodes(), G.number_of_edges()
            
            logger.info("Loading cached CSV files for merging...")
            df_gephi = pd.read_csv(GEPHI_FILE, low_memory=False)
            df_transfer = pd.read_csv(TRANSFER_FILE, low_memory=False)
            
            gephi_id_col = df_gephi.columns[0]
            transfer_id_col = df_transfer.columns[0]
            
            df_gephi.rename(columns={gephi_id_col: 'Node_NIF'}, inplace=True)
            df_transfer.rename(columns={transfer_id_col: 'Node_NIF'}, inplace=True)
            
            df_gephi['Node_NIF'] = df_gephi['Node_NIF'].astype(str)
            df_transfer['Node_NIF'] = df_transfer['Node_NIF'].astype(str)
            
            logger.info(f"Joining {len(df_gephi.columns)-1} Gephi features and {len(df_transfer.columns)-1} Transfer features...")
            
            df_master = pd.merge(df_gephi, df_transfer, on='Node_NIF', how='outer')
            df_master.fillna(0, inplace=True) 
            
            df_master.to_csv(master_features_path, index=False)
            logger.info(f"Master features successfully built and saved to: {master_features_path}")
            
            # --- GLOBAL METRICS EXTRACTION ---
            logger.info("\n--- Derived Global Graph Features (Averages) ---")
            global_metrics = []
            
            for col in df_master.columns:
                if col != 'Node_NIF' and pd.api.types.is_numeric_dtype(df_master[col]):
                    avg_val = df_master[col].mean()
                    max_val = df_master[col].max()
                    min_val = df_master[col].min()
                    
                    if avg_val != 0: 
                        logger.info(f"Mean {col[:30].ljust(30)}: {avg_val:.4f}")
                        global_metrics.append({
                            'Feature_Name': col,
                            'Mean': avg_val,
                            'Max': max_val,
                            'Min': min_val
                        })
            
            df_global = pd.DataFrame(global_metrics)
            df_global.to_csv(global_summary_path, index=False)
            logger.info(f"Global graph feature summary saved to: {global_summary_path}")
            
            logger.info("=========================================")
            logger.info("      SUPER FAST-TRACK RUN SUMMARY       ")
            logger.info("=========================================")
            logger.info(f"Total Nodes in Graph : {num_nodes}")
            logger.info(f"Total Edges in Graph : {num_edges}")
            logger.info(f"Total Features Gen   : {len(df_master.columns) - 1}")
            logger.info(f"Total Pipeline Time  : {time.perf_counter() - pipeline_start_time:.2f} seconds")
            logger.info("=========================================")
            
            sys.exit() 

    # --- 3. STANDARD INTERACTIVE PIPELINE ---
    recalc = input("\nDo you want to recalculate all features from scratch via NetworkX? (yes/no): ").strip().lower()

    if recalc != 'yes':
        if not analysis_output_path.exists() or not waste_output_path.exists():
            logger.warning("You selected 'no', but the intermediate CSV files are missing. You MUST run the whole pipeline.")
            recalc = 'yes' 
        else:
            logger.info("User opted to skip feature recalculation. Relying on intermediate cached CSV files.")

    num_nodes, num_edges = 0, 0

    if recalc == 'yes':
        logger.info("Loading & Cleaning Data...")
        t_data = time.perf_counter()
        df = load_and_clean_data(str(INPUT_FILE))
        logger.info(f"[Timing] Data Loading took {time.perf_counter() - t_data:.2f}s")

        logger.info("Building Graph...")
        
        row_limit = input("Enter the number of data rows to process (or type 'all' for the entire dataset): ").strip().lower()
        
        if row_limit == 'all':
            df_graph = df
            logger.info("Proceeding with the entire dataset.")
        elif row_limit.isdigit():
            n_rows = int(row_limit)
            df_graph = df.head(n_rows)
            logger.info(f"Proceeding with the first {n_rows} rows of the dataset.")
        else:
            logger.warning(f"Invalid input '{row_limit}'. Defaulting to the entire dataset.")
            df_graph = df

        G = build_waste_network(df_graph)
        num_nodes, num_edges = G.number_of_nodes(), G.number_of_edges()
        
        print(f"\n--- GRAPH BUILT ---")
        print(f"Total Nodes: {num_nodes}")
        print(f"Total Edges: {num_edges}")
        confirm = input("Are you sure you want to go ahead? NetworkX calculations on large graphs may take hours/days. (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            logger.info("Pipeline aborted by user due to graph size.")
            sys.exit()

        logger.info("Performing Graph Feature Creation...")
        t_func_graph = time.perf_counter()
        create_graph_features(
            G=G, 
            sme_nodes=sme_nodes, 
            output_filepath=analysis_output_path,
            log_filepath=log_filepath 
        )
        logger.info(f"[Timing] Entire create_graph_features() function took {time.perf_counter() - t_func_graph:.2f}s")
            
        logger.info("Performing Transfer Feature Creation...")
        t_func_trans = time.perf_counter()
        create_transfer_features(
            G=G,
            df=df_graph, 
            output_filepath=waste_output_path,
            log_filepath=log_filepath
        )
        logger.info(f"[Timing] Entire create_transfer_features() function took {time.perf_counter() - t_func_trans:.2f}s")

    # --- 4. MERGE MASTER FEATURES (STANDARD RUN) ---
    logger.info("Merging structural and transfer features into Master Dataset...")
    try:
        df_net = pd.read_csv(analysis_output_path, low_memory=False)
        df_wst = pd.read_csv(waste_output_path, low_memory=False)
        
        df_master = pd.merge(df_net, df_wst, on='Node_NIF', how='outer')
        df_master.fillna(0, inplace=True)
        
        df_master.to_csv(master_features_path, index=False)
        logger.info(f"Master features successfully built and saved to: {master_features_path}")
        
        if num_nodes == 0:
            num_nodes = len(df_master)
            num_edges = "Skipped/Unknown"
            
        # --- GLOBAL METRICS EXTRACTION (STANDARD RUN) ---
        global_metrics = []
        for col in df_master.columns:
            if col != 'Node_NIF' and pd.api.types.is_numeric_dtype(df_master[col]):
                avg_val = df_master[col].mean()
                if avg_val != 0: 
                    global_metrics.append({
                        'Feature_Name': col,
                        'Mean': avg_val,
                        'Max': df_master[col].max(),
                        'Min': df_master[col].min()
                    })
        df_global = pd.DataFrame(global_metrics)
        df_global.to_csv(global_summary_path, index=False)
        logger.info(f"Global graph feature summary saved to: {global_summary_path}")

    except Exception as e:
        logger.error(f"Failed to merge master features: {e}")
    
    # --- 5. PIPELINE SUMMARY ---
    pipeline_end_time = time.perf_counter()
    logger.info("=========================================")
    logger.info("         PIPELINE RUN SUMMARY            ")
    logger.info("=========================================")
    logger.info(f"Total Nodes in Data  : {num_nodes}")
    logger.info(f"Total Edges in Graph : {num_edges}")
    logger.info(f"Total Features Gen   : {len(df_master.columns) - 1 if 'df_master' in locals() else 'Unknown'}")
    logger.info(f"Total Pipeline Time  : {pipeline_end_time - pipeline_start_time:.2f} seconds")
    logger.info("=========================================")

if __name__ == "__main__":
    main()