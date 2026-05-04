import sys
import time
import logging
import pandas as pd
from pathlib import Path

# --- CORRECTED IMPORTS TO MATCH YOUR ACTUAL FILES ---
from phase_2_anomaly_detection.anomaly_engine import run_anomaly_ensemble
from phase_2_anomaly_detection.explainer_logic import explain_and_visualize, print_node_profile
from phase_2_anomaly_detection.visualization_logic import draw_business_suspect_comparison # <-- Added import

def setup_pipeline_logger(log_filepath: Path | str):
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
    
    BASE_DIR = Path(__file__).resolve().parent 
    GEPHI_FILE = BASE_DIR / "data" / "precalc_structural_metrics.csv"
    TRANSFER_FILE = BASE_DIR / "data" / "precalc_domain_metrics.csv"
    RESULTS_DIR = BASE_DIR / "results"
    PLOT_DIR = RESULTS_DIR / "plots" # <-- Added plot directory
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    master_features_path = RESULTS_DIR / "master_node_features.csv"
    anomaly_output_path = RESULTS_DIR / "anomaly_ensemble_results.csv"
    log_filepath = BASE_DIR / "logs" / "pipeline_execution.log"
    log_filepath.parent.mkdir(exist_ok=True)
    
    logger = setup_pipeline_logger(log_filepath)
    logger.info("Initializing SubgraphX Pipeline...")

    # --- 1. AUTO-MERGE FEATURES ---
    if GEPHI_FILE.exists() and TRANSFER_FILE.exists():
        logger.info("Pre-calculated metrics detected. Merging datasets automatically...")
        df_gephi = pd.read_csv(GEPHI_FILE, low_memory=False)
        df_transfer = pd.read_csv(TRANSFER_FILE, low_memory=False)
        
        df_gephi.rename(columns={df_gephi.columns[0]: 'Node_ID'}, inplace=True)
        df_transfer.rename(columns={df_transfer.columns[0]: 'Node_ID'}, inplace=True)
        
        df_master = pd.merge(df_gephi, df_transfer, on='Node_ID', how='outer').fillna(0)
        df_master.to_csv(master_features_path, index=False)
        logger.info(f"Master features built: {len(df_master)} nodes.")
    else:
        logger.error("Pre-calculated files missing in /data/. Exiting.")
        sys.exit(1)

    # --- 2. RUN ANOMALY DETECTION ---
    logger.info("Running Phase 2: Anomaly Detection Ensemble...")
    KNOWN_TARGETS = ["513902872", "509903851", "516096826"]
    
    try:
        run_anomaly_ensemble(
            data_path=master_features_path, 
            output_path=anomaly_output_path, 
            target_nodes=KNOWN_TARGETS
        )
    except Exception as e:
        logger.error(f"Phase 2 Detection failed: {e}")
        sys.exit(1)

    # --- 3. RUN EXPLAINER & PLOTS ---
    logger.info("Running Phase 3: Explainability and Target Profiling...")
    
    # ---> NEW: GENERATE MASTER CONSENSUS DASHBOARD
    try:
        logger.info("Generating Master Consensus Dashboard...")
        df_results = pd.read_csv(anomaly_output_path, dtype={'Node_ID': str}, low_memory=False)
        draw_business_suspect_comparison(KNOWN_TARGETS, df_results, PLOT_DIR)
    except Exception as e:
        logger.error(f"Failed to generate Master Dashboard: {e}")

    # Generate individual target dashboards
    for target in KNOWN_TARGETS:
        try:
            # Print the text profile to the terminal
            print_node_profile(target, BASE_DIR)
            
            # Generate the Shapley plots and save them
            explain_and_visualize(target)
            
        except Exception as e:
            logger.error(f"Explainer failed on node {target}: {e}")

    logger.info(f"Pipeline finished successfully in {time.perf_counter() - pipeline_start_time:.2f}s")

    # --- 4. INTERACTIVE CUSTOM NODE INVESTIGATION ---
    print("\n" + "="*70)
    print(" PIPELINE COMPLETE. ENTERING INTERACTIVE MODE")
    print("="*70)
    
    while True:
        custom_node = input("\nEnter a Node ID to investigate (or type 'exit' to quit): ").strip()
        
        if custom_node.lower() in ['exit', 'quit', 'q']:
            print("Exiting interactive mode. Goodbye!")
            break
            
        if custom_node:
            try:
                # The print_node_profile function returns True if the node exists
                found = print_node_profile(custom_node, BASE_DIR)
                
                if found:
                    # If it exists, generate the visual dashboard
                    explain_and_visualize(custom_node)
                    print(f"\nDashboard saved to results/plots/subgraphx_dashboard_{custom_node}.png")
            except Exception as e:
                logger.error(f"Failed to investigate custom node {custom_node}: {e}")

if __name__ == "__main__":
    main()