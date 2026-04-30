import sys
import time
import logging
import pandas as pd
import subprocess
import os
from pathlib import Path
from processing import load_and_clean_data
from network_logic import build_transfer_network

# Import our custom modules
from phase_2_anomaly_detection.anomaly_engine import run_anomaly_ensemble

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

def open_image(path):
    """Automatically opens the image file based on the OS."""
    try:
        if sys.platform.startswith('darwin'):     # macOS
            subprocess.call(('open', path))
        elif os.name == 'nt':                      # Windows
            os.startfile(path)
        elif os.name == 'posix':                   # Linux
            subprocess.call(('xdg-open', path))
    except Exception as e:
        print(f"Could not automatically open image: {e}")

def main():
    pipeline_start_time = time.perf_counter()
    
    # --- 1. SETUP PATHS ---
    BASE_DIR = Path(__file__).resolve().parent 
    GEPHI_FILE = BASE_DIR / "data" / "precalc_structural_metrics.csv"
    TRANSFER_FILE = BASE_DIR / "data" / "precalc_domain_metrics.csv"
    
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    master_features_path = RESULTS_DIR / "master_node_features.csv"
    anomaly_output_path = RESULTS_DIR / "anomaly_ensemble_results.csv"
    radar_plot_path = RESULTS_DIR / "plots" / "top_5_nodes_radar.png"

    log_filepath = BASE_DIR / "logs" / "pipeline_execution.log"
    log_filepath.parent.mkdir(exist_ok=True)
    logger = setup_pipeline_logger(log_filepath)

    # --- 2. FAST-TRACK LOGIC ---
    use_fast_track = 'no'
    if GEPHI_FILE.exists() and TRANSFER_FILE.exists():
        print("\n" + "="*50)
        print(" PRE-CALCULATED METRICS DETECTED (GEPHI/DOMAIN)")
        print("="*50)
        use_fast_track = input("Fast-track the pipeline? (yes/no): ").strip().lower()
        
        if use_fast_track == 'yes':
            logger.info("Fast-Track Initiated. Merging datasets...")
            df_gephi = pd.read_csv(GEPHI_FILE, low_memory=False)
            df_transfer = pd.read_csv(TRANSFER_FILE, low_memory=False)
            
            # Align Node_IDs
            df_gephi.rename(columns={df_gephi.columns[0]: 'Node_ID'}, inplace=True)
            df_transfer.rename(columns={df_transfer.columns[0]: 'Node_ID'}, inplace=True)
            
            df_master = pd.merge(df_gephi, df_transfer, on='Node_ID', how='outer').fillna(0)
            df_master.to_csv(master_features_path, index=False)
            logger.info(f"Master features built: {len(df_master)} nodes.")

    # --- 3. FALLBACK TO STANDARD (If Fast-track is no) ---
    if use_fast_track != 'yes':
        logger.info("Running standard NetworkX pipeline (This may take a while)...")
        # [Placeholder for your existing create_graph_features logic]
        pass

    # --- 4. PHASE 2: ANOMALY DISCOVERY ---
    logger.info("Running AI Anomaly Ensemble (IF, LOF, KMeans, AE)...")
    try:
        run_anomaly_ensemble(data_path=master_features_path, output_path=anomaly_output_path)
        
        logger.info(f"Pipeline finished successfully in {time.perf_counter() - pipeline_start_time:.2f}s")
        
        # --- 5. THE "WOW" MOMENT: AUTO-OPEN RESULTS ---
        if radar_plot_path.exists():
            print("\n Discovery Complete! Opening Top 5 Anomaly Radar Chart...")
            open_image(radar_plot_path)
            
        print("\n NEXT STEP: Investigate specific nodes with:")
        print("   python phase_2_anomaly_detection/explainer_logic.py")

    except Exception as e:
        logger.error(f"Pipeline failed at Anomaly Stage: {e}")

if __name__ == "__main__":
    main()