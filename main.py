import sys
import time
import logging
import pandas as pd
from pathlib import Path

# Import Phase 2 and Phase 3 modules
from phase_2_anomaly_detection.anomaly_engine import run_temporal_anomaly_detection
from phase_3_explainability.explainer import investigate_targets

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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    master_features_path = RESULTS_DIR / "master_node_features.csv"
    anomaly_output_path = RESULTS_DIR / "temporal_anomaly_results.csv"
    log_filepath = BASE_DIR / "logs" / "pipeline_execution.log"
    log_filepath.parent.mkdir(exist_ok=True)
    
    logger = setup_pipeline_logger(log_filepath)
    logger.info("Initializing SubgraphX Temporal Pipeline...")

    # --- 1. AUTO-LOAD PRE-CALCULATED FEATURES ---
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

    # --- 2. PHASE 2: TEMPORAL ANOMALY DETECTION ---
    logger.info("Running Phase 2: Conditional Motif GCN & LSTM-VAE...")
    KNOWN_TARGETS = ["513902872", "509903851", "516096826"]
    
    try:
        run_temporal_anomaly_detection(
            data_path=master_features_path, 
            output_path=anomaly_output_path, 
            validation_targets=KNOWN_TARGETS
        )
    except Exception as e:
        logger.error(f"Phase 2 failed: {e}")
        sys.exit(1)

    # --- 3. PHASE 3: EXPLAINABILITY ---
    logger.info("Running Phase 3: Explainability and Target Profiling...")
    try:
        investigate_targets(
            results_csv=anomaly_output_path, 
            targets=KNOWN_TARGETS
        )
    except Exception as e:
        logger.error(f"Phase 3 failed: {e}")

    logger.info(f"Pipeline finished successfully in {time.perf_counter() - pipeline_start_time:.2f}s")

if __name__ == "__main__":
    main()