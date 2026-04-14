import pandas as pd
import numpy as np

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw transaction CSV and pre-calculates features for the pipeline.
    Maps proprietary/native column headers to generic academic headers.
    """
    df = pd.read_csv(filepath, low_memory=False)
    
    # --- 0. TRANSLATE RAW PROPRIETARY COLUMNS TO GENERIC COLUMNS ---
    column_mapping = {
        'produtor_nif': 'source_node',
        'destinatario_nif': 'target_node',
        'quantidade_recebida': 'transfer_weight',
        'ler_recebido_perigosidade': 'risk_flag',
        'tipo_op_recebido_codigo': 'operation_type',
        'prod_nif_class': 'source_type',
        'produtor_origem': 'source_category'
    }
    # Rename columns safely
    df.rename(columns=lambda x: column_mapping.get(x, x), inplace=True)
    
    # Standardize specific Portuguese text values to generic English for downstream scripts
    if 'source_type' in df.columns:
        df['source_type'] = df['source_type'].replace('pessoa_singular', 'individual_entity')
        df['source_type'] = df['source_type'].replace('Particular', 'individual_entity')
        
    if 'risk_flag' in df.columns:
        df['risk_flag'] = df['risk_flag'].replace('S', '1') # Normalize the hazard flag

    # --- 1. CASTING AND CLEANING ---
    df['source_node'] = df['source_node'].astype(str)
    df['target_node'] = df['target_node'].astype(str)
    df['transfer_weight'] = pd.to_numeric(df['transfer_weight'], errors='coerce').fillna(0)
    
    # --- 2. FEATURE ENGINEERING (BOOLEAN FLAGS) ---
    df['is_flagged'] = df['risk_flag'].astype(str).str.contains('1', na=False).astype(int)
    df['is_terminal'] = df['operation_type'].astype(str).str.contains('D', na=False).astype(int)
    df['is_individual'] = (df['source_type'] == 'individual_entity').astype(int)
    
    # --- 3. QUANTITY HELPERS ---
    df['terminal_qty'] = df['transfer_weight'] * df['is_terminal']
    df['flagged_qty'] = df['transfer_weight'] * df['is_flagged']
    
    return df