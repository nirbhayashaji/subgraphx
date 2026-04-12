import pandas as pd
import numpy as np

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads e-GAR CSV and pre-calculates features for the Autoencoder.
    Pre-calculating flags here prevents 'astype' warnings in aggregation.
    """
    df = pd.read_csv(filepath, low_memory=False)
    
    # 1. Casting and Cleaning
    df['produtor_nif'] = df['produtor_nif'].astype(str)
    df['destinatario_nif'] = df['destinatario_nif'].astype(str)
    df['quantidade_recebida'] = pd.to_numeric(df['quantidade_recebida'], errors='coerce').fillna(0)
    
    # 2. Feature Engineering (Boolean Flags)
    df['is_dangerous'] = df['ler_recebido_perigosidade'].astype(str).str.contains('1', na=False).astype(int)
    df['is_disposable'] = df['tipo_op_recebido_codigo'].astype(str).str.contains('D', na=False).astype(int)
    df['is_individual'] = (df['prod_nif_class'] == 'pessoa_singular').astype(int)
    
    # 3. Quantity Helpers
    df['disposable_qty'] = df['quantidade_recebida'] * df['is_disposable']
    df['dangerous_qty'] = df['quantidade_recebida'] * df['is_dangerous']
    
    return df