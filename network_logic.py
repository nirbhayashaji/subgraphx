import networkx as nx
import pandas as pd
import numpy as np

def build_waste_network(df: pd.DataFrame) -> nx.DiGraph:
    """Creates a Directed Graph from unique transfers."""
    unique_transfers = df.groupby(['produtor_nif', 'destinatario_nif']).agg(
        total_sum_quantity=('quantidade_recebida', 'sum'),
        dangerous_sum_quantity=('dangerous_qty', 'sum')
    ).reset_index()

    G = nx.from_pandas_edgelist(
        unique_transfers, source='produtor_nif', target='destinatario_nif', 
        edge_attr=True, create_using=nx.DiGraph
    )
    entities = pd.unique(df[['produtor_nif', 'destinatario_nif']].values.ravel('K'))
    G.add_nodes_from(entities)
    return G

def calculate_metrics(G: nx.DiGraph, df: pd.DataFrame) -> pd.DataFrame:
    """Calculates paper metrics including centralities and imbalance ratios."""
    nodes = list(G.nodes())
    metrics_df = pd.DataFrame(index=nodes)
    metrics_df.index.name = 'node'
    
    metrics_df['all_degree'] = pd.Series(dict(G.degree()))
    metrics_df['in_degree'] = pd.Series(dict(G.in_degree()))
    metrics_df['out_degree'] = pd.Series(dict(G.out_degree()))
    
    print("  > Calculating Centralities...")
    metrics_df['pageranks'] = pd.Series(nx.pagerank(G, alpha=0.85))
    metrics_df['betweenesscentrality'] = pd.Series(nx.betweenness_centrality(G))
    try:
        metrics_df['eigencentrality'] = pd.Series(nx.eigenvector_centrality(G, max_iter=1000))
    except:
        metrics_df['eigencentrality'] = 0

    dest_agg = df.groupby('destinatario_nif').agg(
        incoming_total=('quantidade_recebida', 'sum'),
        incoming_disposable=('disposable_qty', 'sum'),
        variance_incoming=('quantidade_recebida', 'var'),
        dangerous_received_quantity=('dangerous_qty', 'sum'),
        total_inc_transfers=('quantidade_recebida', 'count'),
        indiv_count=('is_individual', 'sum')
    )

    prod_agg = df.groupby('produtor_nif').agg(
        outgoing_total=('quantidade_recebida', 'sum'),
        variance_outgoing=('quantidade_recebida', 'var'),
        dangerous_sent_quantity=('dangerous_qty', 'sum'),
        not_only_obras_rcd=('produtor_origem', lambda x: 0 if (x == "OBRAS_RCD").all() else 1)
    )

    metrics_df = metrics_df.join(dest_agg).join(prod_agg).fillna(0)

    eps = 1e-9
    metrics_df['degree_imbalance_ratio'] = (metrics_df['in_degree'] - metrics_df['out_degree']) / (metrics_df['in_degree'] + metrics_df['out_degree'] + eps)
    metrics_df['weighted_degree_imbalance_ratio'] = (metrics_df['incoming_total'] - metrics_df['outgoing_total']) / (metrics_df['incoming_total'] + metrics_df['outgoing_total'] + eps)
    metrics_df['disposed_quantity_ratio'] = metrics_df['incoming_disposable'] / (metrics_df['incoming_total'] + eps)
    metrics_df['dangerous_quantity_difference_ratio'] = (metrics_df['dangerous_received_quantity'] - metrics_df['dangerous_sent_quantity']) / (metrics_df['dangerous_received_quantity'] + metrics_df['dangerous_sent_quantity'] + eps)
    metrics_df['proportion_individual_suppliers'] = metrics_df['indiv_count'] / (metrics_df['total_inc_transfers'] + eps)

    return metrics_df.reset_index().fillna(0)