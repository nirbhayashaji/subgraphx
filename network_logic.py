import networkx as nx
import pandas as pd
import numpy as np

def build_transfer_network(df: pd.DataFrame) -> nx.DiGraph:
    """Creates a Directed Graph from unique transfers."""
    unique_transfers = df.groupby(['source_node', 'target_node']).agg(
        total_sum_quantity=('transfer_weight', 'sum'),
        flagged_sum_quantity=('flagged_qty', 'sum')
    ).reset_index()

    G = nx.from_pandas_edgelist(
        unique_transfers, source='source_node', target='target_node', 
        edge_attr=True, create_using=nx.DiGraph
    )
    entities = pd.unique(df[['source_node', 'target_node']].values.ravel('K'))
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

    dest_agg = df.groupby('target_node').agg(
        incoming_total=('transfer_weight', 'sum'),
        incoming_terminal=('terminal_qty', 'sum'),
        variance_incoming=('transfer_weight', 'var'),
        flagged_received_quantity=('flagged_qty', 'sum'),
        total_inc_transfers=('transfer_weight', 'count'),
        indiv_count=('is_individual', 'sum')
    )

    prod_agg = df.groupby('source_node').agg(
        outgoing_total=('transfer_weight', 'sum'),
        variance_outgoing=('transfer_weight', 'var'),
        flagged_sent_quantity=('flagged_qty', 'sum'),
        not_only_special_category=('source_category', lambda x: 0 if (x == "SPECIAL_CATEGORY").all() else 1)
    )

    metrics_df = metrics_df.join(dest_agg).join(prod_agg).fillna(0)

    eps = 1e-9
    metrics_df['degree_imbalance_ratio'] = (metrics_df['in_degree'] - metrics_df['out_degree']) / (metrics_df['in_degree'] + metrics_df['out_degree'] + eps)
    metrics_df['weighted_degree_imbalance_ratio'] = (metrics_df['incoming_total'] - metrics_df['outgoing_total']) / (metrics_df['incoming_total'] + metrics_df['outgoing_total'] + eps)
    metrics_df['terminal_quantity_ratio'] = metrics_df['incoming_terminal'] / (metrics_df['incoming_total'] + eps)
    metrics_df['flagged_quantity_difference_ratio'] = (metrics_df['flagged_received_quantity'] - metrics_df['flagged_sent_quantity']) / (metrics_df['flagged_received_quantity'] + metrics_df['flagged_sent_quantity'] + eps)
    metrics_df['proportion_individual_suppliers'] = metrics_df['indiv_count'] / (metrics_df['total_inc_transfers'] + eps)

    return metrics_df.reset_index().fillna(0)