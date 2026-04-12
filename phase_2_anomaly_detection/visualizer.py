import matplotlib.pyplot as plt
import networkx as nx
from adjustText import adjust_text

def plot_subgraph(G, center_node, radius=1, save_path=None, influential_nodes=None):
    """Plots ego graph. Highlights influential nodes if provided by explainer."""
    sub = nx.ego_graph(G, center_node, radius=radius)
    fig, ax = plt.subplots(figsize=(12, 9))
    
    pos = nx.spring_layout(sub, seed=42, k=1.0)
    
    colors = []
    sizes = []
    for n in sub.nodes():
        if n == center_node:
            colors.append('#ef5350') # Red for target
            sizes.append(3000)
        elif influential_nodes and n in influential_nodes:
            colors.append('#ab47bc') # Purple for highly influential nodes causing anomaly
            sizes.append(2000)
        else:
            colors.append('#42a5f5') # Blue for normal
            sizes.append(1000)
    
    nx.draw_networkx_nodes(sub, pos, node_color=colors, node_size=sizes, edgecolors='white')
    nx.draw_networkx_edges(sub, pos, edge_color='#bbbbbb', alpha=0.5, connectionstyle='arc3,rad=0.1')
    
    texts = [ax.text(x, y, s=n, size=9, weight='bold' if n == center_node else 'normal') 
             for n, (x, y) in pos.items()]
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', alpha=0.3))

    plt.title(f"Entity Spotlight: {center_node}\n(Purple = High Explanability/Shapley Value)")
    ax.axis('off')
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_top_nodes(G, n=15, save_path=None):
    top = sorted(G.degree, key=lambda x: x[1], reverse=True)[:n]
    sub = G.subgraph([node[0] for node in top])
    fig, ax = plt.subplots(figsize=(12, 10))
    
    pos = nx.circular_layout(sub)
    nx.draw_networkx(sub, pos, with_labels=True, node_color='#ffca28', node_size=2000, 
                     edge_color='#cccccc', width=1.5, font_size=10, font_weight='bold')
    
    plt.title(f"Top {n} Most Active Entities")
    ax.axis('off')
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()