import math
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from pathlib import Path

def draw_shapley_evidence_map(target_node: str, crime_scene: nx.Graph, influential_nodes: dict, plot_dir: Path):
    """Generates the 2-panel Executive Dashboard for a single node investigation."""
    print(f"Generating the Executive Evidence Dashboard for {target_node}...")
    
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1])
    ax_graph = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    display_graph = nx.DiGraph(crime_scene)
    display_graph.remove_edges_from(nx.selfloop_edges(display_graph))

    pos = {target_node: (0.0, 0.0)}
    incoming = [n for n in display_graph.nodes() if n != target_node and display_graph.has_edge(n, target_node)]
    outgoing = [n for n in display_graph.nodes() if n != target_node and display_graph.has_edge(target_node, n) and n not in incoming]

    max_shap = max(list(influential_nodes.values()) + [1e-9])
    total_shap = sum(influential_nodes.values()) if influential_nodes else 1e-9

    def assign_fan(node_list, is_in):
        if not node_list: return
        node_list.sort(key=lambda x: influential_nodes.get(x, 0), reverse=True)
        start, end = (math.pi/2, 3*math.pi/2) if is_in else (-math.pi/2, math.pi/2)
        for i, n in enumerate(node_list):
            r = 3.5 - ((influential_nodes.get(n, 0) / max_shap) * 2.0)
            theta = start + (end - start) * (i / (len(node_list)-1 if len(node_list)>1 else 1))
            pos[n] = (r * math.cos(theta), r * math.sin(theta))

    assign_fan(incoming, True)
    assign_fan(outgoing, False)

    for u, v in display_graph.edges():
        ecolor = '#1f77b4' if v == target_node else ('#2ca02c' if u == target_node else '#ff7f0e')
        impact = influential_nodes.get(u if v == target_node else v, 0) / max_shap
        nx.draw_networkx_edges(display_graph, pos, edgelist=[(u,v)], edge_color=ecolor, 
                               width=1+(3*impact), alpha=0.3+(0.5*impact), ax=ax_graph, 
                               arrowsize=20, connectionstyle="arc3,rad=0.1")

    nx.draw_networkx_nodes(display_graph, pos, nodelist=[target_node], node_color='#d62728', node_size=2500, ax=ax_graph)
    if influential_nodes:
        p_nodes = [n for n in display_graph.nodes() if n != target_node]
        p_colors = [cm.get_cmap('YlOrRd')(influential_nodes.get(n, 0) / max_shap) for n in p_nodes]
        nx.draw_networkx_nodes(display_graph, pos, nodelist=p_nodes, node_color=p_colors, node_size=800, ax=ax_graph)

    labels = {n: f"TARGET\n{n}" if n == target_node else (f"{n}\n({(influential_nodes.get(n, 0)/total_shap)*100:.1f}%)" if (influential_nodes.get(n, 0)/total_shap) > 0.05 else "") for n in display_graph.nodes()}
    nx.draw_networkx_labels(display_graph, pos, labels=labels, font_size=8, ax=ax_graph)
    
    sorted_acc = sorted(influential_nodes.items(), key=lambda x: x[1], reverse=True)
    top_acc = f"{sorted_acc[0][0]} ({(sorted_acc[0][1]/total_shap)*100:.1f}%)" if sorted_acc else "N/A"
    insight = f"INVESTIGATION SUMMARY\nTarget: {target_node}\nStructure: {len(incoming)} In | {len(outgoing)} Out\nTop Accomplice: {top_acc}"
    ax_graph.text(0.02, 0.02, insight, transform=ax_graph.transAxes, bbox=dict(facecolor='white', alpha=0.9), family='monospace')
    ax_graph.set_title(f"GNN SubgraphX Evidence: {target_node}", fontsize=15, fontweight='bold')
    ax_graph.axis('off')

    if influential_nodes:
        names = [str(x[0]) for x in sorted_acc][:15]
        pcts = [(x[1]/total_shap)*100 for x in sorted_acc][:15]
        ax_bar.barh(names[::-1], pcts[::-1], color='#ff4d4d')
        ax_bar.set_title("Accomplice Impact %")
        ax_bar.set_xlabel("Contribution to Anomaly Score")

    plt.tight_layout()
    plt.savefig(plot_dir / f"subgraphx_dashboard_{target_node}.png", dpi=300)
    plt.close()

def draw_business_suspect_comparison(suspect_nodes: list, df_results: pd.DataFrame, plot_dir: Path):
    """Generates a Radar Chart comparing suspects based on their Model Rankings."""
    print("Generating Business Suspect Model Ranking Radar...")
    
    rank_metrics = ['Rank_IF', 'Rank_LOF', 'Rank_KMeans', 'Rank_AE']
    labels = ['Isolation Forest', 'LOF', 'K-Means', 'GNN Autoencoder']
    
    total_nodes = len(df_results)
    plot_data = []
    found_suspects = []
    
    for node in suspect_nodes:
        if node in df_results['Node_ID'].values:
            row = df_results[df_results['Node_ID'] == node].iloc[0]
            # Scoring: (1 - Rank/Total) * 100. Rank 1 becomes ~100% score.
            node_scores = [(1 - (row[m] / total_nodes)) * 100 for m in rank_metrics]
            plot_data.append(node_scores)
            found_suspects.append(node)

    if not plot_data:
        return

    angles = np.linspace(0, 2*np.pi, len(rank_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, scores in enumerate(plot_data):
        values = scores + scores[:1]
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=f"Node {found_suspects[i]}")
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)

    # Casting to clear IDE warnings if needed, though dict(polar=True) works at runtime
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight='bold')
    
    plt.ylim(0, 100)
    ax.set_yticklabels(["Low Risk", "20%", "40%", "60%", "80%", "High Risk"], color="gray", size=9)
    plt.title("Business Suspects: Model Consensus Profile", size=15, pad=30, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    plt.savefig(plot_dir / "business_suspects_radar_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model Ranking Radar saved to: {plot_dir / 'business_suspects_radar_comparison.png'}")