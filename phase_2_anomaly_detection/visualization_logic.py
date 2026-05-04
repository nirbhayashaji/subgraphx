import math
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from pathlib import Path

def draw_shapley_evidence_map(target_node: str, crime_scene: nx.Graph, influential_nodes: dict, df_results: pd.DataFrame, df_features: pd.DataFrame, plot_dir: Path):
    """Generates the 2-panel  Dashboard for a single node investigation."""
    print(f"Generating the  Evidence Dashboard for {target_node}...")
    
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

    # 1. Draw Edges
    for u, v in display_graph.edges():
        ecolor = '#1f77b4' if v == target_node else ('#2ca02c' if u == target_node else '#ff7f0e')
        impact = influential_nodes.get(u if v == target_node else v, 0) / max_shap
        nx.draw_networkx_edges(display_graph, pos, edgelist=[(u,v)], edge_color=ecolor, 
                               width=1+(3*impact), alpha=0.3+(0.5*impact), ax=ax_graph, 
                               arrowsize=20, connectionstyle="arc3,rad=0.1")

    # 2. Draw Nodes
    nx.draw_networkx_nodes(display_graph, pos, nodelist=[target_node], node_color='#d62728', node_size=2500, ax=ax_graph)
    if influential_nodes:
        p_nodes = [n for n in display_graph.nodes() if n != target_node]
        p_colors = [cm.get_cmap('YlOrRd')(influential_nodes.get(n, 0) / max_shap) for n in p_nodes]
        nx.draw_networkx_nodes(display_graph, pos, nodelist=p_nodes, node_color=p_colors, node_size=800, ax=ax_graph)

    # 3. Draw Labels
    labels = {n: f"TARGET\n{n}" if n == target_node else (f"{n}\n({(influential_nodes.get(n, 0)/total_shap)*100:.1f}%)" if (influential_nodes.get(n, 0)/total_shap) > 0.05 else "") for n in display_graph.nodes()}
    nx.draw_networkx_labels(display_graph, pos, labels=labels, font_size=8, ax=ax_graph)
    
    # --- NEW ARROW LEGEND (Properly Placed) ---
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=3, label='Incoming to Target'),
        Line2D([0], [0], color='#2ca02c', lw=3, label='Outgoing from Target'),
        Line2D([0], [0], color='#ff7f0e', lw=3, label='Accomplice to Accomplice')
    ]
    ax_graph.legend(handles=legend_elements, loc='upper left', 
                    title="Transaction Flow", fontsize=9, title_fontproperties={'weight':'bold'},
                    bbox_to_anchor=(0.02, 0.98), framealpha=0.95, edgecolor='#343a40')

    ax_graph.set_title(f"GNN SubgraphX Evidence: {target_node}", fontsize=15, fontweight='bold')
    ax_graph.axis('off')

    # --- NEW BUSINESS INTELLIGENCE PROFILE ---
    # 1. Safely extract rankings
    try:
        node_result = df_results[df_results['Node_ID'] == target_node].iloc[0]
        rank_text = f"Ensemble Rank: #{int(node_result.get('Overall_Rank', 0))} (out of {len(df_results)})"
    except (IndexError, KeyError):
        rank_text = "Ensemble Rank: N/A"

    # 2. Extract physical domain features using exact column names
    try:
        node_features = df_features[df_features['Node_ID'] == target_node].iloc[0]
        
        # Volume metrics
        inc_vol = node_features.get('incoming_total', 0)
        out_vol = node_features.get('outgoing_total', 0)
        
        # Hazardous/Dangerous material logic
        inc_danger = node_features.get('dangerous_received_quantity', 0)
        flag_rate = (inc_danger / inc_vol) * 100 if inc_vol > 0 else 0
        
        # Supplier demographics
        indiv_prop = node_features.get('proportion_individual_suppliers', 0) * 100
        
        # Structural connections
        in_deg = int(node_features.get('in_degree', 0))
        out_deg = int(node_features.get('out_degree', 0))

        domain_text = (
            f"Physical Transfer Profile:\n"
            f"- Volume Flow   : {inc_vol:,.0f} In | {out_vol:,.0f} Out\n"
            f"- Hazardous Rate: {flag_rate:.1f}% (Incoming)\n"
            f"- Supplier Demo : {indiv_prop:.1f}% Individual\n"
            f"- Connections   : {in_deg} Suppliers | {out_deg} Receivers"
        )
    except (IndexError, KeyError):
        domain_text = "Physical Transfer Profile: N/A"

    # 3. Format Top Accomplice from SubgraphX
    sorted_acc = sorted(influential_nodes.items(), key=lambda x: x[1], reverse=True)
    top_acc = f"{sorted_acc[0][0]} ({(sorted_acc[0][1]/total_shap)*100:.1f}%)" if sorted_acc else "N/A"

    # 4. Combine all insights into one powerful text block
    insight = (
        f"TARGET INTELLIGENCE PROFILE\n"
        f"Node ID: {target_node}\n"
        f"====================================\n"
        f"{rank_text}\n\n"
        f"{domain_text}\n\n"
        f"SUBGRAPHX MCTS EXPLANATION\n"
        f"- Subgraph Motif: {len(incoming)} In | {len(outgoing)} Out\n"
        f"- Primary Driver: {top_acc}"
    )

    # 5. Draw the box in the bottom left corner
    ax_graph.text(0.02, 0.02, insight, transform=ax_graph.transAxes, 
                  bbox=dict(boxstyle="round,pad=0.6", facecolor='#f8f9fa', edgecolor='#343a40', alpha=0.95), 
                  family='monospace', fontsize=9, verticalalignment='bottom')

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
    """Generates a clean, presentation-ready 1x2 Consensus Dashboard."""
    print("Generating Presentation-Ready 1x2 Consensus Dashboard...")
    
    # --- 1. DATA PREPARATION ---
    rank_metrics = ['Rank_IF', 'Rank_LOF', 'Rank_KMeans', 'Rank_AE']
    labels = ['Isolation Forest', 'LOF', 'K-Means', 'GNN Autoencoder']
    
    # Bump Chart Data (Wider net: Top 10 from each model + suspects for background noise)
    top_nodes_sets = [set(df_results.nsmallest(10, m)['Node_ID']) for m in rank_metrics + ['Overall_Rank']]
    bump_nodes = set().union(*top_nodes_sets).union(set(suspect_nodes))
    bump_df = df_results[df_results['Node_ID'].isin(bump_nodes)].set_index('Node_ID')
    
    # Heatmap Data (Strict filter: Only Top 5 Overall + Suspects)
    top5_overall = set(df_results.nsmallest(5, 'Overall_Rank')['Node_ID'])
    heatmap_nodes = list(top5_overall.union(set(suspect_nodes)))
    heatmap_df = df_results[df_results['Node_ID'].isin(heatmap_nodes)].set_index('Node_ID')
    heatmap_df = heatmap_df.sort_values('Overall_Rank') # Sort by overall consensus
    ranks_df = heatmap_df[rank_metrics + ['Overall_Rank']]

    # --- 2. SETUP 1x2 FIGURE ---
    fig = plt.figure(figsize=(18, 7)) # Wide format for presentation slides
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.25)

    # --- PLOT A: BUMP CHART / SLOPEGRAPH (Left) ---
    ax1 = fig.add_subplot(gs[0])
    x_positions = np.arange(5)
    
    for node in bump_df.index:
        y_vals = bump_df.loc[node][rank_metrics + ['Overall_Rank']].values
        if node in suspect_nodes:
            ax1.plot(x_positions, y_vals, color='#d62728', linewidth=3.5, marker='o', markersize=8, zorder=10)
            ax1.text(4.1, y_vals[-1], f"{node}", color='#d62728', fontweight='bold', va='center')
        else:
            ax1.plot(x_positions, y_vals, color='#adb5bd', linewidth=1, alpha=0.3, marker='.')

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(labels + ['OVERALL'], fontweight='bold', size=11)
    ax1.invert_yaxis() # Rank 1 at the top
    ax1.set_yscale('log') # Log scale because ranks go from 1 to 160,000
    ax1.set_ylabel("Rank (Log Scale)", fontweight='bold', size=12)
    ax1.set_title("A. Rank Journey: Known Suspects vs Top AI Anomalies", size=15, fontweight='bold', pad=15)
    
    # Add custom legend for Bump Chart
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#d62728', lw=3, label='Known Business Suspects'),
        Line2D([0], [0], color='#adb5bd', lw=1, label='Unsupervised Top Anomalies')
    ]
    ax1.legend(handles=legend_elements, loc='lower left')

    # --- PLOT B: FILTERED CONSENSUS HEATMAP (Right) ---
    ax2 = fig.add_subplot(gs[1])
    # Cap ranks at 2000 for color scaling so deep red = Top 10, green = safe
    heatmap_data = np.clip(ranks_df.values, 1, 2000) 
    cax = ax2.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    
    ax2.set_xticks(np.arange(5))
    ax2.set_xticklabels(labels + ['OVERALL'], fontweight='bold', rotation=35, ha='right', size=11)
    
    # Highlight known suspects in red on the y-axis
    y_labels = []
    for idx in ranks_df.index:
        if idx in suspect_nodes:
            y_labels.append(f"★ {idx} (Suspect)")
        else:
            y_labels.append(str(idx))
            
    ax2.set_yticks(np.arange(len(ranks_df)))
    ax2.set_yticklabels(y_labels, size=10)
    
    # Color the suspect y-ticks red
    for i, label in enumerate(ax2.get_yticklabels()):
        if "★" in label.get_text():
            label.set_color('#d62728')
            label.set_fontweight('bold')

    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax2)
    cbar.set_label('Model Rank (Capped at >2000)', rotation=270, labelpad=15, fontweight='bold')
    ax2.set_title("B. Model Consensus: Top 5 Anomalies + Suspects", size=15, fontweight='bold', pad=15)

    # --- SAVE OUTPUT ---
    plt.tight_layout()
    output_filepath = plot_dir / "presentation_consensus_dashboard.png"
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Presentation Dashboard saved to: {output_filepath}")
    """Generates a comprehensive 2x2 Master Consensus Dashboard."""
    print("Generating Comprehensive 2x2 Master Consensus Dashboard...")
    
    # --- 1. DATA PREPARATION ---
    rank_metrics = ['Rank_IF', 'Rank_LOF', 'Rank_KMeans', 'Rank_AE']
    labels = ['Isolation Forest', 'LOF', 'K-Means', 'GNN Autoencoder']
    total_nodes = len(df_results)
    
    # Get Top 10 from each model and overall
    top_nodes_sets = [set(df_results.nsmallest(10, m)['Node_ID']) for m in rank_metrics + ['Overall_Rank']]
    all_top_nodes = set().union(*top_nodes_sets)
    
    # Combine Top nodes + Known Suspects
    target_nodes = list(all_top_nodes.union(set(suspect_nodes)))
    plot_df = df_results[df_results['Node_ID'].isin(target_nodes)].set_index('Node_ID')
    plot_df = plot_df.sort_values('Overall_Rank') # Sort by overall consensus
    ranks_df = plot_df[rank_metrics + ['Overall_Rank']]

    # --- 2. SETUP 2x2 FIGURE ---
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)

    # --- PLOT 1: RADAR CHART (Top Left) ---
    ax1 = fig.add_subplot(gs[0, 0], polar=True)
    radar_data = []
    found_suspects = [n for n in suspect_nodes if n in plot_df.index]
    
    for node in found_suspects:
        row = plot_df.loc[node]
        scores = [(1 - (row[m] / total_nodes)) * 100 for m in rank_metrics]
        radar_data.append(scores)

    angles = np.linspace(0, 2*np.pi, len(rank_metrics), endpoint=False).tolist()
    angles += angles[:1]
    colors = ['#d62728', '#1f77b4', '#ff7f0e'] # Red for suspects

    if radar_data:
        for i, scores in enumerate(radar_data):
            values = scores + scores[:1]
            ax1.plot(angles, values, color=colors[i % len(colors)], linewidth=2.5, label=f"Suspect {found_suspects[i]}")
            ax1.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)

    ax1.set_theta_offset(np.pi / 2) # type: ignore
    ax1.set_theta_direction(-1)     # type: ignore
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, fontweight='bold', size=12)
    ax1.set_ylim(0, 100)
    ax1.set_yticklabels(["Low Risk", "20%", "40%", "60%", "80%", "High Risk"], color="gray", size=10)
    ax1.set_title("A. Specific Business Suspects (Radar)", size=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # --- PLOT 2: CONSENSUS HEATMAP (Top Right) ---
    ax2 = fig.add_subplot(gs[0, 1])
    # Cap ranks at 2000 for color scaling so deep red = Top 10, white = >2000
    heatmap_data = np.clip(ranks_df.values, 1, 2000) 
    cax = ax2.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    
    ax2.set_xticks(np.arange(5))
    ax2.set_xticklabels(labels + ['OVERALL'], fontweight='bold', rotation=45, ha='right')
    
    # Highlight known suspects in red on the y-axis
    y_labels = []
    for idx in ranks_df.index:
        if idx in suspect_nodes:
            y_labels.append(f"★ {idx} (Suspect)")
        else:
            y_labels.append(str(idx))
            
    ax2.set_yticks(np.arange(len(ranks_df)))
    ax2.set_yticklabels(y_labels, size=8)
    
    # Color the suspect y-ticks red
    for i, label in enumerate(ax2.get_yticklabels()):
        if "★" in label.get_text():
            label.set_color('#d62728')
            label.set_fontweight('bold')

    fig.colorbar(cax, ax=ax2, label='Rank (Capped at 2000)')
    ax2.set_title("B. Model Consensus Heatmap (Top Nodes)", size=16, fontweight='bold', pad=20)

    # --- PLOT 3: BUMP CHART / SLOPEGRAPH (Bottom Left) ---
    ax3 = fig.add_subplot(gs[1, 0])
    x_positions = np.arange(5)
    
    for node in ranks_df.index:
        y_vals = ranks_df.loc[node].values
        if node in suspect_nodes:
            ax3.plot(x_positions, y_vals, color='#d62728', linewidth=3, marker='o', markersize=8, zorder=10)
            ax3.text(4.1, y_vals[-1], f"{node}", color='#d62728', fontweight='bold', va='center')
        else:
            ax3.plot(x_positions, y_vals, color='#adb5bd', linewidth=1, alpha=0.4, marker='.')

    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(labels + ['OVERALL'], fontweight='bold')
    ax3.invert_yaxis() # Rank 1 at the top
    ax3.set_yscale('log') # Log scale because ranks go from 1 to 160,000
    ax3.set_ylabel("Rank (Log Scale)", fontweight='bold')
    ax3.set_title("C. Rank Journey (Suspects vs AI Anomalies)", size=16, fontweight='bold', pad=20)

    # --- PLOT 4: ENSEMBLE OVERLAP (Bottom Right) ---
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Count how many models put each node in their Top 100
    top_100_sets = [set(df_results.nsmallest(100, m)['Node_ID']) for m in rank_metrics]
    overlap_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for node in df_results['Node_ID']:
        appearances = sum(1 for s in top_100_sets if node in s)
        if appearances > 0:
            overlap_counts[appearances] += 1

    bars = ax4.bar(list(overlap_counts.keys()), list(overlap_counts.values()), color=['#ced4da', '#6c757d', '#1f77b4', '#d62728'])
    ax4.set_xticks([1, 2, 3, 4])
    ax4.set_xticklabels(['1 Model', '2 Models', '3 Models', 'All 4 Models (Strong Consensus)'], fontweight='bold')
    ax4.set_ylabel("Number of Nodes in Top 100")
    ax4.set_title("D. Top 100 Anomaly Intersection", size=16, fontweight='bold', pad=20)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.5, str(int(yval)), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_filepath = plot_dir / "master_consensus_dashboard.png"
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Master Dashboard saved to: {output_filepath}")