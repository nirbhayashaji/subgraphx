import math
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from pathlib import Path

def draw_shapley_evidence_map(target_node: str, crime_scene: nx.Graph, influential_nodes: dict, plot_dir: Path):
    """
    Final Refined Evidence Map for Waste Management Networks:
    - Blue = Incoming (Suppliers)
    - Green = Outgoing (Destinations)
    - Orange = Peer-to-Peer (Internal backchannels arching outward to expose collusion)
    - Radius-based spacing for high-density neighborhoods (100+ suppliers)
    """
    print(f"Drawing the color-coded evidence map for {target_node}...")
    plt.figure(figsize=(18, 14))

    # Create a copy for display to safely remove self-loops for visual clarity
    display_graph = nx.DiGraph(crime_scene)
    display_graph.remove_edges_from(nx.selfloop_edges(display_graph))

    # ==========================================
    # 1. LAYOUT: INCREASED RADIUS FOR SPACING
    # ==========================================
    pos = {target_node: (0.0, 0.0)}

    incoming_nodes = [n for n in display_graph.nodes() if n != target_node and display_graph.has_edge(n, target_node)]
    outgoing_nodes = [n for n in display_graph.nodes() if n != target_node and display_graph.has_edge(target_node, n) and n not in incoming_nodes]

    max_shap_val = max(list(influential_nodes.values()) + [1e-9]) if influential_nodes else 1e-9
    total_shap_val = sum(influential_nodes.values()) if influential_nodes else 1e-9

    def assign_fan_positions(node_list, is_incoming):
        if not node_list: return
        node_list.sort(key=lambda x: influential_nodes.get(x, 0), reverse=True)
        n_nodes = len(node_list)
        
        # Determine which side of the target node they sit on (Incoming=Left, Outgoing=Right)
        start_angle, end_angle = (math.pi/2, 3*math.pi/2) if is_incoming else (-math.pi/2, math.pi/2)

        for i, n in enumerate(node_list):
            norm_shap = influential_nodes.get(n, 0) / max_shap_val
            # INCREASED SPACING: Radius ranges from 1.5 to 3.5 to prevent label overlap
            r = 3.5 - (norm_shap * 2.0)

            if n_nodes == 1:
                theta = (start_angle + end_angle) / 2
            else:
                pad = 0.2 
                theta = (start_angle + pad) + (end_angle - start_angle - 2*pad) * (i / (n_nodes - 1))
            pos[n] = (r * math.cos(theta), r * math.sin(theta))

    assign_fan_positions(incoming_nodes, is_incoming=True)
    assign_fan_positions(outgoing_nodes, is_incoming=False)

    # ==========================================
    # 2. EDGE LOGIC: DIRECTIONAL COLORS & OUTWARD ARCHING
    # ==========================================
    # Calculate node sizes list in advance for edge padding fix
    node_sizes_list = [2500 if n == target_node else (influential_nodes.get(n, 0) / max_shap_val) * 1200 + 200 for n in display_graph.nodes()]

    for u, v in display_graph.edges():
        # 1. Incoming (Supplier -> Target)
        if v == target_node:
            ecolor = '#1f77b4' # Steel Blue
            curve_rad = 0.1    # Subtle curve inward
            weight_node = u
            
        # 2. Outgoing (Target -> Destination)
        elif u == target_node:
            ecolor = '#2ca02c' # Forest Green
            curve_rad = 0.1    # Subtle curve inward
            weight_node = v
            
        # 3. INTERNAL PEER-TO-PEER (Neighbor -> Neighbor)
        else:
            ecolor = '#ff7f0e' # Safety Orange
            # ARCH OUTWARD: Pushes collusion paths outside the main business flow
            curve_rad = 0.45    
            weight_node = u 

        # Visual weight (thickness/alpha) based on relative Shapley contribution
        impact = influential_nodes.get(weight_node, 0) / max_shap_val
        alpha = 0.3 + (0.5 * impact)
        width = 1.0 + (3.5 * impact)

        nx.draw_networkx_edges(
            display_graph, pos, 
            edgelist=[(u, v)], 
            arrowstyle='-|>', 
            arrowsize=20 + (10 * impact),
            edge_color=ecolor,
            width=width, 
            alpha=alpha, 
            connectionstyle=f"arc3,rad={curve_rad}",
            node_size=node_sizes_list # Ensures arrowheads are visible outside the circles
        )

    # ==========================================
    # 3. NODE & LABEL STYLING
    # ==========================================
    # Target Node (Red)
    nx.draw_networkx_nodes(display_graph, pos, nodelist=[target_node], node_color='#d62728', node_size=2500, edgecolors='black', linewidths=2)

    # Accomplice Nodes (Heatmap)
    if influential_nodes:
        p_nodes = [n for n in display_graph.nodes() if n != target_node]
        p_sizes = [s for n, s in zip(display_graph.nodes(), node_sizes_list) if n != target_node]
        p_colors = [cm.get_cmap('YlOrRd')(influential_nodes.get(n, 0) / max_shap_val) for n in p_nodes]
        nx.draw_networkx_nodes(display_graph, pos, nodelist=p_nodes, node_color=p_colors, node_size=p_sizes, edgecolors='#555555', linewidths=1)

    # Labels with Smart Decluttering (showing Impact %)
    labels = {}
    for n in display_graph.nodes():
        if n == target_node:
            labels[n] = f"TARGET\n{n}"
        else:
            impact_pct = (influential_nodes.get(n, 0) / total_shap_val) * 100
            # Only label if impact > 0.5% to maintain visual quality
            if impact_pct > 0.5:
                labels[n] = f"{n}\n({impact_pct:.1f}%)"
            else:
                labels[n] = ""

    nx.draw_networkx_labels(
        display_graph, pos, labels=labels, font_size=8, font_weight="bold", 
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5)
    )

    # Business Flow Legend
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=3, label='Incoming Flow (Suppliers)'),
        Line2D([0], [0], color='#2ca02c', lw=3, label='Outgoing Flow (Destinations)'),
        Line2D([0], [0], color='#ff7f0e', lw=3, label='Internal Path (Collusion Bridge)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', title="Transaction Flow Legend")

    plt.title(f"GNN SubgraphX Evidence: Node {target_node}\nStructural Anomaly Explanation", fontsize=18, fontweight='bold', pad=25)
    plt.axis('off')

    # Save finalized plot
    plot_path = plot_dir / f"subgraphx_shapley_{target_node}.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Investigation complete. Visual evidence saved to: {plot_path}")