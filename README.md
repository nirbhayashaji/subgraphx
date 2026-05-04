# SubgraphX-GNN: Explainable Graph Anomaly Detection Pipeline

## Overview
This repository contains an end-to-end Machine Learning pipeline designed to detect and explain anomalous nodes (e.g., fraudulent entities, illicit networks) within complex transfer networks. 

It utilizes a multi-model ensemble approach—combining traditional density-based outlier detection with Graph Neural Networks (GNNs)—and leverages SubgraphX (Monte Carlo Tree Search + Shapley Values) to provide human-readable, forensic explanations of *why* the GNN flagged a specific node.

## Architecture & Process Flow
The pipeline operates in three distinct phases: Feature Engineering, Anomaly Detection, and Explainability/Visualization.

```text
[Raw Data] 
    │
    ├──> processing.py (Data Cleaning)
    │
    ├─ Phase 1: Feature Engineering 
    │    ├── graph_feature_creation.py (Structural Metrics: Centrality, Degree)
    │    └── transfer_feature_creation.py (Domain Metrics: Volume, Hazard Rates)
    │
    ├─ Phase 2: Anomaly Detection Ensemble (anomaly_engine.py)
    │    ├── Isolation Forest (IF)
    │    ├── Local Outlier Factor (LOF)
    │    ├── K-Means Clustering
    │    ├── GNN Autoencoder (model_ae.py)
    │    └──> Borda Count Aggregation (Calculates Overall Consensus Rank)
    │
    └─ Phase 3: Explainability & Visualization (explainer_logic.py)
         ├── SubgraphX MCTS (Isolates the anomalous subgraph/motif)
         ├── Shapley Value Calculation (Scores accomplice impact)
         └── visualization_logic.py (Generates Master Dashboards)

Usage

To run the full end-to-end pipeline and enter Interactive Investigation Mode: 
python main.py
To bypass ML training and generate explainability dashboards for known suspects directly: 
python phase_2_anomaly_detection/explainer_logic.py


---

