# SubgraphX: Directed Network Feature Engineering Pipeline

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Theory-lightgrey)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458)
![Status](https://img.shields.io/badge/Status-Phase_1_Complete-success)

## 📖 Overview
This repository contains a highly optimized, scalable Python pipeline designed to extract structural and domain-specific features from large-scale directed transfer networks (e.g., supply chains, logistics, or transactional networks). 

It acts as the critical feature-engineering phase for downstream **Graph-based Anomaly Detection** and **Graph Explainability** models. The pipeline is specifically optimized to handle massive graphs (100k+ nodes, 200k+ edges) by utilizing Pandas vectorization and supporting pre-calculated structural metrics to bypass standard computational bottlenecks.

## 🧠 A Primer on Graph Explainability & SubgraphX
*Why do we need this pipeline?*

In modern machine learning, Graph Neural Networks (GNNs) are used to detect anomalies in complex networks (like fraudulent transactions or supply chain bottlenecks). However, GNNs act as "black boxes." 

**Explainability algorithms** (like the *SubgraphX* algorithm) solve this by identifying the specific subset of nodes and edges (the "subgraph") that caused the GNN to flag an anomaly. 
**However, for these explainers to work accurately, the initial graph must be incredibly feature-rich.**

This pipeline bridges that gap by transforming raw tabular data into rich node-level vectors consisting of:
1. **Structural Features:** How important is this node to the network? (Centralities, Degrees).
2. **Domain Features:** What is the physical nature of the transfers happening here? (Imbalances, Quantities, Hazards).

## 🚀 Core Capabilities
* **Domain Feature Generation:** Calculates weighted degree imbalances, transactional quantity differences, and node demographics in $O(1)$ time via vectorized aggregations.
* **Gephi Fast-Track Integration:** Allows the ingestion of pre-calculated $O(V \times E)$ graph centralities (e.g., Betweenness, PageRank) from Gephi, merging them seamlessly with transactional domain features.
* **Interactive Execution:** CLI-based prompts allow users to test on dataset subsets, recalculate features from scratch, or execute a "Super Fast-Track" mode that merges cached metrics in seconds.
* **Global Graph Summarization:** Automatically generates comprehensive statistical fingerprints (Mean, Min, Max) of the entire network per run for temporal tracking.

---

## ⚙️ Prerequisites & Setup

### Prerequisites
* **OS:** Linux, macOS, or Windows
* **Python:** v3.10 or higher
* **Memory:** 8GB+ RAM recommended for graphs exceeding 150,000 nodes.

### Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR-USERNAME/subgraphx.git](https://github.com/YOUR-USERNAME/subgraphx.git)
cd subgraphx