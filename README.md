# Batch Routing and TSP Visualization

## Overview
This project optimizes **order batching and routing** using the **Nearest Neighbor heuristic** for the Traveling Salesman Problem (TSP). It minimizes distances both **within each batch** and **between batch centroids**, with clear visualizations.

## Features
1. **Intra-Batch Optimization**: Computes the optimal route within each batch.
2. **Inter-Batch Optimization**: Finds the best route connecting batch centroids.
3. **Visualization**:
   - Colored regions for batches.
   - Black "X" markers for centroids.
   - Red dashed line for the optimal inter-batch route.

## Requirements
Install dependencies using:
```bash
pip install numpy pandas matplotlib
```

## How to Run
1. Save and execute the script:
   ```bash
   python batch_routing_tsp.py
   ```
2. Outputs:
   - **Console**: Routes and distances for each batch and centroid connection.
   - **Visualization**: Batches and optimal routes displayed graphically.

## Use Cases
- **Warehouse management**: Optimize order picking routes.
- **Delivery planning**: Group locations and find efficient delivery routes.

## Limitations
- The Nearest Neighbor heuristic may not always provide the optimal TSP solution. Use advanced methods for exact results.

Push the boundaries of logistics and routing with this tool!
