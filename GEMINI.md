# AI Disaster Prediction Model Guidelines

## Project Context
We are developing a state-of-the-art AI model capable of generating hazard prediction maps for four disaster types: Hail, Flooding, Heatstress, and Landslides across the Contiguous United States (CONUS).

## Core Principles & Architecture Focus
*   **Resolution Synchronization:** A strict requirement is to use a unified common spatial resolution (e.g., 2.5km or 4km) for all inputs and targets. All data preprocessing pipelines must align spatial grids and temporal overlaps precisely.
*   **Input Data:** The primary dynamic input relies on advanced global weather models like DeepMind's WeatherNext-2 / GraphCast / GenCast.
*   **Static Features:** We incorporate static geographical and topographical data (e.g., LULC, DEM/elevation, SZA, soil moisture).
*   **Target Ground Truth:**
    *   **Hail:** MRMS MESH data.
    *   **Heatstress:** NWS HeatRisk.
    *   **Floods:** USGS stream gages / NOAA NWM.
    *   **Landslides:** USGS National Landslide Inventory.
*   **Model Architectures:** Focus is not limited to Graph Neural Networks (GNNs). We are actively exploring and evaluating diffusion models, transformers, and hybrid architectures (e.g., PINNs, NeuralGCM) that might better model the spatio-temporal dynamics and rare extreme events.

## Workflow Execution Rules
1.  **Research First:** Never download data or build models without exhaustive, documented research into availability, resolution, and exact temporal overlapping windows.
2.  **Version Control:** Commit all major plans, reports, research notes, and code changes to the remote repository.
3.  **Extensibility:** Write modular code, allowing components (like a specific dataset loader) to be swapped without breaking the entire pipeline.
4.  **Reproducibility:** Document all data sources, processing scripts, and training configurations meticulously.