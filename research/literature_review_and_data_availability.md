# AI Disaster Prediction: Literature Review & Data Availability Report

## 1. Temporal Data Availability & Synchronization Strategy

To train a robust Spatio-Temporal AI model, the dynamic weather inputs and ground-truth disaster datasets must overlap chronologically. 

### 1.1 Ground Truth Datasets
*   **Hail (MRMS MESH):** 
    *   *Availability:* Operational data is solid from **October 2014 to Present** (historical reanalysis exists for 1998-2011).
    *   *Resolution:* ~1km.
*   **Heatstress (NWS HeatRisk):**
    *   *Availability:* Historical reconstructions available from **2005 to Present**.
    *   *Resolution:* ~2.5km.
*   **Landslides (USGS National Landslide Inventory):**
    *   *Availability:* Continuous, spanning decades up to **2024**. Data precision varies (some exact dates, some date ranges).
*   **Flooding (USGS / NOAA NWM):**
    *   *Availability:* Standard gage and model proxy data is robust over the last decade.

### 1.2 Input Features (Google DeepMind Weather Data)
*   **WeatherBench 2 (ERA5 based):** Contains historical baseline inputs from **1959 to 2023**.
*   **WeatherNext 2 / Gen (DeepMind):** Real-time operational data is from **2022 to Present** (historic forecasts from 2019/2020). 

### 1.3 Recommended Overlap Window
**The optimal training/validation window is 2015 to 2023.**
*   This window ensures full coverage from MRMS (post-2014), HeatRisk, Landslides, and ERA5/WeatherBench 2 data. 
*   We can use 2015–2021 for training and 2022–2023 for validation/testing.

---

## 2. SOTA Architectures: GNNs vs. Diffusion Models

Recent literature shows a massive paradigm shift in AI weather and extreme event modeling from pure Deterministic GNNs to Generative Diffusion Models.

### 2.1 Deterministic GNNs (e.g., GraphCast)
*   **How they work:** Process weather states on an icosahedral mesh using Graph Neural Networks, optimized via Mean Squared Error (MSE).
*   **Pros:** Extremely fast, highly accurate for standard atmospheric variables (RMSE), and relatively easier to train.
*   **Cons:** Because they optimize for average error, they tend to "blur" out extreme, rare events (like peak heatwaves or intense storm structures) at longer lead times.

### 2.2 Diffusion Models (e.g., GenCast, SEEDS)
*   **How they work:** Learn the probability distribution of weather states using a denoising framework (often utilizing a GNN or Transformer backbone).
*   **Pros:** Currently the SOTA for predicting **extremes and generating ensembles**. They maintain physical realism (sharp gradients) and don't "average out" severe disasters.
*   **Cons:** Slower inference time (iterative denoising) and requires more complex training setups.

### 2.3 Hybrid / Foundation Models (e.g., NeuralGCM)
Combine differentiable physics solvers with ML to ensure predictions obey physical conservation laws (mass, momentum), which is crucial for hydrology (flooding).

## 3. Recommendations for Model Development
1.  **Architecture:** Given that disasters (hail, landslides, extreme heat) are rare "tail" events, a **Probabilistic Diffusion Model with a GNN Backbone (similar to GenCast)** is highly recommended over a deterministic GNN. It will capture the peak intensities much better.
2.  **Dataset Sourcing:** For the 2015-2023 window, we should target the **WeatherBench 2 ERA5** cloud buckets (Zarr format) as our primary historical input feature set. 
3.  **Resolution:** We should standardize everything to a **0.25° (~25km) or 2.5km super-resolution grid** depending on compute constraints, regridding the 1km MRMS and point-based landslide data to match the chosen grid.