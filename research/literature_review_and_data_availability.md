# AI Disaster Prediction: Literature Review & Data Availability Report

## 1. Temporal Data Availability & Synchronization Strategy

To train a robust Spatio-Temporal AI model, the dynamic weather inputs and ground-truth disaster datasets must overlap chronologically. 

### 1.1 Ground Truth Datasets
*   **Hail (MRMS MESH):** Operational data starts in **late 2014**.
*   **Heatstress (NWS HeatRisk):** Historical archive starts in **2005**.
*   **Landslides (USGS):** Continuous, up to **2024**.
*   **Flooding (USGS / NOAA NWM):** Robust over the last decade.

### 1.2 Input Features (Google DeepMind Weather Data)
*   **WeatherBench 2 (ERA5 based):** **1959 to 2023**.
*   **WeatherNext 2 / Gen:** Real-time from **2022 to Present**. 

### 1.3 Recommended Overlap Window
**The optimal training/validation window is 2015 to 2023.**

---

## 2. Expanded SOTA Architecture Analysis (2024-2026)

To determine the best architecture for disaster prediction (hail, floods, heat, landslides), we must evaluate the latest generative paradigms beyond standard GNNs and Diffusion.

### 2.1 Generative Paradigms Comparison

| Architecture | Core Mechanism | Pros for Disaster Prediction | Cons for Disaster Prediction | Notable Earth/Weather Models |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Diffusion (SDM)** | Solves Stochastic Differential Equations (SDE) to reverse Gaussian noise iteratively. | **SOTA for Ensembles.** Excellent at probabilistic tail-risk (extreme events) and high-fidelity textures. | **Slow inference.** Errors can compound over long rollouts (autoregressive steps). | GenCast, SEEDS, NVIDIA CorrDiff |
| **Flow Matching (FM)** | Solves Ordinary Differential Equations (ODE) learning deterministic paths between noise and data. | **Extremely Fast.** Requires far fewer steps than SDM. Maintains high-resolution spatial details (sharpness). | Mathematical trajectories can be harder to tune for highly chaotic local variables. | FlowCast-ODE, FLUX.1 |
| **Masked Diffusion (MDM)** | Joint sequence modeling using Transformers; masks and unmasks tokens across space *and* time. | **SOTA for Long-Range Stability.** Mitigates error accumulation. Much faster than SDM for sequences. | Less established for high-frequency micro-scale extremes (like hail) compared to subseasonal trends. | OmniCast, SeasonCast |
| **Autoregressive (AR)** | Predicts the next spatial token based on previous tokens (like a LLM for images/grids). | Excellent sequence-to-sequence reasoning and temporal consistency. | Computationally heavy for high-res 2D/3D grids without heavy latent compression. | NVIDIA Atlas (Latent AR + Diffusion) |
| **GANs (Generative Adversarial)** | Two networks competing to generate realistic data. | **Instant Inference.** Real-time generation (sub-second). | Prone to "mode collapse" (failing to capture the full probability distribution of rare events). | StyleGAN derivations (older downscaling models) |

### 2.2 SOTA Earth System Models (ESMs) & Hybrid Architectures

The most advanced models today are **Hybrids**, utilizing different architectures for different temporal and spatial scales.

*   **NVIDIA Earth-2 Architecture:**
    *   *Nowcasting (0-6h):* **StormScope** (Generative AI predicting radar/satellite).
    *   *Medium Range (15 days):* **Atlas** (Autoregressive Latent Diffusion Transformer) to step through time while maintaining sharp fronts.
    *   *Downscaling:* **CorrDiff** (Standard Diffusion) to add 2km stochastic textures (heavy rain, wind gusts) to coarse 25km grids.
*   **Deep Learning Earth System Model (DLESyM):**
    *   Focuses on extreme long-term climate stability (1000+ years without drift), crucial for slow-onset disasters (droughts/heatstress) but less relevant for flash events (hail).
*   **ICON / Traditional GCMs:**
    *   Moving towards 1km global resolution, but computationally too expensive for rapid ensemble disaster warning compared to AI emulators.

---

## 3. Architectural Recommendation for this Project

Given our goal is predicting specific, localized disasters (Hail, Floods, Heatstress, Landslides) over CONUS at a 2.5km/4km resolution:

**Recommended Path:** A **Hybrid Latent Diffusion / Flow Matching approach (similar to NVIDIA's Earth-2/CorrDiff philosophy)**.
1.  **Backbone (Latent/Temporal):** Use a DeepMind WeatherBench2 baseline (or a lightweight Autoregressive Transformer/GNN) to process the coarse spatio-temporal weather state.
2.  **Disaster Generation (Spatial Detail):** Use a **Flow Matching** or **Conditional Diffusion Model** at the final stage to predict the specific disaster probabilities. 
    *   *Why?* Standard GNNs blur out the extremes. Flow Matching will give us the sharp, high-fidelity local intensity required for Hail and Floods, but with much faster inference times than Standard Diffusion.