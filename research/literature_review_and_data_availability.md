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

### 1.3 Recommended Overlap Window (Revised)

**The optimal, computationally feasible training/validation window is January 1, 2022 to December 31, 2023.**

*   *Why?* Google DeepMind's **WeatherNext 2.0** dataset (the recommended operational standard) only has a historical archive starting on **January 1, 2022**.
*   *Feasibility:* Training a spatio-temporal AI model on 8 years (2015-2023) of high-resolution 2.5km data across CONUS would require massive supercomputing clusters and months of runtime. Limiting the window to **2 years (2022-2023)** is far more realistic for development and compute budgets, while still providing enough seasonal cycles to capture extreme disaster events.
*   *Split Strategy:* We recommend using **2022 for Training** and **2023 for Validation/Testing**.

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

---

## 4. Scientific References & Foundational Papers

The architectural strategies outlined above are grounded in recent, peer-reviewed AI meteorology research:

1.  **GenCast (Standard Diffusion for Ensembles):**
    *   *Paper:* "GenCast: Diffusion-based ensemble forecasting for medium-range weather" (Nature, 2024).
    *   *Authors:* Price, I., Sanchez-Gonzalez, A., et al. (Google DeepMind).
    *   *Relevance:* Establishes conditional diffusion as the SOTA for probabilistic extreme event forecasting, outperforming traditional ECMWF ensembles. [arXiv:2312.15796](https://arxiv.org/abs/2312.15796)
2.  **CorrDiff (Diffusion Downscaling / Hybrid Approach):**
    *   *Paper:* "Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling" (Communications Earth & Environment, 2025).
    *   *Authors:* Mardani, M., Brenowitz, N., et al. (NVIDIA).
    *   *Relevance:* Validates the two-step architecture (Deterministic Mean + Stochastic Residual Diffusion) used in NVIDIA Earth-2 to generate 2km high-fidelity extremes (typhoons) from 25km inputs. [arXiv:2309.15214](https://arxiv.org/abs/2309.15214)
3.  **Flow Matching for Continuous/Fast Forecasting:**
    *   *Paper:* "FlowCast-ODE: Continuous Hourly Weather Forecasting with Dynamic Flow Matching and ODE Solver" (2025).
    *   *Relevance:* Demonstrates that deterministic Flow Matching (solving ODEs) can achieve diffusion-level sharpness for short-range local extremes (nowcasting) with significantly faster inference times. [arXiv:2509.14775](https://arxiv.org/abs/2509.14775)
4.  **OmniCast (Masked Diffusion for Long-Range Stability):**
    *   *Paper:* "OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales" (NeurIPS 2025).
    *   *Authors:* Nguyen, T., Gupta, J.K., et al.
    *   *Relevance:* Shows how Masked Diffusion Transformers eliminate the error accumulation seen in autoregressive rollouts, proving vital for long-range (subseasonal) prediction stability. [arXiv:2510.18707](https://arxiv.org/abs/2510.18707)