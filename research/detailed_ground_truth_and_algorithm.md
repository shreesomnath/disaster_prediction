# Detailed Research: Ground Truth Alternatives & Architectural Design

## 1. Ground Truth Alternatives & Selection Rationale

This section provides a rigorous comparative analysis of available ground truth datasets for each disaster type, justifying our final selections based on the requirements of our AI pipeline.

### 1.1 Hail Prediction Ground Truth
**Goal:** Map maximum hail size at a 2.5km resolution to predict severe storm impacts.

*   **Alternative A: SPC Storm Reports (Storm Prediction Center)**
    *   **Pros:** Actual human-verified ground observations. Measures exact impact.
    *   **Cons:** Extreme population bias (reports cluster near cities/roads). Daylight bias. Point data only, leaving massive spatial gaps.
    *   **Usability:** Poor for direct dense-grid pixel-to-pixel training due to sparsity. Good for independent validation.
*   **Alternative B: Satellite Proxy (GOES-16/17 GLM & IR)**
    *   **Pros:** Uniform, continuous spatial coverage. No radar beam blockage issues.
    *   **Cons:** Proxy only. Cannot measure surface hail size directly; infers it from cloud top cooling and lightning density.
    *   **Usability:** Better as an input feature than a definitive ground truth target.
*   **Alternative C: MRMS MESH (Multi-Radar Multi-Sensor Maximum Expected Size of Hail) [SELECTED]**
    *   **Pros:** High spatial resolution (~1km natively). Continuous gridded coverage across CONUS. Objective, algorithm-driven calculation merging multiple radars.
    *   **Cons:** Derived product based on radar reflectivity and freezing level algorithms; can overestimate in certain thermodynamic environments.
    *   **Usability:** **Exceptional.** It provides the continuous 2D spatial grid required for computer vision/spatial AI models (like Diffusion/Flow Matching) to learn spatial patterns.
    *   **Related Research:** Used extensively in ML severe weather forecasting (e.g., Gagne et al., 2019; McGovern et al., 2017).

### 1.2 Flooding Ground Truth
**Goal:** Identify high-risk inundation and flash flood events.

*   **Alternative A: Satellite SAR Inundation (e.g., Sentinel-1)**
    *   **Pros:** Direct, objective observation of surface water extent.
    *   **Cons:** Low temporal revisit time (6-12 days). Fails in urban environments (radar shadow) and under dense forest canopy. Cannot capture rapid flash floods.
    *   **Usability:** High value for post-event mapping, extremely low value for continuous 6-hourly temporal training.
*   **Alternative B: USGS Stream Gages [SELECTED as Primary Point Truth]**
    *   **Pros:** Highly accurate, direct physical measurement of stream stage and discharge. High temporal frequency (15-min to hourly).
    *   **Cons:** Point data only. Does not provide the 2D spatial extent of inundation. Misses un-gaged flash flood catchments.
    *   **Usability:** Excellent for validating threshold exceedance (flood stage).
*   **Alternative C: NOAA National Water Model (NWM) [SELECTED as Spatial Truth]**
    *   **Pros:** Gridded continuous streamflow, soil moisture, and runoff for the entire CONUS. Routes water through 2.7 million reaches.
    *   **Cons:** It is an assimilation model, not a direct observation. Subject to forcing errors.
    *   **Usability:** Provides the necessary 2D spatial grid.
    *   **Strategy:** We will use **NOAA NWM** for the dense spatial grid target, and apply **USGS Stream Gages** as a sparse, hard-anchor validation set to correct NWM biases.

### 1.3 Heatstress Ground Truth
**Goal:** Map the human impact and severity of heat waves, not just raw temperature.

*   **Alternative A: Raw Meteorological Variables (Apparent Temp / Heat Index from PRISM/RTMA)**
    *   **Pros:** Physically grounded, raw data.
    *   **Cons:** Ignores acclimatization. 90°F (32°C) is a standard summer day in Phoenix but a deadly heatwave in Seattle. Does not represent true "disaster" risk equally across geographies.
    *   **Usability:** Good input feature, poor target for generic "disaster" mapping.
*   **Alternative B: Land Surface Temperature (LST from MODIS/VIIRS)**
    *   **Pros:** Very high spatial resolution (1km). Directly captures Urban Heat Island (UHI) effects.
    *   **Cons:** Measures the ground "skin" temperature, not the 2-meter air temperature where humans experience heat. Heavily obscured by clouds.
*   **Alternative C: NWS HeatRisk [SELECTED]**
    *   **Pros:** Incorporates climatological context (how unusual the heat is for that specific location and time of year). Directly tied to CDC health impact data and mortality risk.
    *   **Cons:** Derived categorical product (0-4 scale), lacking continuous physical units.
    *   **Usability:** **Optimal.** It defines the *disaster* aspect by standardizing human risk across varying climates.

### 1.4 Landslides Ground Truth
**Goal:** Predict areas of high susceptibility to mass movement triggered by weather.

*   **Alternative A: NASA Global Landslide Catalog (GLC)**
    *   **Pros:** Global coverage, categorized by trigger (rainfall, earthquake).
    *   **Cons:** Highly dependent on media reports. Lower spatial precision than regional datasets.
    *   **Usability:** Good for global pre-training, poor for high-res CONUS targeting.
*   **Alternative B: InSAR Deformation Mapping**
    *   **Pros:** Direct measurement of millimeter-scale slope movement.
    *   **Cons:** Computationally massive to process at CONUS scale. Extremely low temporal frequency.
*   **Alternative C: USGS National Landslide Inventory [SELECTED]**
    *   **Pros:** Most comprehensive, verified database for the US.
    *   **Cons:** Massive reporting bias (events in remote areas are missed). Imprecise timestamps (Date_Min and Date_Max often span weeks). Severe class imbalance.
    *   **Usability:** **Sufficient but challenging.**
    *   **Mitigation Strategy:** We will utilize Spatial Stratified Negative Sampling to handle class imbalance and treat the target as a probabilistic susceptibility map rather than a deterministic binary occurrence map.

---

## 2. Summary Table of Ground Truth Selection

| Disaster Type | Selected Ground Truth | Alternative Considered | Primary Reason for Selection | Primary Weakness (Mitigation) |
| :--- | :--- | :--- | :--- | :--- |
| **Hail** | **MRMS MESH** | SPC Storm Reports | Provides continuous 2.5km spatial grid; no population bias. | Derived radar proxy. (Use SPC reports for final test-set validation). |
| **Flooding** | **NOAA NWM + USGS Gages** | SAR Satellite Inundation | High temporal frequency (6h) and continuous spatial routing. | NWM is a model, not observation. (Anchor with USGS Gage physical data). |
| **Heatstress**| **NWS HeatRisk** | Apparent Temp (RTMA) | Accounts for local climatological acclimatization and health impacts. | Categorical variable. (Use multi-class cross-entropy loss). |
| **Landslides**| **USGS Inventory** | NASA GLC | Highest spatial precision for CONUS; verified events. | Temporal imprecision. (Predict temporal susceptibility windows, not exact hour). |

---

## 3. Our Proposed Algorithm: Hybrid Latent Flow Matching (HLFM)

Standard Graph Neural Networks (GNNs) tend to smooth out extreme values, making them poor predictors for sharp, localized disasters like severe hail or flash floods. Standard Diffusion Models are excellent for detail but are computationally slow for operational 6-hourly rollouts.

**Our Solution:** A custom **Hybrid Latent Flow Matching (HLFM)** architecture.

### 3.1 Architectural Phases
1.  **Spatio-Temporal Context Encoder (The Backbone)**
    *   **Input:** Dynamic WeatherNext 2.0 tensors (6-hourly, multiple pressure levels) + Static topographical/LULC tensors.
    *   **Mechanism:** A Vision Transformer (ViT) or efficient 3D-UNet maps the coarse-resolution atmospheric dynamics into a compressed latent space. This captures the large-scale weather systems (fronts, atmospheric rivers).
2.  **Latent Flow Matching (The Generator)**
    *   **Mechanism:** Instead of simulating complex stochastic diffusion (SDEs), Flow Matching solves Ordinary Differential Equations (ODEs) to transport a simple base distribution (Gaussian noise) to the complex data distribution (the disaster ground truth map).
    *   **Advantage:** Flow Matching allows for "straight-path" deterministic generation. It maintains the high-fidelity, sharp spatial gradients necessary for mapping extreme boundaries (e.g., the edge of a hail swath) but requires **10x fewer sampling steps** than standard diffusion.
3.  **Multi-Hazard Decoding Heads**
    *   The model diverges into four distinct decoder heads.
    *   **Hail Head:** Zero-inflated continuous regression (predicts exact MESH size, biased towards predicting zeros for no-hail areas).
    *   **Flood Head:** High-resolution spatial regression constrained by the static DEM (Digital Elevation Model) to ensure water pools in valleys.
    *   **Heat Head:** Multi-class classification (predicting HeatRisk Categories 0-4).
    *   **Landslide Head:** Binary probabilistic classification mapping temporal susceptibility.

### 3.2 Usability & Implementation
*   **Loss Function:** We will employ a custom physics-informed loss function. For example, the flood head will incur heavy penalties if it predicts inundation on a steep slope without a corresponding downstream accumulation.
*   **Inference Speed:** By utilizing Flow Matching over standard Diffusion, we anticipate sub-minute generation times for CONUS 2.5km grid generation on an A100 GPU, enabling rapid operational ensemble forecasting.