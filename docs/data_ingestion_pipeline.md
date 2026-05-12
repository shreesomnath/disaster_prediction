# Data Ingestion & Preprocessing Pipeline (Phase 2)

## 1. Overview & Data Structures
This document outlines the rigorous engineering pipeline required to ingest, temporally synchronize, and spatially harmonize dynamic weather inputs and static ground truth targets. The pipeline is designed specifically to feed into our **Hybrid Latent Flow Matching (HLFM)** architecture.

All data is projected onto a strict **2.5km CONUS grid (EPSG:5070 Albers Equal Area)**. Temporal synchronization forces all data into a standardized **6-hourly cadence (00z, 06z, 12z, 18z)** covering the 2022-2023 temporal window.

## 2. Dynamic Weather Input Features
Our primary predictive signals originate from Google DeepMind's WeatherNext 2.0 framework.

*   **Source:** Google Cloud Storage (`gs://weathernext/weathernext_2_0_0/zarr`).
*   **Data Structure:** Multi-dimensional Zarr arrays.
    *   Dimensions: `[time, pressure_level, latitude, longitude]`.
    *   Variables (Surface): `2m_temperature`, `10m_u_wind`, `10m_v_wind`, `surface_pressure`, `total_precipitation_6hr`.
    *   Variables (Upper Air - 500mb/850mb/250mb): `geopotential`, `temperature`, `specific_humidity`, `u_wind`, `v_wind`.
*   **Preprocessing Pipeline:**
    1.  **Temporal Slicing:** Isolate exactly 2022-01-01 to 2023-12-31.
    2.  **Spatial Regridding:** Employ `rioxarray` with bilinear interpolation to downscale the native 0.25° grid to our target 2.5km EPSG:5070 grid.
    3.  **Normalization:** Compute global mean and standard deviation for each channel over the 2022 training set to apply standard Z-score normalization, preserving variance for extreme outliers.

## 3. Ground Truth Target Acquisition & Structuring

The pipeline handles the specific intricacies and structural biases of our chosen ground truth datasets (see `research/detailed_ground_truth_and_algorithm.md` for rationale).

### 3.1 Hail: MRMS MESH (Maximum Expected Size of Hail)
*   **Target Tensor Shape:** `[time, 1, grid_y, grid_x]` (Continuous float).
*   **Pipeline:**
    1.  **Ingestion:** Retrieve GRIB2 files from IEM archives.
    2.  **Aggregation:** MRMS is hourly; we apply a `max()` reduction over the 6-hour window corresponding to the WeatherNext step (e.g., 00z-06z).
    3.  **Regridding:** Max-pooling upscaling from native ~1km to 2.5km.
    4.  **Handling:** Zeros dominate this dataset. Output is stored as sparse matrices (`scipy.sparse`) before batching to save memory.

### 3.2 Flooding: NOAA NWM + USGS Gages
*   **Target Tensor Shape:** `[time, 2, grid_y, grid_x]` (Channel 1: NWM Streamflow Continuous, Channel 2: USGS Exceedance Binary Mask).
*   **Pipeline (NWM Spatial):**
    1.  Extract 6-hourly routing variables.
    2.  Rasterize reach-level line strings onto the 2.5km grid.
*   **Pipeline (USGS Point):**
    1.  Query NWIS API for all CONUS gages.
    2.  Identify timestamps where `stage > minor_flood_stage`.
    3.  Map coordinates to the 2.5km index. Create a sparse binary mask (1 for flooded, 0 for unknown/no-flood).

### 3.3 Heatstress: NWS HeatRisk
*   **Target Tensor Shape:** `[time, 1, grid_y, grid_x]` (Categorical int: 0, 1, 2, 3, 4).
*   **Pipeline:**
    1.  **Ingestion:** GeoTIFFs from WPC archive.
    2.  **Temporal Mapping:** HeatRisk is daily. Replicate the daily maximum value across the four 6-hourly time steps of that day.
    3.  **Regridding:** Nearest-neighbor resampling (to preserve discrete integer classes) to 2.5km.

### 3.4 Landslides: USGS National Inventory
*   **Target Tensor Shape:** `[time, 1, grid_y, grid_x]` (Probabilistic float).
*   **Pipeline:**
    1.  **Ingestion:** Vector polygons/points from ScienceBase.
    2.  **Temporal Blurring:** Since landslide timestamps (`Date_Min`, `Date_Max`) are imprecise, we apply a 1D Gaussian temporal blur. If an event occurred between Jan 1 and Jan 3, the 6-hourly steps within that window receive a distributed probability between 0.1 and 1.0 rather than a hard binary flag.
    3.  **Rasterization:** Burn polygons to grid using fractional coverage.

## 4. Static Feature Ingestion
Static features provide the geographic context crucial for the HLFM's spatial predictions.

*   **Features:** Elevation (DEM), Slope, Aspect, Land Use Land Cover (LULC), Soil Type.
*   **Data Structure:** `[static_channels, grid_y, grid_x]`.
*   **Pipeline:**
    1.  Download USGS 3DEP (DEM) and NLCD (LULC).
    2.  Calculate topographic derivatives (Slope, Aspect) natively.
    3.  Regrid to 2.5km. LULC uses majority-vote resampling; Elevation uses bilinear interpolation.
    4.  Concatenate into a single static tensor file (`static_features.zarr`).

## 5. Output Data Architecture for Training
The final preprocessed dataset will be structured to maximize I/O throughput during GPU training:

```text
/data/processed/
  ├── static/
  │    └── static_features.zarr        (Shape: [C_static, H, W])
  ├── dynamic_inputs/
  │    └── weathernext_2022_2023.zarr  (Shape: [T, C_dyn, H, W])
  └── targets/
       ├── hail_mesh.zarr              (Shape: [T, 1, H, W])
       ├── flood_nwm.zarr              (Shape: [T, 2, H, W])
       ├── heatrisk.zarr               (Shape: [T, 1, H, W])
       └── landslides.zarr             (Shape: [T, 1, H, W])
```

## 6. Software Stack & Next Steps
*   **Core:** `xarray`, `zarr`, `dask` (Chunked parallel processing), `rioxarray`.
*   **Next Steps:**
    1. Implement the `DatasetBuilder` class which orchestrates parallel Dask workers to execute this pipeline chunk-by-chunk over the temporal dimension.
    2. Construct PyTorch `DataLoader` wrappers that efficiently sample time-aligned slices across all Zarr stores simultaneously for the HLFM model.