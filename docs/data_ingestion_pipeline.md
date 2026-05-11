# Data Ingestion & Preprocessing Pipeline (Phase 2)

## 1. Overview
This document outlines the engineering pipeline to ingest, synchronize, and preprocess the dynamic weather inputs and static ground truth targets for the 2022-2023 training/validation window. All data will be projected onto a unified **2.5km CONUS grid**.

## 2. Target Grid Definition
*   **Domain:** Contiguous United States (CONUS).
*   **Resolution:** 2.5km x 2.5km.
*   **Projection:** TBD (likely EPSG:4326 or a CONUS-specific Albers Equal Area projection like EPSG:5070 for accurate spatial distance calculations).
*   **Temporal Resolution:** 6-hourly (00z, 06z, 12z, 18z) to match the standard output of WeatherNext 2.

## 3. Data Sources & Acquisition Strategy

### 3.1 Dynamic Weather Input (WeatherNext 2.0)
*   **Source:** Google Cloud Storage (`gs://weathernext/weathernext_2_0_0/zarr`).
*   **Access Method:** Requires access approval via the DeepMind WeatherNext form. Once approved, data will be accessed using `xarray` and `zarr` Python libraries.
*   **Processing:** 
    *   Extract 2022-2023 data.
    *   Downscale/Interpolate from native 0.25° (~25km) to the 2.5km CONUS grid.

### 3.2 Ground Truth: Hail (MRMS MESH)
*   **Source:** Iowa Environmental Mesonet (IEM) archive or AWS Open Data.
*   **Format:** GRIB2 or NetCDF.
*   **Native Resolution:** ~1km.
*   **Processing:** 
    *   Extract Maximum Expected Size of Hail (MESH).
    *   Upscale (Aggregate using Max or 90th percentile) to 2.5km grid.
    *   Temporal alignment: Aggregate 2-minute/hourly data to 6-hour windows (Max hail size within the 6h period).

### 3.3 Ground Truth: Heatstress (NWS HeatRisk)
*   **Source:** NWS Weather Prediction Center (WPC) Historical Archive.
*   **Format:** GeoTIFF or KML.
*   **Native Resolution:** 2.5km (NDFD grid).
*   **Processing:** 
    *   Already near target resolution. Reproject to exact CONUS grid.
    *   Temporal alignment: Map daily maximum risk scores to the corresponding 6-hour windows or treat as a daily static target.

### 3.4 Ground Truth: Landslides (USGS)
*   **Source:** USGS National Landslide Inventory (ScienceBase).
*   **Format:** Vector (Shapefiles/GeoJSON - Polygons and Points).
*   **Processing:** 
    *   Filter for events with `Date_Min` and `Date_Max` within 2022-2023.
    *   Rasterization: Burn polygons and points into the 2.5km grid (1 for occurrence, 0 for no occurrence).
    *   *Challenge:* Extreme class imbalance.

### 3.5 Ground Truth: Flooding (USGS Stream Gages / NOAA NWM)
*   **Source:** USGS Water Data for the Nation (NWIS) API / NOAA National Water Model.
*   **Format:** Time-series (Point data for gages).
*   **Processing:** 
    *   Fetch gage data for 2022-2023.
    *   Identify flood stage exceedances.
    *   Spatial Mapping: Map gage locations to their corresponding 2.5km grid cells.

### 3.6 Static Features (Topography, LULC)
*   **Sources:** USGS 3DEP (Elevation), NLCD (Land Cover).
*   **Processing:** Download once, rasterize/regrid to the 2.5km target grid. Extract slope, aspect, and land cover classes.

## 4. Software Stack
*   `xarray` / `zarr`: Handling large multi-dimensional weather arrays.
*   `rioxarray` / `rasterio`: Geospatial raster manipulation and regridding.
*   `geopandas`: Vector data handling (Landslides, Stream Gages).
*   `dask`: Parallel computing for out-of-core data processing.

## 5. Next Steps (Development)
1.  Set up the Python environment (`requirements.txt`).
2.  Write the base `GridManager` class to define the 2.5km CONUS grid.
3.  Implement individual downloader/preprocessor scripts for each dataset in a modular `/src/data/` directory.