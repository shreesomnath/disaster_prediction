# AI Disaster Prediction: Research Phase Guidelines

## Project Context
We are developing a state-of-the-art AI model capable of generating hazard prediction maps for four disaster types: Hail, Flooding, Heatstress, and Landslides across the Contiguous United States (CONUS).

## Primary Mission: Rigorous Research
The core objective of this phase is not immediate implementation, but rigorous, documented research to define the optimal foundations for the AI model. 

## Core Tasks & Deliverables

1. **Ground Truth & Alternatives:** 
   - Investigate and identify the best possible ground truth datasets for each disaster type.
   - You must not assume the first option is fixed; find and evaluate alternative datasets, listing their pros, cons, and usability.
   - *Deliverable:* A comparative summary table detailing the selected ground truths, alternatives considered, primary reasons for selection, and mitigation strategies for any weaknesses.

2. **Model Architecture Selection:**
   - Evaluate state-of-the-art AI architectures (e.g., standard Diffusion, Flow Matching, Masked Diffusion, Autoregressive, Hybrid Latent models) for spatio-temporal disaster mapping on a 2.5km CONUS grid.
   - Contrast their capabilities specifically for high-resolution, extreme rare-event prediction, rather than just standard GNNs.

3. **Prior US-Based Products & Research:**
   - Identify existing operational products or mapping systems currently used in the US (e.g., NWS products, DeepMind's WeatherNext, NVIDIA Earth-2).
   - Explain how our proposed architecture compares to, utilizes, or improves upon these existing baselines.

4. **Supporting Publications:**
   - Provide concrete scientific references and foundational peer-reviewed papers that support both the chosen ground truth datasets and the architectural strategies.

## Workflow Execution Rules
1.  **Thoroughness:** Never settle for generic or surface-level information. Dive deep into the spatial resolution, temporal frequency, and inherent biases of all data and models.
2.  **Documentation:** All findings must be thoroughly documented in markdown format within the `research/` directory.
3.  **Flexibility:** Treat all selections as recommendations based on evidence, maintaining an extensible pipeline that can swap data sources if alternative validation proves better.
4.  **Version Control:** Commit all major research reports and findings to the remote repository continuously.