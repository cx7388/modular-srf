# Determining the Weights of Criteria Using the Revised Simos' Procedure

**Author:** River Huang ([river.huang@psi.ch](mailto:river.huang@psi.ch))  
**Date:** January 15, 2026

## Overview
This browser-based application replicates the revised Simos' method for weight elicitation, as described in the following paper:

> Figueira, J., & Roy, B. (2002). Determining the weights of criteria in the ELECTRE type methods with a revised Simos' procedure. *European Journal of Operational Research, 139*(2), 317-326. [doi:10.1016/s0377-2217(01)00370-8](https://www.sciencedirect.com/science/article/pii/S0377221701003708)

It supports the predefined SRF family together with the standalone Modular SRF framework, including variability analysis with feasible-region sampling, extreme-scenario exploration, ASI diagnostics, PCA maps, and XLSX export of detailed variability tables.

## Installation and Usage
1. **Create a virtual environment** and install the dependencies:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # For Windows
    source venv/bin/activate  # For macOS/Linux
    pip install -r requirements.txt
    ```

2. **Launch the Flask server** using the following command:
    ```bash
    python -m flask --app simos_method run --port 8000
    ```

## Documentation
- Main manual: [USER_MANUAL.md](USER_MANUAL.md)
- Technical details: [USER_MANUAL.md#11-technical-details](USER_MANUAL.md#11-technical-details)

## Project Layout
- `simos_method/__init__.py`: Flask routes, request parsing, deck preprocessing, and JSON responses
- `simos_method/static/python/srf_methods.py`: core SRF calculations, variability analysis, and inconsistency detection
- `simos_method/static/python/utils.py`: rounding, ASI, and PCA helpers
- `simos_method/static/python/freeopt.py`: compatibility shim for the free optimization solver stack
- `simos_method/static/js/main.js`: page-level UI behavior and import/export helpers
- `simos_method/static/js/cardUtils.js`: drag-and-drop card logic
- `simos_method/static/js/uiUtils.js`: dynamic method inputs and modular questionnaire behavior
- `simos_method/static/js/backend.js`: frontend payload assembly and `/calculate` request handling
- `simos_method/static/js/results.js`: result table, sampling distribution, extreme-scenario heatmap, and PCA visualizations

## Development Notes
- The frontend serializes method-specific inputs in `simos_method/static/js/backend.js`, and the backend normalizes them again in `simos_method/__init__.py`. Keep both sides aligned when changing payload shapes.
- Core optimization behavior is centralized in `simos_method/static/python/srf_methods.py`; most method additions or constraint changes eventually pass through that file.
- Variability runs use a configurable sampling count (default `200`). Continuous models use hit-and-run sampling targeting the uniform distribution over the feasible region.
- Runtime plot/export/progress data is written to:
  - `simos_method/static/data/srf_samples.json`
  - `simos_method/static/data/srf_extreme_scenarios.json`
  - `simos_method/static/data/pca_output.json`
  - `simos_method/static/data/srf_export_payload.json`
  - `simos_method/static/data/calculation_progress.json`


