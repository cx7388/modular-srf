# Modular SRF Weight Elicitation Tool

Browser-based software for criteria-weight elicitation with the standalone Modular SRF framework and the main predefined SRF variants.

**Author:** River Huang ([river.huang@psi.ch](mailto:river.huang@psi.ch))  
**Developed for:** Laboratory for Energy Systems Analysis (LEA), Paul Scherrer Institute (PSI)

## Overview
This project provides an interactive deck-of-cards interface for eliciting criteria weights in multi-criteria decision aiding. It supports:

- the standalone **Modular SRF** framework
- predefined SRF methods such as **SRF**, **SRF-II**, **Robust SRF**, **WAP**, **Imprecise SRF**, **Belief-degree Imprecise SRF**, and **HFL-SRF**
- variability analysis with **sampling distributions**, **extreme scenarios**, **ASI diagnostics**, and **PCA maps**
- export/import workflows for **JSON configurations** and **XLSX results**

## References
The current modular architecture is based on:

> Huang, R., Kadzinski, M., Figueira, J. R., Corrente, S., Siskos, E., and Burgherr, P. (2026). A Modular Simos-Roy-Figueira framework for tailored weight elicitation in multi-criteria decision aiding. *Expert Systems With Applications, 311*, 131315. https://doi.org/10.1016/j.eswa.2026.131315

## Key Features
- Interactive drag-and-drop card arrangement
- Modular questionnaire for assembling SRF configurations
- Support for precise, interval, probabilistic, and HFL-style inputs
- Optional robustness constraints such as minimum-weight and anti-dictatorship rules
- Configurable sampling count for variability analysis, with default `200`
- Uniform hit-and-run sampling for continuous feasible regions
- Dedicated extreme-scenario heatmap and PCA visualization
- Calculation progress reporting during heavier runs

## Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/cx7388/modular-srf.git
   cd modular-srf
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask app:
   ```bash
   python -m flask --app simos_method run --port 8000
   ```
5. Open:
   ```text
   http://localhost:8000
   ```

## Requirements
- Python 3.9 or newer
- A modern browser
- Local installation of the dependencies in [requirements.txt](requirements.txt)

## Documentation
- Main user guide: [USER_MANUAL.md](USER_MANUAL.md)
- Technical details: [USER_MANUAL.md#11-technical-details](USER_MANUAL.md#11-technical-details)

## Project Layout
- `simos_method/__init__.py`: Flask routes, request parsing, preprocessing, and JSON responses
- `simos_method/static/python/srf_methods.py`: SRF calculations, modular resolution, variability analysis, and inconsistency detection
- `simos_method/static/python/freeopt.py`: free-solver compatibility layer used instead of direct `gurobipy`
- `simos_method/static/python/utils.py`: rounding, ASI, and PCA helpers
- `simos_method/static/js/main.js`: import/export and general page behavior
- `simos_method/static/js/cardUtils.js`: drag-and-drop card logic
- `simos_method/static/js/uiUtils.js`: method-specific inputs and modular questionnaire behavior
- `simos_method/static/js/backend.js`: request assembly and `/calculate` handling
- `simos_method/static/js/results.js`: results table and Plotly visualizations

## Runtime Data Files
During calculations the app writes temporary JSON files under `simos_method/static/data/`:

- `srf_samples.json`
- `srf_extreme_scenarios.json`
- `pca_output.json`
- `srf_export_payload.json`
- `calculation_progress.json`

These files are regenerated as needed for frontend visualizations and export.

## Development Notes
- Keep frontend and backend payload handling aligned when changing inputs:
  - [backend.js](simos_method/static/js/backend.js)
  - [__init__.py](simos_method/__init__.py)
- Core solver/model behavior is centralized in [srf_methods.py](simos_method/static/python/srf_methods.py).
- The free solver shim in [freeopt.py](simos_method/static/python/freeopt.py) includes compatibility handling for different PuLP environments.

## License
See [LICENSE](LICENSE).
