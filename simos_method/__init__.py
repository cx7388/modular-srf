"""
Determining the Weights of Criteria Using the Revised Simos' Procedure

Author: River Huang (river.huang@psi.ch)
Date: January 15, 2025

Overview:
This browser-based application implements multiple SRF configurations
for criteria weight elicitation using a modular architecture, as described
in the following paper:

Huang, R., Kadzinski, M., Figueira, J. R., Corrente, S., Siskos, E.,
and Burgherr, P. (2026). A Modular Simos-Roy-Figueira framework for
tailored weight elicitation in multi-criteria decision aiding.
Expert Systems With Applications, 311, 131315.
doi:10.1016/j.eswa.2026.131315

Features:
- Interactive web interface for arranging criteria cards
- Support for multiple SRF (Simos' Revised Framework) methods
- Real-time weight calculation and visualization
- Robust statistical analysis including ASI values and PCA

Installation and Usage:
1. Create a virtual environment and install the dependencies:
    python -m venv venv
    .\venv\Scripts\activate  # For Windows
    source venv/bin/activate  # For macOS/Linux
    pip install -r requirements.txt

2. Launch the Flask server using the following command:
    python -m flask --app simos_method run --port 8000
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, Response
from pathlib import Path
import os
import pandas as pd

from simos_method.static.python.srf_methods import calc_srf_flat, identify_inconsistency_recommendations
import simos_method.static.python.srf_methods as srf_methods_module
from simos_method.static.python.freeopt import warmup_solver_backend

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
DATA_DIR = Path(__file__).resolve().parent / 'static' / 'data'
SRF_SAMPLES_PATH = DATA_DIR / 'srf_samples.json'
PCA_OUTPUT_PATH = DATA_DIR / 'pca_output.json'

# Warm up solver backend once at startup to reduce first optimization latency.
if str(os.getenv('SRF_SKIP_SOLVER_WARMUP', '0')).strip().lower() not in {'1', 'true', 'yes'}:
    warmup_solver_backend()


def _serve_json_or_default(path_obj, default_payload):
    if path_obj.exists():
        return send_file(path_obj, mimetype='application/json', max_age=0)
    return Response(default_payload, mimetype='application/json')


def preprocess_dropzone(cards_arrangement):
    """
    Preprocesses the card arrangement data from the dropzone interface.
    
    Args:
        cards_arrangement (pd.DataFrame): Raw data from the frontend containing card positions
            and types. Expected columns: ['col', 'class', 'id']
            
    Returns:
        pd.DataFrame: Processed dataframe with:
            - Renumerated columns starting from 1
            - Rank assignments based on column positions
            - Index set to card IDs
    """
    # convert column numbers from string to integers
    cards_arrangement['col'] = cards_arrangement['col'].map(int)

    # Trim blank-card padding that may exist before the first criterion or after the
    # last criterion. Interior white cards must stay because they encode rank gaps.
    cards_arrangement = cards_arrangement.sort_values(by=['col'])
    start = (cards_arrangement["class"] != "white").values.argmax()
    end = len(cards_arrangement) - (cards_arrangement["class"][::-1] != "white").values.argmax()
    cards_arrangement = cards_arrangement.iloc[start:end]

    # set the column number as an index of the dataframe
    col_renumerate = {col_old: col_new for col_new, col_old in
                      enumerate(sorted(cards_arrangement.loc[:, 'col'].unique()), start=1)}
    cards_arrangement.replace({'col': col_renumerate}, inplace=True)
    cards_arrangement = cards_arrangement.set_index('col').sort_index()

    # Walk from right to left so each blank-card column inherits the rank of the
    # criterion block immediately to its right.
    cards_arrangement['rank'] = None
    rank = 0
    for col in range(cards_arrangement.index.max(), 0, -1):
        if 'criterion' in cards_arrangement.loc[[col], 'class'].to_list():
            rank += 1
        cards_arrangement.loc[col, 'rank'] = rank

    cards_arrangement.set_index('id', inplace=True)

    return cards_arrangement


def postprocess_dropzone(simos_calc_results):
    """
    Formats the calculation results for frontend display.
    
    Args:
        simos_calc_results (pd.DataFrame): Raw calculation results from SRF methods
            
    Returns:
        pd.DataFrame: Formatted results with:
            - Renamed columns for better readability
            - Added sum row
            - Formatted rank values as strings
    """
    simos_calc_results = simos_calc_results.copy()

    # Rename known output columns and keep any extra diagnostic columns.
    rename_map = {
        'r': 'Rank [r]',
        'name': 'Criteria',
        'k_i': 'Weights [%]',
        'k_center': 'Center weight [k_center]',
        'k_min': 'Min weight [k_min]',
        'k_max': 'Max weight [k_max]',
    }
    simos_calc_results.rename(columns=rename_map, inplace=True)

    preferred_order = [
        'Rank [r]',
        'Criteria',
        'Weights [%]',
        'Center weight [k_center]',
        'Min weight [k_min]',
        'Max weight [k_max]',
    ]
    ordered_columns = [col for col in preferred_order if col in simos_calc_results.columns]
    extra_columns = [col for col in simos_calc_results.columns if col not in ordered_columns]
    simos_calc_results = simos_calc_results[ordered_columns + extra_columns]

    # Sort criteria rows by rank then name (before appending summary row).
    sort_view = simos_calc_results.copy()
    if 'Rank [r]' in sort_view.columns:
        sort_view['_rank_sort_key'] = pd.to_numeric(sort_view['Rank [r]'], errors='coerce')
        sort_cols = ['_rank_sort_key']
        if 'Criteria' in sort_view.columns:
            sort_cols.append('Criteria')
        sort_view = sort_view.sort_values(by=sort_cols, na_position='last')
        sort_view.drop(columns=['_rank_sort_key'], inplace=True)
    elif 'Criteria' in sort_view.columns:
        sort_view = sort_view.sort_values(by=['Criteria'], na_position='last')

    # Add summary row with key totals.
    n_criteria = len(sort_view)
    sum_row = {col: '' for col in sort_view.columns}
    if 'Rank [r]' in sum_row:
        sum_row['Rank [r]'] = 'Sum'
    if 'Criteria' in sum_row:
        sum_row['Criteria'] = n_criteria
    if 'Weights [%]' in sort_view.columns:
        sum_row['Weights [%]'] = pd.to_numeric(
            sort_view['Weights [%]'], errors='coerce'
        ).sum()
    if 'Center weight [k_center]' in sort_view.columns:
        sum_row['Center weight [k_center]'] = pd.to_numeric(
            sort_view['Center weight [k_center]'], errors='coerce'
        ).sum()

    simos_calc_results = pd.concat(
        [sort_view, pd.DataFrame([sum_row], index=['Sum'])],
        axis=0
    )

    if 'Rank [r]' in simos_calc_results.columns:
        simos_calc_results['Rank [r]'] = simos_calc_results['Rank [r]'].astype(str)

    return simos_calc_results


@app.route('/')
def index():
    """Renders the landing page."""
    return render_template('index.html')


@app.route('/elicitation')
def elicitation():
    """Renders the SRF elicitation page."""
    return render_template('elicitation.html')


@app.route('/calculator')
def calculator_alias():
    """Backward-compatible alias for the elicitation page."""
    return redirect(url_for('elicitation'))


@app.route('/data/srf_samples.json')
@app.route('/static/data/srf_samples.json')
def data_srf_samples():
    """Serves SRF samples for boxplot/PCA with a safe fallback payload."""
    return _serve_json_or_default(SRF_SAMPLES_PATH, '[]')


@app.route('/data/pca_output.json')
@app.route('/static/data/pca_output.json')
def data_pca_output():
    """Serves PCA coordinates with a safe fallback payload."""
    return _serve_json_or_default(PCA_OUTPUT_PATH, '{}')


@app.route('/debug/hfl-config')
def debug_hfl_config():
    """Small runtime probe to verify which HFL mapper/ranges are loaded by the running server."""
    module_path = Path(getattr(srf_methods_module, '__file__', ''))
    return jsonify({
        'module_path': str(module_path),
        'module_exists': module_path.exists(),
        'module_mtime': module_path.stat().st_mtime if module_path.exists() else None,
        'has_old_mapper': hasattr(srf_methods_module, '_map_hfl_term'),
        'mappers': sorted([name for name in dir(srf_methods_module) if name.startswith('_map_hfl')]),
        'hfl_card_range': [
            getattr(srf_methods_module, 'HFL_CARD_MIN_TERM', None),
            getattr(srf_methods_module, 'HFL_CARD_MAX_TERM', None),
        ],
        'hfl_z_range': [
            getattr(srf_methods_module, 'HFL_Z_MIN_TERM', None),
            getattr(srf_methods_module, 'HFL_Z_MAX_TERM', None),
        ]
    })


@app.route('/calculate', methods=['POST'])
def calculate():
    """
    Endpoint for processing card arrangements and calculating criteria weights.
    
    Expected JSON payload:
        - zValue (float): Z-ratio parameter
        - e0Value (int): E0 parameter for SRF-II method
        - wValue (int): Decimal precision for weight normalization
        - srf_method (str): Selected SRF method variant
        - cards_arrangement (list): Card position and type data
        
    Returns:
        JSON containing:
            - crit_weights: Calculated criteria weights
            - asi_value: ASI (Average Stability Index) value
    """

    # Reset output files using absolute paths so static fetches are robust to cwd differences.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for json_path in [SRF_SAMPLES_PATH, PCA_OUTPUT_PATH]:
        if json_path.exists():
            json_path.unlink()
    # Keep placeholder files to avoid transient 404s on frontend fetch.
    SRF_SAMPLES_PATH.write_text('[]', encoding='utf-8')
    PCA_OUTPUT_PATH.write_text('{}', encoding='utf-8')

    try:
        # Convert user inputs into appropriate data types and structures
        data = request.json
        w_value = int(data.get('wValue', []))
        srf_method = data.get('srf_method', [])
        modular_options = data.get('modularOptions', {}) if srf_method == 'modular_srf' else None
        modular_profile = data.get('modularProfile', None) if srf_method == 'modular_srf' else None
        if srf_method == 'modular_srf':
            # Resolve the questionnaire answers again on the server so saved imports
            # and future UI changes still converge to one canonical configuration.
            modular_options_raw = modular_options if isinstance(modular_options, dict) else {}
            modular_options, modular_profile = srf_methods_module.resolve_modular_configuration(
                modular_options=modular_options,
                modular_profile=modular_profile
            )
            for passthrough_key in ('sampling_size', 'sample_size'):
                if passthrough_key in modular_options_raw:
                    modular_options[passthrough_key] = modular_options_raw.get(passthrough_key)
        method_for_parsing = modular_profile if srf_method == 'modular_srf' else srf_method
        inconsistency_suggestions_raw = data.get('inconsistencySuggestions', 3)
        try:
            inconsistency_suggestions = int(inconsistency_suggestions_raw)
        except (TypeError, ValueError):
            inconsistency_suggestions = 3
        inconsistency_suggestions = max(1, min(inconsistency_suggestions, 20))
        optional_constraints = data.get('optionalConstraints', None)
        if not isinstance(optional_constraints, dict):
            optional_constraints = None
        min_delta_raw = data.get('minDelta', 1.0)
        try:
            min_delta = float(min_delta_raw)
            if min_delta < 0:
                min_delta = 1.0
        except (TypeError, ValueError):
            min_delta = 1.0

        # z/e payloads arrive in different shapes depending on the selected method.
        # Normalize them here into the scalar/dict structures expected downstream.
        if srf_method == 'modular_srf':
            procedure = modular_options.get('procedure', 'standard')
            distance_type = modular_options.get('distance_type', 'precise')
            distance_format = modular_options.get('distance_format', 'interval')
            z_type = modular_options.get('z_type', 'precise')
            z_format = modular_options.get('z_format', 'interval')
            use_probability = (
                modular_options.get('probability', 'no') == 'yes'
                and procedure != 'direct'
                and (distance_type == 'imprecise' or (procedure == 'standard' and z_type == 'imprecise'))
            )

            if procedure == 'direct':
                z_value = {k: float(v) for k, v in data.get('zValue', {}).items()}
                e_value = 0
            elif procedure == 'zero':
                z_value = 1.0
                if distance_type == 'imprecise':
                    if use_probability:
                        e_value = {k: float(v) for k, v in data.get('eValue', {}).items()}
                    elif distance_format == 'fuzzy':
                        e_raw = data.get('eValue', {})
                        e_value = {k: int(v) for k, v in e_raw.items()} if isinstance(e_raw, dict) else {}
                    else:
                        e_raw = data.get('eValue', {})
                        e_value = {k: int(v) for k, v in e_raw.items()} if isinstance(e_raw, dict) else {}
                else:
                    e_value = int(data.get('eValue', 0))
            else:
                # standard deck
                if z_type == 'imprecise':
                    if use_probability:
                        z_value = {k: float(v) for k, v in data.get('zValue', {}).items()}
                    elif z_format == 'fuzzy':
                        z_raw = data.get('zValue', {})
                        if isinstance(z_raw, dict):
                            z_value = {k: int(v) for k, v in z_raw.items()}
                        else:
                            z_single = int(float(z_raw))
                            z_value = {'emin': z_single, 'emax': z_single}
                    else:
                        z_value = {k: float(v) for k, v in data.get('zValue', {}).items()}
                else:
                    z_value = float(data.get('zValue', []))

                if distance_type == 'imprecise':
                    if use_probability:
                        e_value = {k: float(v) for k, v in data.get('eValue', {}).items()}
                    elif distance_format == 'fuzzy':
                        e_raw = data.get('eValue', {})
                        e_value = {k: int(v) for k, v in e_raw.items()} if isinstance(e_raw, dict) else {}
                    else:
                        e_value = {k: int(v) for k, v in data.get('eValue', {}).items()}
                else:
                    e_value = 0
        else:
            # z values
            if method_for_parsing == 'wap':
                z_value = {k: float(v) for k, v in data.get('zValue', {}).items()}
            elif method_for_parsing == 'imprecise_srf':
                z_value = {k: float(v) for k, v in data.get('zValue', {}).items()}
            elif method_for_parsing == 'belief_degree_imprecise_srf':
                z_value = {k: float(v) for k, v in data.get('zValue', {}).items()}
            elif method_for_parsing == 'hfl_srf':
                z_raw = data.get('zValue', {})
                if isinstance(z_raw, dict):
                    z_value = {k: int(v) for k, v in z_raw.items()}
                else:
                    z_single = int(float(z_raw))
                    z_value = {'emin': z_single, 'emax': z_single}
            else:
                z_value = float(data.get('zValue', []))

            # e values
            if method_for_parsing == 'imprecise_srf':
                e_value = {k: int(v) for k, v in data.get('eValue', {}).items()}
            elif method_for_parsing == 'belief_degree_imprecise_srf':
                e_value = {k: float(v) for k, v in data.get('eValue', {}).items()}
            elif method_for_parsing == 'hfl_srf':
                e_raw = data.get('eValue', {})
                if isinstance(e_raw, dict):
                    e_value = {k: int(v) for k, v in e_raw.items()}
                else:
                    e_value = {}
            else:
                e_value = int(data.get('eValue', []))
        cards_arrangement = pd.DataFrame(data.get('cards_arrangement', []))
        required_columns = {'col', 'class', 'id'}
        if cards_arrangement.empty:
            raise ValueError('No cards were provided. Please arrange at least two criteria cards.')
        if not required_columns.issubset(set(cards_arrangement.columns)):
            raise ValueError('Invalid card arrangement payload. Please refresh the page and try again.')

        # Some variants encode spacing entirely in explicit z/e inputs, so white cards
        # in the drag-and-drop deck are only a UI aid and must not affect the model.
        modular_imprecise_distance = (
            srf_method == 'modular_srf'
            and modular_options.get('procedure') in {'standard', 'zero'}
            and modular_options.get('distance_type') == 'imprecise'
        )
        if method_for_parsing in {'wap', 'hfl_srf'} or modular_imprecise_distance:
            cards_arrangement = cards_arrangement[cards_arrangement['class'] != 'white'].copy()

        n_criteria = int((cards_arrangement['class'] == 'criterion').sum())
        if n_criteria < 2:
            raise ValueError('Please arrange at least two criteria cards before calculating.')

        # Convert the deck layout into rank-oriented data consumed by the solver layer.
        cards_arrangement = preprocess_dropzone(cards_arrangement)

        # Run the requested SRF pipeline. If the optimization is infeasible, try to
        # return targeted inconsistency suggestions instead of only a generic error.
        try:
            (simos_calc_results,
             asi_value) = calc_srf_flat(cards_arrangement,
                                        z_value,
                                        e_value,
                                        w_value,
                                        srf_method,
                                        modular_options=modular_options,
                                        modular_profile=modular_profile,
                                        min_delta=min_delta,
                                        extra_constraints=optional_constraints)
        except Exception as calc_exc:
            calc_message = str(calc_exc)
            is_solver_infeasibility = (
                'No optimal solution found' in calc_message
                or 'No feasible solution found' in calc_message
            )
            if is_solver_infeasibility:
                try:
                    inconsistency_e_value = e_value
                    report = identify_inconsistency_recommendations(
                        cards_arrangement,
                        z_value,
                        inconsistency_e_value,
                        srf_method=srf_method,
                        extra_constraints=optional_constraints,
                        max_suggestions=inconsistency_suggestions,
                        modular_options=modular_options,
                        modular_profile=modular_profile
                    )
                    if report.get('detected'):
                        return jsonify({
                            'crit_weights': '[]',
                            'asi_value': None,
                            'inconsistency': report
                        })
                except Exception:
                    # Keep the original calculation exception if EI analysis itself fails.
                    pass
            raise calc_exc

        # Reorder and label columns so the frontend can render one consistent table.
        simos_calc_results = postprocess_dropzone(simos_calc_results)

        return jsonify({
            'crit_weights': simos_calc_results.to_json(orient='records'),
            'asi_value': asi_value,
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


if __name__ == '__main__':
    app.run(debug=False, port=8000)
