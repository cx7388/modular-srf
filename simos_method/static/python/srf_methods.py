from simos_method.static.python import freeopt as gp
import numpy as np
import pandas as pd
import random
import hashlib
import json
import re
from pathlib import Path
from scipy.linalg import svd
from scipy.optimize import linprog
from simos_method.static.python.freeopt import GRB

from simos_method.static.python.utils import round_up_selected, calc_asi, calc_pca

"""
Elicitation of criteria importance weights through Revised Simos' Methods

This module implements various variants of the Revised Simos' method for weight elicitation,
including the original SRF, SRF-II, Robust SRF, WAP, Imprecise SRF, Belief Degree Imprecise SRF, 
and HFL SRF. The implementation follows the modular framework described in the original paper by 
Huang et al. (2025).

The module provides two main functions:
- calc_srf_flat: Main entry point for weight calculations using any SRF variant
- calc_srf_modular: Core implementation of the modular SRF framework

It also includes functions for random sampling of criteria weights used in statistical analysis.
"""

HFL_CARD_MIN_TERM = 1
HFL_CARD_MAX_TERM = 5
HFL_Z_MIN_TERM = 1
HFL_Z_MAX_TERM = 10
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
SRF_SAMPLES_PATH = DATA_DIR / 'srf_samples.json'
SRF_EXTREME_SCENARIOS_PATH = DATA_DIR / 'srf_extreme_scenarios.json'
CALCULATION_PROGRESS_PATH = DATA_DIR / 'calculation_progress.json'
EXPORT_PAYLOAD_PATH = DATA_DIR / 'srf_export_payload.json'
DATA_DIR.mkdir(parents=True, exist_ok=True)
INCONSISTENCY_BIG_M = 10_000.0
INCONSISTENCY_CARDINALITY_WEIGHT = 1_000_000_000.0
# Admissible range of ratio inputs; matches the z input fields of the elicitation page.
INCONSISTENCY_Z_MIN = 1.1
INCONSISTENCY_Z_MAX = 1000.0
INCONSISTENCY_MAX_GAP_INCREASE = 30
INCONSISTENCY_MAX_E0_INCREASE = 100
INCONSISTENCY_MAX_COMBINED_CHANGE = 12
INCONSISTENCY_RANGE_SCAN_STEPS = 10
# Rounding every value of a rescaled z distribution to two decimals moves its
# expected value by at most 0.005; examples keep twice that distance to range ends.
INCONSISTENCY_DISTRIBUTION_ROUNDING_MARGIN = 0.01
# Search budget per request, so inconsistency analysis stays responsive.
INCONSISTENCY_MAX_CANDIDATE_SETS = 40
INCONSISTENCY_MAX_CHECKS_PER_SET = 100
INCONSISTENCY_MAX_FEASIBILITY_CHECKS = 600
MAX_USER_SAMPLE_SIZE = 20_000
DEFAULT_SAMPLING_SIZE = 200
MODULAR_ALLOWED_PROFILES = {
    'srf',
    'srf_ii',
    'wap',
    'imprecise_srf',
    'belief_degree_imprecise_srf',
    'hfl_srf',
}
MODULAR_DEFAULT_OPTIONS = {
    'procedure': 'standard',
    'distance_type': 'precise',
    'distance_format': 'interval',
    'z_type': 'precise',
    'z_format': 'interval',
    'probability': 'no',
    'output_type': 'single',
    'unit_weight': 'fixed',
    'variability_method': 'sampling',
}


def _write_json_payload(path_obj, payload, default_payload='{}'):
    """
    Writes JSON payloads atomically so the frontend can safely poll while a
    long calculation is still updating status files.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path_obj.with_suffix(f'{path_obj.suffix}.tmp')
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(',', ':')),
            encoding='utf-8'
        )
        temp_path.replace(path_obj)
    except Exception:
        path_obj.write_text(default_payload, encoding='utf-8')


def reset_calculation_progress(message='Preparing calculation...'):
    """
    Initializes the progress payload consumed by the frontend polling loop.
    """
    update_calculation_progress(
        stage='preparing',
        message=message,
        current=0,
        total=1,
        active=True,
        done=False,
        status='running'
    )


def update_calculation_progress(stage,
                                message,
                                current=None,
                                total=None,
                                active=True,
                                done=False,
                                status='running'):
    """
    Persists coarse-grained calculation progress for the active run.
    """
    percent = None
    if isinstance(current, (int, float)) and isinstance(total, (int, float)) and float(total) > 0:
        percent = max(0.0, min(100.0, round((float(current) / float(total)) * 100.0, 1)))

    payload = {
        'active': bool(active),
        'done': bool(done),
        'status': str(status),
        'stage': str(stage),
        'message': str(message),
        'current': None if current is None else float(current),
        'total': None if total is None else float(total),
        'percent': percent,
    }
    _write_json_payload(
        CALCULATION_PROGRESS_PATH,
        payload,
        default_payload='{"active":false,"done":false,"status":"idle","stage":"idle","message":"Idle","current":0,"total":0,"percent":0}'
    )


def finish_calculation_progress(message='Calculation complete.', status='completed'):
    """
    Marks the current calculation as finished.
    """
    update_calculation_progress(
        stage='finished',
        message=message,
        current=1,
        total=1,
        active=False,
        done=True,
        status=status
    )


def clear_variability_export_payload():
    """
    Resets the variability export payload so stale results are not downloaded.
    """
    EXPORT_PAYLOAD_PATH.write_text("{}", encoding='utf-8')


def clear_distribution_samples():
    """
    Resets the saved sampling cloud used by the variability figures.
    """
    SRF_SAMPLES_PATH.write_text("[]", encoding='utf-8')


def clear_extreme_scenarios():
    """
    Resets the saved extreme-scenario table used by the dedicated figure/export.
    """
    SRF_EXTREME_SCENARIOS_PATH.write_text("[]", encoding='utf-8')


def _coerce_sample_size(raw_value):
    """
    Parses an optional user-provided sample count.
    """
    try:
        parsed_value = int(raw_value)
    except (TypeError, ValueError):
        return None

    if parsed_value <= 0:
        return None

    return min(parsed_value, MAX_USER_SAMPLE_SIZE)


def _should_emit_progress(current, total, target_updates=24):
    """
    Throttles status-file writes so progress remains responsive without
    overwhelming the filesystem.
    """
    if not isinstance(current, (int, float)) or not isinstance(total, (int, float)) or total <= 0:
        return True
    interval = max(1, int(total) // max(1, target_updates))
    return int(current) in {0, 1, int(total)} or (int(current) % interval == 0)


def _rename_solution_columns(cards_arrangement, values_df):
    """
    Renames criterion-id columns to user-facing criterion names when possible.
    """
    export_df = values_df.copy()
    try:
        export_df = export_df.rename(columns=cards_arrangement['name'])
    except Exception:
        pass
    return export_df


def _format_sampling_export_rows(export_df):
    labeled_df = export_df.copy().reset_index(drop=True)
    labeled_df.insert(0, 'Sample', [f'Sample {idx + 1}' for idx in range(len(labeled_df))])
    return labeled_df


def _format_extreme_export_rows(export_df):
    scenario_labels = []
    for idx, raw_label in enumerate(export_df.index, start=1):
        label = str(raw_label).strip()
        if label.startswith('vertex_'):
            suffix = label.split('_')[-1]
            if suffix.isdigit():
                label = f'Vertex {int(suffix) + 1}'
        if not label:
            label = f'Scenario {idx}'
        scenario_labels.append(label)

    labeled_df = export_df.copy().reset_index(drop=True)
    labeled_df.insert(0, 'Scenario', scenario_labels)
    return labeled_df


def _build_variability_export_section(cards_arrangement, values_df, mode, source):
    """
    Builds one labeled XLSX-export section for variability details.
    """
    if not (isinstance(values_df, pd.DataFrame) and not values_df.empty):
        return None

    renamed_df = _rename_solution_columns(cards_arrangement, values_df)
    if mode == 'extreme':
        labeled_df = _format_extreme_export_rows(renamed_df)
        sheet_name = 'Extreme Scenarios'
    else:
        labeled_df = _format_sampling_export_rows(renamed_df)
        sheet_name = 'Sampling Results'

    return {
        'mode': mode,
        'source': source,
        'sheet_name': sheet_name,
        'records': labeled_df.to_dict(orient='records'),
    }


def _persist_variability_export_payload(cards_arrangement,
                                        sampling_df=None,
                                        extreme_df=None,
                                        sampling_source='samples',
                                        extreme_source='min_max'):
    """
    Persists both variability detail tables used by XLSX export.
    """
    payload = {}

    sampling_section = _build_variability_export_section(
        cards_arrangement,
        sampling_df,
        mode='sampling',
        source=sampling_source
    )
    if sampling_section is not None:
        payload['sampling_results'] = sampling_section

    extreme_section = _build_variability_export_section(
        cards_arrangement,
        extreme_df,
        mode='extreme',
        source=extreme_source
    )
    if extreme_section is not None:
        payload['extreme_scenarios'] = extreme_section

    if not payload:
        clear_variability_export_payload()
        return

    _write_json_payload(
        EXPORT_PAYLOAD_PATH,
        payload
    )


def _persist_extreme_scenarios(cards_arrangement, values_df):
    """
    Persists labeled extreme scenarios for the dedicated frontend figure.
    """
    if not (isinstance(values_df, pd.DataFrame) and not values_df.empty):
        clear_extreme_scenarios()
        return

    renamed_df = _rename_solution_columns(cards_arrangement, values_df)
    labeled_df = _format_extreme_export_rows(renamed_df)
    labeled_df.to_json(str(SRF_EXTREME_SCENARIOS_PATH), orient='records')


def _resolve_sample_budget(raw_sample_size):
    """
    Returns the requested sample count or the application default.
    """
    return raw_sample_size if isinstance(raw_sample_size, int) else DEFAULT_SAMPLING_SIZE


def _normalize_modular_options(modular_options):
    """
    Normalizes questionnaire answers for the modular SRF configuration.
    """
    normalized = dict(MODULAR_DEFAULT_OPTIONS)
    if isinstance(modular_options, dict):
        for key in MODULAR_DEFAULT_OPTIONS:
            if key in modular_options and modular_options[key] is not None:
                normalized[key] = str(modular_options[key]).strip() or MODULAR_DEFAULT_OPTIONS[key]

    if normalized['procedure'] not in {'standard', 'zero', 'direct'}:
        normalized['procedure'] = MODULAR_DEFAULT_OPTIONS['procedure']
    if normalized['distance_type'] not in {'precise', 'imprecise'}:
        normalized['distance_type'] = MODULAR_DEFAULT_OPTIONS['distance_type']
    if normalized['distance_format'] not in {'interval', 'fuzzy'}:
        normalized['distance_format'] = MODULAR_DEFAULT_OPTIONS['distance_format']
    if normalized['z_type'] not in {'precise', 'imprecise', 'na'}:
        normalized['z_type'] = MODULAR_DEFAULT_OPTIONS['z_type']
    if normalized['z_format'] not in {'interval', 'fuzzy'}:
        normalized['z_format'] = MODULAR_DEFAULT_OPTIONS['z_format']
    if normalized['probability'] not in {'no', 'yes'}:
        normalized['probability'] = MODULAR_DEFAULT_OPTIONS['probability']
    if normalized['output_type'] not in {'single', 'variability'}:
        normalized['output_type'] = MODULAR_DEFAULT_OPTIONS['output_type']
    if normalized['unit_weight'] not in {'fixed', 'dynamic'}:
        normalized['unit_weight'] = MODULAR_DEFAULT_OPTIONS['unit_weight']
    if normalized['variability_method'] not in {'sampling', 'extreme'}:
        normalized['variability_method'] = MODULAR_DEFAULT_OPTIONS['variability_method']

    # Q3(b) and Q3(c) do not use global z (handled by SRF-II and WAP logic, respectively).
    if normalized['procedure'] in {'zero', 'direct'}:
        normalized['z_type'] = 'na'
    elif normalized['z_type'] == 'na':
        # In standard deck, global z is always required.
        normalized['z_type'] = 'precise'
    if normalized['procedure'] == 'direct':
        normalized['probability'] = 'no'
        normalized['unit_weight'] = 'fixed'

    # Probability is available only for interval-based imprecision.
    has_imprecise_distance = normalized['distance_type'] == 'imprecise'
    has_imprecise_z = (
        normalized['procedure'] == 'standard'
        and normalized['z_type'] == 'imprecise'
    )
    all_imprecise_are_interval = (
        (not has_imprecise_distance or normalized['distance_format'] == 'interval')
        and (not has_imprecise_z or normalized['z_format'] == 'interval')
    )
    if not ((has_imprecise_distance or has_imprecise_z) and all_imprecise_are_interval):
        normalized['probability'] = 'no'

    # If probability is enabled in standard procedure, z is also probabilistic.
    if normalized['procedure'] == 'standard' and normalized['probability'] == 'yes':
        normalized['z_type'] = 'imprecise'
        normalized['z_format'] = 'interval'

    if normalized['distance_type'] != 'imprecise':
        normalized['distance_format'] = 'interval'
    if normalized['z_type'] != 'imprecise':
        normalized['z_format'] = 'interval'
    if normalized['output_type'] == 'variability':
        # Dynamic analysis now always produces both sample clouds and extreme scenarios.
        normalized['variability_method'] = 'sampling'
    else:
        normalized['variability_method'] = 'sampling'

    return normalized


def resolve_modular_configuration(modular_options=None, modular_profile=None):
    """
    Resolves modular questionnaire answers into a normalized option dict and
    an effective implemented SRF profile.
    """
    options = _normalize_modular_options(modular_options)

    if isinstance(modular_options, dict) and len(modular_options) > 0:
        # Collapse the questionnaire answers to the closest implemented profile so
        # parsing and model-building can reuse the classical SRF branches.
        has_imprecise_distance = options['distance_type'] == 'imprecise'
        has_imprecise_z = options['procedure'] == 'standard' and options['z_type'] == 'imprecise'

        if options['procedure'] == 'direct':
            profile = 'wap'
        elif options['procedure'] == 'zero':
            profile = 'srf_ii'
        elif (has_imprecise_distance or has_imprecise_z) and options['probability'] == 'yes':
            profile = 'belief_degree_imprecise_srf'
        elif ((has_imprecise_distance and options['distance_format'] == 'fuzzy')
              or (has_imprecise_z and options['z_format'] == 'fuzzy')):
            profile = 'hfl_srf'
        elif has_imprecise_distance or has_imprecise_z:
            profile = 'imprecise_srf'
        else:
            profile = 'srf'
    else:
        profile_candidate = str(modular_profile).strip() if modular_profile is not None else ''
        profile = profile_candidate if profile_candidate in MODULAR_ALLOWED_PROFILES else 'srf'

    return options, profile


def _resolve_modular_classical_equivalent(options):
    """
    Debug helper: returns the classical SRF variant that should be behaviorally
    equivalent to the given modular configuration, or None when no strict
    equivalence mapping applies.

    This mapping is intentionally not used in the production calculation path.
    It is meant for parity checks only.
    """
    if not isinstance(options, dict):
        return None

    procedure = options.get('procedure', 'standard')
    distance_type = options.get('distance_type', 'precise')
    distance_format = options.get('distance_format', 'interval')
    z_type = options.get('z_type', 'precise')
    z_format = options.get('z_format', 'interval')
    probability = options.get('probability', 'no')
    output_type = options.get('output_type', 'single')
    unit_weight = options.get('unit_weight', 'fixed')
    variability_method = options.get('variability_method', 'sampling')

    # 1) Modular -> SRF
    if (
        procedure == 'standard'
        and distance_type == 'precise'
        and z_type == 'precise'
        and output_type == 'single'
        and unit_weight == 'fixed'
    ):
        return 'srf'

    # 2) Modular -> SRF-II
    if (
        procedure == 'zero'
        and distance_type == 'precise'
        and output_type == 'single'
        and unit_weight == 'fixed'
    ):
        return 'srf_ii'

    # 3) Modular -> Robust SRF
    if (
        procedure == 'standard'
        and distance_type == 'precise'
        and z_type == 'precise'
        and output_type == 'variability'
        and unit_weight == 'dynamic'
        and variability_method == 'sampling'
    ):
        return 'robust_srf'

    # 4) Modular -> WAP
    if (
        procedure == 'direct'
        and distance_type == 'imprecise'
        and output_type == 'variability'
        and variability_method == 'sampling'
    ):
        return 'wap'

    # 5) Modular -> Imprecise SRF (interval, no probability)
    if (
        procedure == 'standard'
        and distance_type == 'imprecise'
        and distance_format == 'interval'
        and z_type == 'imprecise'
        and z_format == 'interval'
        and probability == 'no'
        and output_type == 'variability'
        and unit_weight == 'fixed'
        and variability_method == 'sampling'
    ):
        return 'imprecise_srf'

    # 6) Modular -> Belief-degree Imprecise SRF
    if (
        procedure == 'standard'
        and distance_type == 'imprecise'
        and distance_format == 'interval'
        and z_type == 'imprecise'
        and z_format == 'interval'
        and probability == 'yes'
        and output_type == 'variability'
        and unit_weight == 'fixed'
        and variability_method == 'sampling'
    ):
        return 'belief_degree_imprecise_srf'

    # 7) Modular -> HFL-SRF
    if (
        procedure == 'standard'
        and distance_type == 'imprecise'
        and distance_format == 'fuzzy'
        and z_type == 'imprecise'
        and z_format == 'fuzzy'
        and output_type == 'variability'
        and unit_weight == 'fixed'
        and variability_method == 'sampling'
    ):
        return 'hfl_srf'

    return None


def _is_modular_robust_equivalent(options):
    """
    Returns True when modular options correspond to the robust SRF-equivalent
    setting:
    - standard procedure
    - precise distance
    - precise global ratio
    - variability output with sampling
    - dynamic unit-weight policy
    """
    if not isinstance(options, dict):
        return False

    return bool(
        options.get('procedure') == 'standard'
        and options.get('distance_type') == 'precise'
        and options.get('z_type') == 'precise'
        and options.get('output_type') == 'variability'
        and options.get('variability_method') == 'sampling'
        and options.get('unit_weight') == 'dynamic'
    )


def _is_modular_wap_equivalent(options):
    """
    Returns True when modular options correspond to the WAP-equivalent setting:
    - direct-ratio procedure
    - imprecise local ratio input
    - variability output with sampling
    """
    if not isinstance(options, dict):
        return False

    return bool(
        options.get('procedure') == 'direct'
        and options.get('distance_type') == 'imprecise'
        and options.get('output_type') == 'variability'
        and options.get('variability_method') == 'sampling'
    )


def _derive_reproducible_sampling_seed(cards_arrangement,
                                       z_value,
                                       e_value,
                                       w_value,
                                       min_delta,
                                       comp_rule_within,
                                       comp_rule_successive,
                                       ratio_mode,
                                       normalized,
                                       extra_cond,
                                       sample_size_hint=None):
    """
    Builds a deterministic 32-bit seed from effective model inputs/structure.
    This keeps stochastic robustness outputs reproducible and comparable across
    equivalent classical/modular configurations.
    """
    cards_json = cards_arrangement.sort_index().to_json(orient='split')
    payload = {
        'cards': cards_json,
        'z_value': z_value,
        'e_value': e_value,
        'w_value': w_value,
        'min_delta': float(min_delta),
        'comp_rule_within': comp_rule_within,
        'comp_rule_successive': comp_rule_successive,
        'ratio_mode': ratio_mode,
        'normalized': bool(normalized),
        'extra_cond': extra_cond if isinstance(extra_cond, dict) else None,
        'sample_size_hint': sample_size_hint,
    }
    payload_json = json.dumps(payload, sort_keys=True, default=str, separators=(',', ':'))
    digest_hex = hashlib.sha256(payload_json.encode('utf-8')).hexdigest()
    return int(digest_hex[:8], 16)


def debug_check_modular_classical_equivalence(cards_arrangement,
                                              z_value,
                                              e_value,
                                              w_value,
                                              modular_options=None,
                                              modular_profile=None,
                                              min_delta=1.0,
                                              extra_constraints=None,
                                              random_seed=12345,
                                              numpy_seed=12345):
    """
    Debug utility that compares modular SRF output against its expected
    classical equivalent for configurations where strict parity is expected.

    Returns a dictionary with max absolute differences for overlapping numeric
    output columns and ASI.
    """
    options, _effective_profile = resolve_modular_configuration(
        modular_options=modular_options,
        modular_profile=modular_profile
    )
    classical_method = _resolve_modular_classical_equivalent(options)
    if classical_method is None:
        raise ValueError(
            "No strict classical-equivalence mapping is defined for the provided modular configuration."
        )

    py_state = random.getstate()
    np_state = np.random.get_state()

    try:
        if random_seed is not None:
            random.seed(random_seed)
        if numpy_seed is not None:
            np.random.seed(numpy_seed)
        classical_df, classical_asi = calc_srf_flat(
            cards_arrangement.copy(),
            z_value,
            e_value,
            w_value,
            classical_method,
            modular_options=None,
            modular_profile=None,
            min_delta=min_delta,
            extra_constraints=extra_constraints
        )

        if random_seed is not None:
            random.seed(random_seed)
        if numpy_seed is not None:
            np.random.seed(numpy_seed)
        modular_df, modular_asi = calc_srf_flat(
            cards_arrangement.copy(),
            z_value,
            e_value,
            w_value,
            'modular_srf',
            modular_options=options,
            modular_profile=modular_profile,
            min_delta=min_delta,
            extra_constraints=extra_constraints
        )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

    shared_cols = [col for col in classical_df.columns if col in modular_df.columns]
    numeric_cols = [col for col in shared_cols if col not in {'name'}]
    column_max_abs_diff = {}
    max_abs_weight_diff = 0.0

    for col in numeric_cols:
        classical_vals = pd.to_numeric(classical_df[col], errors='coerce')
        modular_vals = pd.to_numeric(modular_df[col], errors='coerce')
        if not (classical_vals.notna().any() or modular_vals.notna().any()):
            continue
        diff_val = (classical_vals - modular_vals).abs().max()
        if pd.isna(diff_val):
            continue
        diff_float = float(diff_val)
        column_max_abs_diff[col] = diff_float
        if col.startswith('k_') or col in {'k_i', 'r'}:
            max_abs_weight_diff = max(max_abs_weight_diff, diff_float)

    if classical_asi is None and modular_asi is None:
        asi_abs_diff = 0.0
    elif classical_asi is not None and modular_asi is not None:
        asi_abs_diff = abs(float(classical_asi) - float(modular_asi))
    else:
        asi_abs_diff = None

    return {
        'classical_method': classical_method,
        'max_abs_weight_diff': max_abs_weight_diff,
        'asi_abs_diff': asi_abs_diff,
        'column_max_abs_diff': column_max_abs_diff,
        'modular_options': options,
    }


def _resolve_modular_structure(options):
    """
    Resolves modular options into explicit SRF component choices.
    """
    # These low-level flags describe how the optimization model should behave.
    # They intentionally stay independent from UI labels or classical method names.
    procedure = options.get('procedure', 'standard')
    distance_type = options.get('distance_type', 'precise')
    distance_format = options.get('distance_format', 'interval')
    z_type = options.get('z_type', 'precise')
    z_format = options.get('z_format', 'interval')
    use_probability = (
        options.get('probability', 'no') == 'yes'
        and (distance_type == 'imprecise' or z_type == 'imprecise')
    )

    comp_rule_within = 'equal'
    normalized = True
    srf_objective = None

    if procedure == 'direct':
        comp_rule_successive = 'fully-flexible'
        ratio_mode = 'interval-successive'
    elif procedure == 'zero':
        if distance_type == 'imprecise':
            if use_probability:
                comp_rule_successive = 'probability-distribution'
            elif distance_format == 'fuzzy':
                comp_rule_successive = 'hfl-linguistic-interval'
            else:
                comp_rule_successive = 'interval-constrained'
        else:
            comp_rule_successive = 'fixed-spacing'
        ratio_mode = 'linear-spacing'
    else:
        # standard deck
        if distance_type == 'imprecise':
            if use_probability:
                comp_rule_successive = 'probability-distribution'
            elif distance_format == 'fuzzy':
                comp_rule_successive = 'hfl-linguistic-interval'
            else:
                comp_rule_successive = 'interval-constrained'
        else:
            comp_rule_successive = 'fixed-spacing'

        if z_type == 'imprecise':
            if use_probability:
                ratio_mode = 'probability-cloud'
            elif z_format == 'fuzzy':
                ratio_mode = 'hfl-ratio-interval'
            else:
                ratio_mode = 'interval-total'
        else:
            ratio_mode = 'exact-ratio'

    return srf_objective, comp_rule_within, comp_rule_successive, ratio_mode, normalized


def _map_hfl_card_term(term_value):
    """
    Maps an HFL linguistic term index for successive rank-gap cards.
    Allowed domain is [1, 5].
    """
    alpha = int(term_value)
    if alpha < HFL_CARD_MIN_TERM or alpha > HFL_CARD_MAX_TERM:
        raise ValueError(
            f"Invalid HFL gap term {alpha}. Allowed range is [{HFL_CARD_MIN_TERM}, {HFL_CARD_MAX_TERM}]."
        )
    return alpha


def _map_hfl_z_term(term_value):
    """
    Maps an HFL linguistic term index for global z contrast.
    Allowed domain is [1, 10].
    """
    alpha = int(term_value)
    if alpha < HFL_Z_MIN_TERM or alpha > HFL_Z_MAX_TERM:
        raise ValueError(
            f"Invalid HFL z term {alpha}. Allowed range is [{HFL_Z_MIN_TERM}, {HFL_Z_MAX_TERM}]."
        )
    return alpha


def _extract_probability_pairs(input_values, value_prefix, beta_prefix):
    """
    Extracts {(value, probability)} pairs from flat form-data dictionaries.
    """
    cloud = {}
    for key, value in input_values.items():
        if not key.startswith(value_prefix):
            continue
        suffix = key[len(value_prefix):]
        beta_key = f"{beta_prefix}{suffix}"
        if beta_key not in input_values:
            continue
        v = float(value)
        p = float(input_values[beta_key])
        cloud[v] = cloud.get(v, 0.0) + p
    return cloud


def _normalize_probability_cloud(cloud):
    """
    Normalizes a probability cloud and removes zero/negative probabilities.
    """
    cleaned = {float(v): float(p) for v, p in cloud.items() if float(p) > 0}
    total_prob = float(sum(cleaned.values()))
    if total_prob <= 0:
        raise ValueError("Probability cloud must contain positive probabilities.")
    return {v: p / total_prob for v, p in cleaned.items()}


def _probability_cloud_stats(cloud):
    """
    Returns normalized cloud, support bounds, and expected value.
    """
    normalized = _normalize_probability_cloud(cloud)
    values = np.array(list(normalized.keys()), dtype=float)
    probs = np.array(list(normalized.values()), dtype=float)
    return normalized, float(values.min()), float(values.max()), float(np.dot(values, probs))


def _build_belief_expected_inputs(cards_arrangement, z_value, e_value):
    """
    Combines belief-degree probability inputs into expected values for the central solution.
    """
    criteria_cards = cards_arrangement[cards_arrangement['class'] == 'criterion'].sort_values('rank')

    rank_white_count = {}
    for rank in cards_arrangement['rank'].unique():
        rank_white_count[rank] = cards_arrangement[cards_arrangement['rank'] == rank]['class'].to_list().count('white') + 1

    rank_groups = {}
    for rank in criteria_cards['rank'].unique():
        rank_groups[rank] = criteria_cards[criteria_cards['rank'] == rank].index.tolist()
    sorted_ranks = sorted(rank_groups.keys())

    expected_e = {}
    for i in range(1, len(sorted_ranks)):
        prev_rank = sorted_ranks[i - 1]
        if rank_white_count.get(prev_rank, 1) <= 1:
            continue

        cloud = _extract_probability_pairs(
            e_value,
            value_prefix=f"e-value-{prev_rank}-",
            beta_prefix=f"e-beta-{prev_rank}-",
        )
        if not cloud:
            cloud = {float(rank_white_count[prev_rank] - 1): 1.0}
        _, _, _, expected_gap = _probability_cloud_stats(cloud)
        expected_e[f'emin_{prev_rank}'] = expected_gap
        expected_e[f'emax_{prev_rank}'] = expected_gap

    z_cloud = _extract_probability_pairs(
        z_value,
        value_prefix='z-value-',
        beta_prefix='z-beta-',
    )
    if not z_cloud:
        raise ValueError("No valid (z, beta) pairs were provided for belief-degree SRF.")
    _, _, _, expected_z = _probability_cloud_stats(z_cloud)
    expected_z_dict = {'zmin': expected_z, 'zmax': expected_z}

    return expected_z_dict, expected_e


def _is_extra_constraints_enabled(extra_cond):
    """
    Returns True if optional extra constraints are enabled by the user.
    """
    if not isinstance(extra_cond, dict):
        return False

    dictatorship_req = extra_cond.get('dictatorship', {})
    min_weight_req = extra_cond.get('minimum_weight', {})

    return bool(
        isinstance(dictatorship_req, dict) and dictatorship_req.get('enabled')
        or isinstance(min_weight_req, dict) and min_weight_req.get('enabled')
    )


def _add_optional_extra_constraints(model, weights, criteria_cards, extra_cond, name_prefix="extra"):
    """
    Adds optional extra constraints:
      - minimum-weight requirement for all criteria
      - anti-dictatorship requirement (automatic for all criteria)
    `name_prefix` keeps constraint names unique when a model holds several weight vectors.
    """
    if not isinstance(extra_cond, dict):
        return

    # Minimum-weight requirement (all criteria)
    min_weight_req = extra_cond.get('minimum_weight', {})
    if isinstance(min_weight_req, dict) and min_weight_req.get('enabled'):
        min_weight_value = float(min_weight_req.get('value', 0.0))
        if min_weight_value < 0:
            raise ValueError("Minimum weight requirement must be non-negative.")

        n_criteria = len(criteria_cards.index)
        if n_criteria * min_weight_value > 100 + 1e-9:
            raise ValueError(
                "Minimum weight requirement is infeasible: "
                "sum of lower bounds exceeds 100."
            )

        for idx in criteria_cards.index:
            model.addConstr(
                weights[idx] >= min_weight_value,
                f"{name_prefix}_min_weight_{idx}"
            )

    # Anti-dictatorship requirement:
    # each criterion weight cannot exceed the sum of all remaining criteria weights.
    dictatorship_req = extra_cond.get('dictatorship', {})
    if isinstance(dictatorship_req, dict) and dictatorship_req.get('enabled'):
        n_criteria = len(criteria_cards.index)
        if n_criteria < 2:
            raise ValueError("Anti-dictatorship requirement needs at least two criteria.")

        total_weight = gp.quicksum(weights[idx] for idx in criteria_cards.index)
        for idx in criteria_cards.index:
            model.addConstr(
                weights[idx] <= total_weight - weights[idx],
                f"{name_prefix}_anti_dictatorship_{idx}"
            )


def _check_model_feasibility(cards_arrangement,
                             z_value,
                             e_value,
                             comp_rule_within,
                             comp_rule_successive,
                             ratio_mode,
                             normalized,
                             extra_cond,
                             min_delta=1.0,
                             conditional_gap_milp=False,
                             dynamic_unit_weight=False):
    """
    Checks feasibility once before running full sampling/robustness calculations.
    """
    model, weights, rank_groups, criteria_cards, delta = _build_srf_model(
        cards_arrangement,
        z_value,
        e_value,
        comp_rule_within,
        comp_rule_successive,
        ratio_mode,
        normalized,
        extra_cond=extra_cond,
        min_delta=min_delta,
        conditional_gap_milp=conditional_gap_milp,
        dynamic_unit_weight=dynamic_unit_weight,
        launch_smaa=False
    )

    _optimize_model(model)

    if model.status != GRB.OPTIMAL:
        raise ValueError(
            "No feasible solution found with the selected anti-dictatorship/minimum-weight requirements. "
            "Please relax these optional constraints."
        )


def _optimize_model(model):
    """
    Solves a model while handling INF_OR_UNBD by disabling dual reductions.
    """
    model.optimize()
    if model.status == GRB.INF_OR_UNBD:
        model.setParam("DualReductions", 0)
        model.optimize()


def _resolve_method_structure(srf_method):
    """
    Returns SRF modular components for a selected method.
    """
    srf_objective_map = {
        'srf': None,
        'srf_ii': None,
        'belief_degree_imprecise_srf': None,
        'hfl_srf': None,
        'robust_srf': 'Maximize ASI',
        'wap': 'Maximize ASI',
        'imprecise_srf': 'Maximize ASI',
    }
    comp_rule_within_map = {
        'srf': 'equal',
        'srf_ii': 'equal',
        'robust_srf': 'equal',
        'wap': 'equal',
        'imprecise_srf': 'equal',
        'belief_degree_imprecise_srf': 'equal',
        'hfl_srf': 'equal',
    }
    comp_rule_successive_map = {
        'srf': 'fixed-spacing',
        'srf_ii': 'fixed-spacing',
        'hfl_srf': 'hfl-linguistic-interval',
        'robust_srf': 'fully-flexible',
        'wap': 'fully-flexible',
        'imprecise_srf': 'interval-constrained',
        'belief_degree_imprecise_srf': 'probability-distribution',
    }
    ratio_mode_map = {
        'srf': 'exact-ratio',
        'robust_srf': 'exact-ratio',
        'hfl_srf': 'hfl-ratio-interval',
        'srf_ii': 'linear-spacing',
        'wap': 'interval-successive',
        'imprecise_srf': 'interval-total',
        'belief_degree_imprecise_srf': 'probability-cloud',
    }
    normalized_map = {
        'srf': True,
        'srf_ii': True,
        'robust_srf': True,
        'wap': True,
        'imprecise_srf': True,
        'belief_degree_imprecise_srf': True,
        'hfl_srf': True,
    }

    if srf_method not in srf_objective_map:
        raise ValueError('Invalid SRF method')

    srf_objective = srf_objective_map[srf_method]
    comp_rule_within = comp_rule_within_map[srf_method]
    comp_rule_successive = comp_rule_successive_map[srf_method]
    ratio_mode = ratio_mode_map[srf_method]
    normalized = normalized_map[srf_method]

    return srf_objective, comp_rule_within, comp_rule_successive, ratio_mode, normalized


def _resolve_model_configuration(srf_method, modular_options=None, modular_profile=None):
    """
    Resolves the optimization structure used by `calc_srf_flat`.

    Inconsistency analysis builds its feasibility checks from the same result, so
    an input change it reports as restoring consistency is one the weight
    calculation can actually solve.
    """
    if srf_method == 'modular_srf':
        # Modular SRF first resolves questionnaire answers into structural choices,
        # while classical methods already encode those choices in the method name.
        options, effective_method = resolve_modular_configuration(
            modular_options=modular_options,
            modular_profile=modular_profile
        )
        (srf_objective,
         comp_rule_within,
         comp_rule_successive,
         ratio_mode,
         normalized) = _resolve_modular_structure(options)
    else:
        options = None
        effective_method = srf_method
        (srf_objective,
         comp_rule_within,
         comp_rule_successive,
         ratio_mode,
         normalized) = _resolve_method_structure(effective_method)

    is_modular = options is not None
    output_variability = bool(is_modular and options.get('output_type') == 'variability')
    dynamic_unit_weight = bool(is_modular and options.get('unit_weight') == 'dynamic')

    # The zero-procedure + dynamic-unit modular variant already encodes its
    # variability through rank-specific gaps. Adding the conditional gap MILP
    # layer on top of that shrinks the feasible region and reproduces the
    # post-2b2bcdd regression seen in the attached case.
    zero_dynamic_sampling_case = bool(
        output_variability
        and options.get('procedure') == 'zero'
        and dynamic_unit_weight
    )
    # Imprecise distance variants sometimes need extra binary logic so gap bounds are
    # enforced conditionally instead of with one global spacing parameter.
    conditional_gap_milp = bool(
        output_variability
        and options.get('procedure') in {'standard', 'zero'}
        and options.get('distance_type') == 'imprecise'
        and comp_rule_successive in {
            'interval-constrained',
            'probability-distribution',
            'hfl-linguistic-interval',
        }
        and not zero_dynamic_sampling_case
    )

    # Dynamic unit weight (Q12b) is modeled through fully-flexible successive constraints.
    if dynamic_unit_weight and comp_rule_successive == 'fixed-spacing':
        comp_rule_successive = 'fully-flexible'

    return {
        'modular_options': options,
        'effective_method': effective_method,
        'srf_objective': srf_objective,
        'comp_rule_within': comp_rule_within,
        'comp_rule_successive': comp_rule_successive,
        'ratio_mode': ratio_mode,
        'normalized': normalized,
        'output_variability': output_variability,
        'dynamic_unit_weight': dynamic_unit_weight,
        'conditional_gap_milp': conditional_gap_milp,
    }


def _attach_solution_summary_columns(simos_calc_results, srf_samples=None, srf_min_max=None, decimals=2):
    """
    Adds key feasible-region summary statistics to the selected solution table:
      - center weight (mean of samples)
      - min weight (across samples)
      - max weight (across samples)

    If no samples are available, all three values fall back to the selected weights.
    """
    if not isinstance(simos_calc_results, pd.DataFrame) or 'k_i' not in simos_calc_results.columns:
        return simos_calc_results

    results = simos_calc_results.copy()
    selected = pd.to_numeric(results['k_i'], errors='coerce')

    center = selected.copy()
    min_weights = selected.copy()
    max_weights = selected.copy()

    if isinstance(srf_samples, pd.DataFrame) and not srf_samples.empty:
        # Align sample columns to result row index (criterion ids).
        aligned = srf_samples.reindex(columns=list(results.index))
        aligned = aligned.apply(pd.to_numeric, errors='coerce')

        if not aligned.empty and not aligned.dropna(axis=0, how='all').empty:
            center = aligned.mean(axis=0, skipna=True).reindex(results.index).fillna(center)

            # If no exact optimization bounds are available, use sample range.
            if not (isinstance(srf_min_max, pd.DataFrame) and not srf_min_max.empty):
                min_weights = aligned.min(axis=0, skipna=True).reindex(results.index).fillna(min_weights)
                max_weights = aligned.max(axis=0, skipna=True).reindex(results.index).fillna(max_weights)

    if isinstance(srf_min_max, pd.DataFrame) and not srf_min_max.empty:
        bounds = srf_min_max.reindex(columns=list(results.index)).apply(pd.to_numeric, errors='coerce')
        if not bounds.empty and not bounds.dropna(axis=0, how='all').empty:
            min_weights = bounds.min(axis=0, skipna=True).reindex(results.index).fillna(min_weights)
            max_weights = bounds.max(axis=0, skipna=True).reindex(results.index).fillna(max_weights)

    if isinstance(decimals, int) and decimals >= 0:
        center = center.round(decimals)
        min_weights = min_weights.round(decimals)
        max_weights = max_weights.round(decimals)

    results['k_center'] = center
    results['k_min'] = min_weights
    results['k_max'] = max_weights
    return results


def _persist_distribution_samples(cards_arrangement, samples_df):
    """
    Persists feasible-region samples for frontend boxplot rendering.
    """
    if not (isinstance(samples_df, pd.DataFrame) and not samples_df.empty):
        return

    export_df = _rename_solution_columns(cards_arrangement, samples_df)
    export_df.to_json(str(SRF_SAMPLES_PATH), orient='records')


def _select_best_robustness_rule(robustness_rules):
    """
    Picks the (ASI, barycenter) pair of the robustness rule with the highest ASI.

    A rule whose model is infeasible produces no scenarios at all, which surfaces
    as an ASI of None and an empty barycenter. Such rules are skipped so they
    neither break the comparison nor silently supply NaN weights. When no rule
    survives, the feasible region is empty for every rule and the caller's
    infeasibility handling takes over.
    """
    usable = [
        (asi, barycenter)
        for asi, barycenter in (robustness_rules or [])
        if asi is not None
        and barycenter is not None
        and not pd.isna(barycenter).all()
    ]
    if not usable:
        raise ValueError(
            "No feasible solution found for any robustness rule. The requested "
            "ratios cannot be satisfied by this ranking; please relax them."
        )

    return max(usable, key=lambda rule: rule[0])


def calc_srf_flat(cards_arrangement, z_value, e_value, w_value, srf_method,
                  modular_options=None,
                  modular_profile=None,
                  min_delta=1.0,
                  extra_constraints=None,
                  sample_size=None):
    """
    Calculates criteria weights using the specified SRF variant.

    Args:
        cards_arrangement (pd.DataFrame): Preprocessed card arrangement data
        z_value (float): Ratio between most and least important criteria
        e_value (int): Unit weight interval for SRF-II method
        w_value (int): Decimal precision for weight normalization
        srf_method (str): SRF variant to use ('srf', 'srf_ii', 'robust_srf', etc.)
        modular_options (dict, optional): Questionnaire answers for modular SRF.
        modular_profile (str, optional): Effective profile mapped from modular answers.
        min_delta (float, optional): Minimum delta for random sampling. Defaults to 1.0.
        extra_constraints (dict, optional): Optional anti-dictatorship/minimum-weight requirements.
        sample_size (int, optional): User-requested sample count for sampling-based routines.

    Returns:
        tuple:
            - pd.DataFrame: Calculated criteria weights
            - float: ASI (Average Stability Index) value
    """
    n_crit_cards = cards_arrangement['class'].to_list().count('criterion')
    is_modular = srf_method == 'modular_srf'
    requested_sampling_size = _coerce_sample_size(sample_size)

    update_calculation_progress(
        stage='setup',
        message='Preparing SRF calculation...',
        current=0,
        total=1,
        active=True,
        done=False
    )

    model_config = _resolve_model_configuration(
        srf_method,
        modular_options=modular_options,
        modular_profile=modular_profile
    )
    resolved_modular_options = model_config['modular_options']
    srf_objective = model_config['srf_objective']
    comp_rule_within = model_config['comp_rule_within']
    comp_rule_successive = model_config['comp_rule_successive']
    ratio_mode = model_config['ratio_mode']
    normalized = model_config['normalized']

    modular_output_variability = model_config['output_variability']
    modular_sampling_size = requested_sampling_size
    if is_modular:
        raw_modular_options = modular_options if isinstance(modular_options, dict) else {}
        if modular_sampling_size is None:
            raw_sampling_size = (
                raw_modular_options.get('sampling_size')
                if 'sampling_size' in raw_modular_options
                else raw_modular_options.get('sample_size')
            )
            modular_sampling_size = _coerce_sample_size(raw_sampling_size)
    modular_dynamic_unit = model_config['dynamic_unit_weight']
    modular_robust_equivalent = bool(
        is_modular and _is_modular_robust_equivalent(resolved_modular_options)
    )
    modular_wap_equivalent = bool(
        is_modular and _is_modular_wap_equivalent(resolved_modular_options)
    )
    modular_maximize_asi_equivalent = bool(modular_robust_equivalent or modular_wap_equivalent)

    # Some modular "single output" configurations are still defined over a feasible
    # region rather than one exact point. In those cases we report the center of
    # sampled/extreme solutions as the representative weight vector.
    modular_center_single_required = bool(
        is_modular
        and not modular_output_variability
        and (
            modular_dynamic_unit
            or resolved_modular_options.get('distance_type') == 'imprecise'
            or (
                resolved_modular_options.get('procedure') == 'standard'
                and resolved_modular_options.get('z_type') == 'imprecise'
            )
        )
    )
    conditional_gap_milp = model_config['conditional_gap_milp']

    """
    [O6] Extra Constraints
    """
    # Additional constraints are introduced here.
    valid_methods = {
        'srf',
        'srf_ii',
        'robust_srf',
        'wap',
        'imprecise_srf',
        'belief_degree_imprecise_srf',
        'hfl_srf',
        'modular_srf',
    }
    if srf_method not in valid_methods:
        raise ValueError('Invalid SRF method')
    extra_cond = extra_constraints if isinstance(extra_constraints, dict) else None

    if _is_extra_constraints_enabled(extra_cond):
        update_calculation_progress(
            stage='validation',
            message='Checking feasibility of the selected constraints...',
            current=0,
            total=1,
            active=True,
            done=False
        )
        _check_model_feasibility(cards_arrangement,
                                 z_value,
                                 e_value,
                                 comp_rule_within=comp_rule_within,
                                 comp_rule_successive=comp_rule_successive,
                                 ratio_mode=ratio_mode,
                                 normalized=normalized,
                                 extra_cond=extra_cond,
                                 min_delta=min_delta,
                                 conditional_gap_milp=conditional_gap_milp,
                                 dynamic_unit_weight=modular_dynamic_unit)

    # Ensure stochastic robustness routines are reproducible for identical inputs.
    stochastic_configuration = bool(
        comp_rule_successive != 'fixed-spacing'
        or ratio_mode in {'interval-total', 'interval-successive', 'probability-cloud', 'hfl-ratio-interval'}
        or modular_output_variability
        or modular_center_single_required
    )
    if stochastic_configuration:
        # Fix the pseudo-random seed from model inputs so repeated runs of the same
        # configuration produce stable ASI/PCA outputs and parity checks.
        seed_value = _derive_reproducible_sampling_seed(
            cards_arrangement=cards_arrangement,
            z_value=z_value,
            e_value=e_value,
            w_value=w_value,
            min_delta=min_delta,
            comp_rule_within=comp_rule_within,
            comp_rule_successive=comp_rule_successive,
            ratio_mode=ratio_mode,
            normalized=normalized,
            extra_cond=extra_cond,
            sample_size_hint=_resolve_sample_budget(modular_sampling_size if is_modular else requested_sampling_size)
        )
        random.seed(seed_value)
        np.random.seed(seed_value)

    """
    Robustness rules and Stability Analysis
    """
    srf_min_max = None
    srf_vertices = None
    srf_samples = None
    robustness_rules = None

    if is_modular:
        if modular_maximize_asi_equivalent:
            # Keep the modular pipeline, but apply the same robustness workflow as
            # classical Max-ASI methods (e.g., robust SRF and WAP equivalents).
            srf_min_max, asi_srf_min_max = calc_srf_min_max(cards_arrangement,
                                                            z_value,
                                                            e_value,
                                                            comp_rule_within=comp_rule_within,
                                                            comp_rule_successive=comp_rule_successive,
                                                            ratio_mode=ratio_mode,
                                                            normalized=normalized,
                                                            extra_cond=extra_cond,
                                                            min_delta=min_delta,
                                                            conditional_gap_milp=conditional_gap_milp,
                                                            dynamic_unit_weight=modular_dynamic_unit)

            srf_vertices, asi_srf_vertices = calc_srf_vertices(cards_arrangement,
                                                               z_value,
                                                               e_value,
                                                               comp_rule_within=comp_rule_within,
                                                               comp_rule_successive=comp_rule_successive,
                                                               ratio_mode=ratio_mode,
                                                               normalized=normalized,
                                                               extra_cond=extra_cond,
                                                               min_delta=min_delta,
                                                               n_samples=20 * n_crit_cards,
                                                               conditional_gap_milp=conditional_gap_milp,
                                                               dynamic_unit_weight=modular_dynamic_unit)

            sample_budget = _resolve_sample_budget(modular_sampling_size)
            srf_samples, asi_srf_samples = calc_srf_rand_samples(cards_arrangement,
                                                                 z_value,
                                                                 e_value,
                                                                 comp_rule_within=comp_rule_within,
                                                                 comp_rule_successive=comp_rule_successive,
                                                                 ratio_mode=ratio_mode,
                                                                 normalized=normalized,
                                                                 extra_cond=extra_cond,
                                                                 min_delta=min_delta,
                                                                 n_samples=sample_budget,
                                                                 conditional_gap_milp=conditional_gap_milp,
                                                                 dynamic_unit_weight=modular_dynamic_unit)

            # Kept as a list of (ASI, barycenter) pairs rather than a dict keyed by
            # ASI: rules that turn out infeasible all report ASI None, and two rules
            # can legitimately tie, so ASI values do not make unique keys.
            robustness_rules = [
                (asi_srf_min_max, srf_min_max.mean() if srf_min_max is not None else None),
                (asi_srf_vertices, srf_vertices.mean() if srf_vertices is not None else None),
                (asi_srf_samples, srf_samples.mean() if srf_samples is not None else None)
            ]
        else:
            # Modular runs may need either full variability outputs or only enough
            # samples to compute a representative center solution.
            needs_distribution = modular_output_variability or modular_center_single_required
            if needs_distribution:
                probabilistic_or_hfl = (
                    comp_rule_successive in ['probability-distribution', 'hfl-linguistic-interval']
                    or ratio_mode in ['probability-cloud', 'hfl-ratio-interval']
                )
                sample_budget = _resolve_sample_budget(modular_sampling_size)
                vertex_budget = (
                    min(max(5 * n_crit_cards, 20), 120)
                    if probabilistic_or_hfl
                    else max(10 * n_crit_cards, 40)
                )

                if modular_output_variability:
                    srf_min_max, _ = calc_srf_min_max(cards_arrangement,
                                                      z_value,
                                                      e_value,
                                                      comp_rule_within=comp_rule_within,
                                                      comp_rule_successive=comp_rule_successive,
                                                      ratio_mode=ratio_mode,
                                                      normalized=normalized,
                                                      extra_cond=extra_cond,
                                                      min_delta=min_delta,
                                                      conditional_gap_milp=conditional_gap_milp,
                                                      dynamic_unit_weight=modular_dynamic_unit)

                    srf_vertices, _ = calc_srf_vertices(cards_arrangement,
                                                        z_value,
                                                        e_value,
                                                        comp_rule_within=comp_rule_within,
                                                        comp_rule_successive=comp_rule_successive,
                                                        ratio_mode=ratio_mode,
                                                        normalized=normalized,
                                                        extra_cond=extra_cond,
                                                        min_delta=min_delta,
                                                        n_samples=vertex_budget,
                                                        conditional_gap_milp=conditional_gap_milp,
                                                        dynamic_unit_weight=modular_dynamic_unit)

                srf_samples, _ = calc_srf_rand_samples(cards_arrangement,
                                                       z_value,
                                                       e_value,
                                                       comp_rule_within=comp_rule_within,
                                                       comp_rule_successive=comp_rule_successive,
                                                       ratio_mode=ratio_mode,
                                                       normalized=normalized,
                                                       extra_cond=extra_cond,
                                                       min_delta=min_delta,
                                                       n_samples=sample_budget,
                                                       conditional_gap_milp=conditional_gap_milp,
                                                       dynamic_unit_weight=modular_dynamic_unit)

                if isinstance(srf_samples, pd.DataFrame) and srf_samples.empty and isinstance(srf_vertices, pd.DataFrame):
                    srf_samples = srf_vertices.copy()

                if (modular_center_single_required
                        and (not isinstance(srf_samples, pd.DataFrame) or srf_samples.empty)):
                    if not (isinstance(srf_min_max, pd.DataFrame) and not srf_min_max.empty):
                        srf_min_max, _ = calc_srf_min_max(cards_arrangement,
                                                          z_value,
                                                          e_value,
                                                          comp_rule_within=comp_rule_within,
                                                          comp_rule_successive=comp_rule_successive,
                                                          ratio_mode=ratio_mode,
                                                          normalized=normalized,
                                                          extra_cond=extra_cond,
                                                          min_delta=min_delta,
                                                          conditional_gap_milp=conditional_gap_milp,
                                                          dynamic_unit_weight=modular_dynamic_unit)
                    if isinstance(srf_min_max, pd.DataFrame) and not srf_min_max.empty:
                        srf_samples = srf_min_max.copy()
    else:
        if comp_rule_successive in ['fixed-spacing']:
            pass
        else:
            probabilistic_or_hfl = srf_method in ['belief_degree_imprecise_srf', 'hfl_srf']
            vertex_budget = (
                min(max(5 * n_crit_cards, 20), 120)
                if probabilistic_or_hfl
                else 20 * n_crit_cards
            )
            sample_budget = _resolve_sample_budget(requested_sampling_size)

            # Compute the variation range of the weight of each separate criterion (Max-Min approach)
            srf_min_max, asi_srf_min_max = calc_srf_min_max(cards_arrangement,
                                                            z_value,
                                                            e_value,
                                                            comp_rule_within=comp_rule_within,
                                                            comp_rule_successive=comp_rule_successive,
                                                            ratio_mode=ratio_mode,
                                                            normalized=normalized,
                                                            extra_cond=extra_cond,
                                                            min_delta=min_delta,
                                                            conditional_gap_milp=conditional_gap_milp,
                                                            dynamic_unit_weight=modular_dynamic_unit)

            # Finding and recording vertices of the polyhedron P by solving LP repeatedly
            srf_vertices, asi_srf_vertices = calc_srf_vertices(cards_arrangement,
                                                               z_value,
                                                               e_value,
                                                               comp_rule_within=comp_rule_within,
                                                               comp_rule_successive=comp_rule_successive,
                                                               ratio_mode=ratio_mode,
                                                               normalized=normalized,
                                                               extra_cond=extra_cond,
                                                               min_delta=min_delta,
                                                               n_samples=vertex_budget,
                                                               conditional_gap_milp=conditional_gap_milp,
                                                               dynamic_unit_weight=modular_dynamic_unit)

            # Random sampling to statistically analyze the feasible region
            srf_samples, asi_srf_samples = calc_srf_rand_samples(cards_arrangement,
                                                                 z_value,
                                                                 e_value,
                                                                 comp_rule_within=comp_rule_within,
                                                                 comp_rule_successive=comp_rule_successive,
                                                                 ratio_mode=ratio_mode,
                                                                 normalized=normalized,
                                                                 extra_cond=extra_cond,
                                                                 min_delta=min_delta,
                                                                 n_samples=sample_budget,
                                                                 conditional_gap_milp=conditional_gap_milp,
                                                                 dynamic_unit_weight=modular_dynamic_unit)

            if isinstance(srf_samples, pd.DataFrame) and srf_samples.empty and isinstance(srf_vertices, pd.DataFrame):
                srf_samples = srf_vertices.copy()

            # Store ASI values and barycenters of each robustness rule.
            # Kept as a list of (ASI, barycenter) pairs rather than a dict keyed by
            # ASI: rules that turn out infeasible all report ASI None, and two rules
            # can legitimately tie, so ASI values do not make unique keys.
            robustness_rules = [
                (asi_srf_min_max, srf_min_max.mean() if srf_min_max is not None else None),
                (asi_srf_vertices, srf_vertices.mean() if srf_vertices is not None else None),
                (asi_srf_samples, srf_samples.mean() if srf_samples is not None else None)
            ]

    """
    SRF Calculations
    """
    if is_modular and modular_maximize_asi_equivalent:
        simos_calc_results = pd.DataFrame(columns=['r', 'name', 'k_i'],
                                          index=cards_arrangement[cards_arrangement['class'] == 'criterion'].index[::-1])
        simos_calc_results['r'] = cards_arrangement['rank']
        simos_calc_results['name'] = cards_arrangement['name']

        # Select the mean criteria weight based on the max ASI of the three methods.
        asi_value, best_barycenter = _select_best_robustness_rule(robustness_rules)
        simos_calc_results['k_i'] = best_barycenter

        if normalized:
            simos_calc_results['k_i'] = round_up_selected(simos_calc_results['k_i'], w_value, target_sum=100)
    elif is_modular:
        simos_calc_results = calc_srf_modular(cards_arrangement,
                                              z_value,
                                              e_value,
                                              comp_rule_within=comp_rule_within,
                                              comp_rule_successive=comp_rule_successive,
                                              ratio_mode=ratio_mode,
                                              normalized=normalized,
                                              extra_cond=extra_cond,
                                              w_value=w_value,
                                              min_delta=min_delta,
                                              conditional_gap_milp=conditional_gap_milp,
                                              dynamic_unit_weight=modular_dynamic_unit)

        if modular_output_variability or modular_center_single_required:
            if isinstance(srf_samples, pd.DataFrame) and not srf_samples.empty:
                # Expose one central k_i column to the UI even when the underlying
                # modular result is a region summarized by samples/min-max bounds.
                center_weights = srf_samples.mean(axis=0, skipna=True).reindex(list(simos_calc_results.index))
                if normalized:
                    center_weights = round_up_selected(center_weights, w_value, target_sum=100)
                simos_calc_results['k_i'] = center_weights
            elif isinstance(srf_min_max, pd.DataFrame) and not srf_min_max.empty:
                center_weights = srf_min_max.mean(axis=0, skipna=True).reindex(list(simos_calc_results.index))
                if normalized:
                    center_weights = round_up_selected(center_weights, w_value, target_sum=100)
                simos_calc_results['k_i'] = center_weights

        if modular_output_variability:
            asi_value = calc_asi(srf_samples) if isinstance(srf_samples, pd.DataFrame) and not srf_samples.empty else None
        else:
            asi_value = None
    elif srf_method == 'belief_degree_imprecise_srf':
        expected_z_value, expected_e_value = _build_belief_expected_inputs(
            cards_arrangement,
            z_value,
            e_value
        )
        # Central solution from combined beliefs (expected values),
        # while distributions for ASI/PCA are still obtained through simulation.
        simos_calc_results = calc_srf_modular(cards_arrangement,
                                              expected_z_value,
                                              expected_e_value,
                                              comp_rule_within=comp_rule_within,
                                              comp_rule_successive='interval-constrained',
                                              ratio_mode='interval-total',
                                              normalized=normalized,
                                              extra_cond=extra_cond,
                                              w_value=w_value,
                                              min_delta=min_delta,
                                              conditional_gap_milp=conditional_gap_milp,
                                              dynamic_unit_weight=modular_dynamic_unit)
        asi_value = calc_asi(srf_samples) if isinstance(srf_samples, pd.DataFrame) and not srf_samples.empty else None
    elif srf_objective is None:
        simos_calc_results = calc_srf_modular(cards_arrangement,
                                              z_value,
                                              e_value,
                                              comp_rule_within=comp_rule_within,
                                              comp_rule_successive=comp_rule_successive,
                                              ratio_mode=ratio_mode,
                                              normalized=normalized,
                                              extra_cond=extra_cond,
                                              w_value=w_value,
                                              min_delta=min_delta,
                                              conditional_gap_milp=conditional_gap_milp,
                                              dynamic_unit_weight=modular_dynamic_unit)
        asi_value = calc_asi(srf_samples) if isinstance(srf_samples, pd.DataFrame) and not srf_samples.empty else None
    elif srf_objective == 'Maximize ASI':
        simos_calc_results = pd.DataFrame(columns=['r', 'name', 'k_i'],
                                          index=cards_arrangement[cards_arrangement['class'] == 'criterion'].index[::-1])
        simos_calc_results['r'] = cards_arrangement['rank']
        simos_calc_results['name'] = cards_arrangement['name']

        # Select the mean criteria weight based on the max ASI of the three methods
        asi_value, best_barycenter = _select_best_robustness_rule(robustness_rules)
        simos_calc_results['k_i'] = best_barycenter

        if normalized:
            simos_calc_results['k_i'] = round_up_selected(simos_calc_results['k_i'], w_value, target_sum=100)
    else:
        raise ValueError('Invalid SRF objective')

    # Attach optimization/simulation summary statistics only for non-crisp methods.
    methods_with_distribution_summary = {
        'robust_srf',
        'wap',
        'imprecise_srf',
        'belief_degree_imprecise_srf',
        'hfl_srf',
    }
    should_attach_distribution_summary = (
        srf_method in methods_with_distribution_summary
        or (is_modular and modular_output_variability)
    )
    if should_attach_distribution_summary:
        try:
            summary_decimals = max(0, int(w_value))
        except (TypeError, ValueError):
            summary_decimals = 1
        simos_calc_results = _attach_solution_summary_columns(
            simos_calc_results,
            srf_samples=srf_samples,
            srf_min_max=srf_min_max,
            decimals=summary_decimals
        )
        if (not (is_modular and modular_output_variability)
                and isinstance(srf_min_max, pd.DataFrame)
                and not srf_min_max.empty):
            # For classical variability-oriented methods, report the ASI from
            # the extreme-scenario matrix, which matches the documented
            # interpretation of the interface and the paper-style robust output.
            asi_value = calc_asi(srf_min_max)
        elif asi_value is None and isinstance(srf_samples, pd.DataFrame) and not srf_samples.empty:
            asi_value = calc_asi(srf_samples)
        elif asi_value is None and isinstance(srf_min_max, pd.DataFrame) and not srf_min_max.empty:
            asi_value = calc_asi(srf_min_max)

    if should_attach_distribution_summary:
        if isinstance(srf_samples, pd.DataFrame) and not srf_samples.empty:
            _persist_distribution_samples(cards_arrangement, srf_samples)
        else:
            clear_distribution_samples()

        if isinstance(srf_min_max, pd.DataFrame) and not srf_min_max.empty:
            _persist_extreme_scenarios(cards_arrangement, srf_min_max)
        else:
            clear_extreme_scenarios()

        _persist_variability_export_payload(
            cards_arrangement,
            sampling_df=srf_samples if isinstance(srf_samples, pd.DataFrame) and not srf_samples.empty else None,
            extreme_df=srf_min_max if isinstance(srf_min_max, pd.DataFrame) and not srf_min_max.empty else None,
            sampling_source='samples',
            extreme_source='min_max'
        )
    else:
        clear_distribution_samples()
        clear_extreme_scenarios()
        clear_variability_export_payload()

    # Export the 2D projection consumed by Plotly whenever a sample cloud exists.
    if isinstance(srf_samples, pd.DataFrame) and not srf_samples.empty:
        update_calculation_progress(
            stage='projection',
            message='Preparing PCA projection...',
            current=0,
            total=1,
            active=True,
            done=False
        )
        pca_vertices = srf_vertices
        if (isinstance(srf_samples, pd.DataFrame)
                and isinstance(srf_vertices, pd.DataFrame)
                and not srf_samples.empty
                and not srf_vertices.empty):
            overlapping_index = set(srf_samples.index).intersection(set(srf_vertices.index))
            if srf_samples is srf_vertices or overlapping_index:
                pca_vertices = None
        calc_pca(srf_samples, selected=simos_calc_results['k_i'], vertices=pca_vertices)

    update_calculation_progress(
        stage='finalizing',
        message='Finalizing SRF results...',
        current=1,
        total=1,
        active=True,
        done=False
    )

    return simos_calc_results, asi_value


def _add_relaxable_issue(model, issue_vars, issue_meta, issue_id, expr, bound, metadata, residual_cap=None):
    """
    Adds one relaxable inconsistency issue linked to a binary variable.
    bound='lb' encodes expr >= 0, bound='ub' encodes expr <= 0.
    `residual_cap` bounds the violation, e.g. to what the smallest admissible
    input value still allows.
    """
    y_var = model.addVar(vtype=GRB.BINARY, name=f"ei_{issue_id}")
    residual_var = model.addVar(lb=0.0, name=f"ri_{issue_id}")
    if bound == 'lb':
        model.addConstr(expr >= -residual_var, f"ei_lb_{issue_id}")
    elif bound == 'ub':
        model.addConstr(expr <= residual_var, f"ei_ub_{issue_id}")
    else:
        raise ValueError("Invalid relaxable issue bound type.")
    model.addConstr(residual_var <= INCONSISTENCY_BIG_M * y_var, f"ei_resid_link_{issue_id}")
    if residual_cap is not None:
        model.addConstr(residual_var <= residual_cap, f"ei_resid_cap_{issue_id}")

    issue_vars[issue_id] = y_var
    issue_meta[issue_id] = {
        'bound': bound,
        'expr': expr,
        'residual_var': residual_var,
        **metadata
    }
    return y_var


def _safe_positive(value, default=1.0):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if abs(numeric) > 1e-9 else float(default)


def _nullspace_matrix(matrix, atol=1e-12):
    """
    Computes an orthonormal basis of the nullspace of a matrix.
    """
    u_mat, singular_values, vt_mat = svd(matrix)
    rank = int((singular_values > atol).sum())
    return vt_mat[rank:].T


def _phase_one_feasible_point(A_ub, b_ub, A_eq, b_eq):
    """
    Finds one feasible point for a linear constraint system using scipy.linprog.
    """
    n_vars = int(A_eq.shape[1] if A_eq is not None and A_eq.size else A_ub.shape[1])
    result = linprog(
        c=np.zeros(n_vars),
        A_ub=A_ub if A_ub.size else None,
        b_ub=b_ub if b_ub.size else None,
        A_eq=A_eq if A_eq is not None and A_eq.size else None,
        b_eq=b_eq if b_eq is not None and b_eq.size else None,
        bounds=[(None, None)] * n_vars,
        method='highs'
    )
    if not result.success:
        raise ValueError(f"Infeasible constraint system for hit-and-run sampling: {result.message}")
    return result.x


def _lp_var_is_continuous(var):
    """
    Returns True when a PuLP-backed variable is continuous.
    """
    return str(getattr(var, 'cat', 'Continuous')).strip().lower() == 'continuous'


def _extract_freeopt_polytope(model):
    """
    Converts the current FreeOpt/PuLP model into matrix form for hit-and-run.
    """
    variables = list(getattr(model, '_vars', []))
    n_vars = len(variables)
    if n_vars == 0:
        return (
            variables,
            np.empty((0, 0), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0, 0), dtype=float),
            np.empty((0,), dtype=float),
        )

    var_pos = {var: idx for idx, var in enumerate(variables)}
    A_ub_rows = []
    b_ub_rows = []
    A_eq_rows = []
    b_eq_rows = []

    for idx, var in enumerate(variables):
        lb = getattr(var, 'lowBound', None)
        ub = getattr(var, 'upBound', None)

        if lb is not None:
            lb = float(lb)
            if np.isfinite(lb):
                row = np.zeros(n_vars, dtype=float)
                row[idx] = -1.0
                A_ub_rows.append(row)
                b_ub_rows.append(-lb)

        if ub is not None:
            ub = float(ub)
            if np.isfinite(ub):
                row = np.zeros(n_vars, dtype=float)
                row[idx] = 1.0
                A_ub_rows.append(row)
                b_ub_rows.append(ub)

    for constraint in getattr(model, '_problem', {}).constraints.values():
        row = np.zeros(n_vars, dtype=float)
        for var, coeff in constraint.items():
            row[var_pos[var]] = float(coeff)

        rhs = -float(constraint.constant)
        sense = int(constraint.sense)
        if sense == -1:
            A_ub_rows.append(row)
            b_ub_rows.append(rhs)
        elif sense == 1:
            A_ub_rows.append(-row)
            b_ub_rows.append(-rhs)
        elif sense == 0:
            A_eq_rows.append(row)
            b_eq_rows.append(rhs)

    A_ub = np.asarray(A_ub_rows, dtype=float) if A_ub_rows else np.empty((0, n_vars), dtype=float)
    b_ub = np.asarray(b_ub_rows, dtype=float) if b_ub_rows else np.empty((0,), dtype=float)
    A_eq = np.asarray(A_eq_rows, dtype=float) if A_eq_rows else np.empty((0, n_vars), dtype=float)
    b_eq = np.asarray(b_eq_rows, dtype=float) if b_eq_rows else np.empty((0,), dtype=float)

    # Some SRF models express exact relations through two opposite inequalities
    # (for example, when an interval input collapses to one exact value). Keeping
    # those pairs in A_ub can trap hit-and-run at a boundary vertex even though
    # the feasible region still has positive dimension. Fold such pairs into the
    # equality system before sampling.
    if A_ub.size:
        used_rows = np.zeros(len(A_ub), dtype=bool)
        promoted_eq_rows = []
        promoted_eq_rhs = []
        row_tol = 1e-10

        for left in range(len(A_ub)):
            if used_rows[left]:
                continue
            for right in range(left + 1, len(A_ub)):
                if used_rows[right]:
                    continue
                if (np.allclose(A_ub[left], -A_ub[right], atol=row_tol, rtol=0.0)
                        and np.isclose(b_ub[left], -b_ub[right], atol=row_tol, rtol=0.0)):
                    promoted_eq_rows.append(A_ub[left].copy())
                    promoted_eq_rhs.append(float(b_ub[left]))
                    used_rows[left] = True
                    used_rows[right] = True
                    break

        if promoted_eq_rows:
            remaining_mask = ~used_rows
            A_ub = A_ub[remaining_mask]
            b_ub = b_ub[remaining_mask]
            promoted_eq = np.asarray(promoted_eq_rows, dtype=float)
            promoted_rhs = np.asarray(promoted_eq_rhs, dtype=float)
            if A_eq.size:
                A_eq = np.vstack([A_eq, promoted_eq])
                b_eq = np.concatenate([b_eq, promoted_rhs])
            else:
                A_eq = promoted_eq
                b_eq = promoted_rhs

    return variables, A_ub, b_ub, A_eq, b_eq


def _hit_and_run_polytope(A_ub,
                          b_ub,
                          A_eq,
                          x0,
                          n_samples,
                          burn_in,
                          thinning,
                          rng,
                          progress_stage=None,
                          progress_message='Sampling feasible solutions'):
    """
    Hit-and-run sampler over a polytope defined by A_ub x <= b_ub and A_eq x = const.
    """
    basis = _nullspace_matrix(A_eq)
    if basis.size == 0:
        if progress_stage is not None:
            update_calculation_progress(
                progress_stage,
                progress_message,
                current=n_samples,
                total=n_samples,
                active=True,
                done=False
            )
        return np.repeat(np.asarray(x0, dtype=float)[None, :], repeats=n_samples, axis=0)

    x_vec = np.asarray(x0, dtype=float).copy()
    samples = []
    eps = 1e-12
    n_target_steps = int(burn_in + thinning * n_samples)
    max_steps = max(n_target_steps * 3, n_target_steps + 500)
    steps = 0

    if progress_stage is not None:
        update_calculation_progress(
            progress_stage,
            progress_message,
            current=0,
            total=n_samples,
            active=True,
            done=False
        )

    while len(samples) < n_samples and steps < max_steps:
        steps += 1
        direction = basis @ rng.normal(size=basis.shape[1])
        norm_dir = np.linalg.norm(direction)
        if norm_dir <= eps:
            continue
        direction = direction / norm_dir

        lower_t = -np.inf
        upper_t = np.inf
        feasible = True
        for row_a, bound_b in zip(A_ub, b_ub):
            ad_val = float(row_a @ direction)
            ax_val = float(row_a @ x_vec)
            if abs(ad_val) <= eps:
                if ax_val > float(bound_b) + 1e-10:
                    feasible = False
                    break
                continue
            t_val = (float(bound_b) - ax_val) / ad_val
            if ad_val > 0:
                upper_t = min(upper_t, t_val)
            else:
                lower_t = max(lower_t, t_val)
            if lower_t > upper_t:
                feasible = False
                break
        if not feasible:
            continue
        if not np.isfinite(lower_t) or not np.isfinite(upper_t) or upper_t - lower_t <= 2 * eps:
            continue

        step_t = float(rng.uniform(lower_t + eps, upper_t - eps))
        x_vec = x_vec + step_t * direction

        if steps > burn_in and ((steps - burn_in) % thinning == 0):
            samples.append(x_vec.copy())
            if progress_stage is not None and _should_emit_progress(len(samples), n_samples):
                update_calculation_progress(
                    progress_stage,
                    progress_message,
                    current=len(samples),
                    total=n_samples,
                    active=True,
                    done=False
                )

    if not samples:
        return np.repeat(np.asarray(x0, dtype=float)[None, :], repeats=n_samples, axis=0)

    if len(samples) < n_samples:
        pad = np.repeat(np.asarray(samples[-1])[None, :], repeats=n_samples - len(samples), axis=0)
        output = np.vstack([np.asarray(samples), pad])
        if progress_stage is not None:
            update_calculation_progress(
                progress_stage,
                progress_message,
                current=n_samples,
                total=n_samples,
                active=True,
                done=False
            )
        return output

    output = np.asarray(samples[:n_samples])
    if progress_stage is not None:
        update_calculation_progress(
            progress_stage,
            progress_message,
            current=n_samples,
            total=n_samples,
            active=True,
            done=False
    )
    return output


def _try_hit_and_run_model_samples(model,
                                   weights,
                                   criteria_cards,
                                   n_samples,
                                   normalized=True,
                                   progress_message='Uniformly sampling feasible solutions'):
    """
    Runs hit-and-run on the continuous feasible model and projects samples to weights.
    """
    variables = list(getattr(model, '_vars', []))
    if not variables or any(not _lp_var_is_continuous(var) for var in variables):
        return None

    _optimize_model(model)
    if model.status != GRB.OPTIMAL:
        return None

    variables, A_ub, b_ub, A_eq, b_eq = _extract_freeopt_polytope(model)
    if not variables:
        return None

    try:
        x0 = np.asarray([float(var.X) for var in variables], dtype=float)
        if not np.all(np.isfinite(x0)):
            raise ValueError("Non-finite seed point from solver.")
    except Exception:
        x0 = _phase_one_feasible_point(A_ub, b_ub, A_eq, b_eq)

    rng_seed = int(np.random.randint(0, np.iinfo(np.int32).max))
    rng = np.random.default_rng(rng_seed)
    n_vars = len(variables)
    burn_in = min(max(200, 8 * n_vars), 2000)
    thinning = max(1, min(6, max(1, n_vars // 2)))
    sampled_points = _hit_and_run_polytope(
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        x0=x0,
        n_samples=int(max(1, n_samples)),
        burn_in=burn_in,
        thinning=thinning,
        rng=rng,
        progress_stage='sampling',
        progress_message=progress_message
    )

    weight_positions = {
        var: idx for idx, var in enumerate(variables)
    }
    weight_matrix = np.column_stack([
        sampled_points[:, weight_positions[weights[idx]]]
        for idx in criteria_cards.index
    ]) if len(criteria_cards.index) else np.empty((len(sampled_points), 0), dtype=float)

    if normalized and weight_matrix.size:
        row_sums = weight_matrix.sum(axis=1, keepdims=True)
        positive_mask = row_sums[:, 0] > 0
        if positive_mask.any():
            weight_matrix[positive_mask] = (
                weight_matrix[positive_mask] / row_sums[positive_mask]
            ) * 100.0

    hitrun_df = pd.DataFrame(weight_matrix, columns=criteria_cards.index)
    hitrun_df.index = [f'sample_{idx + 1}' for idx in range(len(hitrun_df))]
    return hitrun_df


def _try_hit_and_run_zero_dynamic_samples(cards_arrangement,
                                          e_value,
                                          extra_cond,
                                          min_delta,
                                          n_samples,
                                          normalized=True,
                                          conditional_gap_milp=False):
    """
    Specialized hit-and-run sampler for modular SRF case:
    zero-criterion procedure + imprecise distance intervals + dynamic C.

    Returns:
        pd.DataFrame or None: Criterion-level samples in original criterion index columns.
    """
    if not normalized:
        return None
    if not isinstance(e_value, dict):
        return None
    if not ({'emin_0', 'emax_0'} & set(e_value.keys()) or 'e0' in e_value):
        return None

    criteria_cards = cards_arrangement[cards_arrangement['class'] == 'criterion'].sort_values('rank')
    if criteria_cards.empty:
        return None

    rank_groups = {}
    for rank in criteria_cards['rank'].unique():
        rank_groups[int(rank)] = criteria_cards[criteria_cards['rank'] == rank].index.tolist()
    sorted_ranks = sorted(rank_groups.keys())
    n_ranks = len(sorted_ranks)
    if n_ranks < 2:
        return None

    rank_pos = {rank: idx for idx, rank in enumerate(sorted_ranks)}
    rank_sizes = np.array([len(rank_groups[rank]) for rank in sorted_ranks], dtype=float)

    min_weight_req = extra_cond.get('minimum_weight', {}) if isinstance(extra_cond, dict) else {}
    min_weight_enabled = isinstance(min_weight_req, dict) and bool(min_weight_req.get('enabled'))
    min_weight_value = float(min_weight_req.get('value', 0.0)) if min_weight_enabled else 0.0
    lb_rank = max(0.0, min_weight_value / 100.0)

    dictatorship_req = extra_cond.get('dictatorship', {}) if isinstance(extra_cond, dict) else {}
    dictatorship_enabled = isinstance(dictatorship_req, dict) and bool(dictatorship_req.get('enabled'))
    ub_rank = 0.5 if dictatorship_enabled else 1.0

    rank_white_count = {}
    for rank in cards_arrangement['rank'].unique():
        rank_white_count[int(rank)] = (
            cards_arrangement[cards_arrangement['rank'] == rank]['class'].to_list().count('white') + 1
        )

    gap_bounds = []
    bar_sum_low = 0.0
    bar_sum_high = 0.0
    for i in range(1, n_ranks):
        prev_rank = sorted_ranks[i - 1]
        default_e = float(max(0, rank_white_count.get(prev_rank, 1) - 1))
        e_low = float(e_value.get(f'emin_{prev_rank}', e_value.get(f'emax_{prev_rank}', default_e)))
        e_high = float(e_value.get(f'emax_{prev_rank}', e_value.get(f'emin_{prev_rank}', e_low)))
        if e_high < e_low:
            raise ValueError(f"Invalid interval for rank gap after rank {prev_rank}: emin > emax.")
        gap_bounds.append((e_low, e_high))
        bar_sum_low += e_low + 1.0
        bar_sum_high += e_high + 1.0

    e0_low = float(e_value.get('emin_0', e_value.get('e0', e_value.get('emax_0', 0.0))))
    e0_high = float(e_value.get('emax_0', e_value.get('e0', e0_low)))
    if e0_high < e0_low:
        raise ValueError("Invalid e0 interval: emin_0 > emax_0.")
    if e0_low < 0 or e0_high < 0:
        raise ValueError("e0 interval bounds must be non-negative.")

    z_low = (bar_sum_low + e0_high + 1.0) / (e0_high + 1.0)
    z_high = (bar_sum_high + e0_low + 1.0) / (e0_low + 1.0)
    if z_high < z_low:
        z_low, z_high = z_high, z_low

    delta_frac = float(max(min_delta, 0.0) / 100.0)

    A_rows = []
    b_rows = []

    for r in range(n_ranks):
        row = np.zeros(n_ranks)
        row[r] = -1.0
        A_rows.append(row)
        b_rows.append(-lb_rank)

        row = np.zeros(n_ranks)
        row[r] = 1.0
        A_rows.append(row)
        b_rows.append(ub_rank)

    for r in range(n_ranks - 1):
        row = np.zeros(n_ranks)
        row[r] = 1.0
        row[r + 1] = -1.0
        A_rows.append(row)
        b_rows.append(-delta_frac)

    if conditional_gap_milp and len(gap_bounds) >= 2:
        def _append_gap_relation(first_gap_idx, second_gap_idx, min_gap):
            """
            Encodes diff(first_gap) - diff(second_gap) >= min_gap
            in the rank-weight polytope coordinates.
            """
            row = np.zeros(n_ranks)
            row[first_gap_idx] += 1.0
            row[first_gap_idx + 1] += -1.0
            row[second_gap_idx] += -1.0
            row[second_gap_idx + 1] += 1.0
            A_rows.append(row)
            b_rows.append(-float(min_gap))

        for left in range(len(gap_bounds) - 1):
            for right in range(left + 1, len(gap_bounds)):
                left_low, left_high = gap_bounds[left]
                right_low, right_high = gap_bounds[right]
                can_left_gt = left_high > right_low
                can_right_gt = right_high > left_low
                can_equal = not (left_high < right_low or right_high < left_low)

                # Mirror the deterministic consequences of the exact MILP logic:
                # when one gap can never exceed the other, preserve that weak/strict
                # ordering in the continuous rank-weight sampler as well.
                if not can_left_gt and not can_right_gt and can_equal:
                    _append_gap_relation(left, right, 0.0)
                    _append_gap_relation(right, left, 0.0)
                    continue

                if not can_left_gt:
                    _append_gap_relation(
                        right,
                        left,
                        0.0 if can_equal else delta_frac
                    )
                    continue

                if not can_right_gt:
                    _append_gap_relation(
                        left,
                        right,
                        0.0 if can_equal else delta_frac
                    )

    row = np.zeros(n_ranks)
    row[0] = z_low
    row[-1] = -1.0
    A_rows.append(row)
    b_rows.append(0.0)

    row = np.zeros(n_ranks)
    row[0] = -z_high
    row[-1] = 1.0
    A_rows.append(row)
    b_rows.append(0.0)

    A_ub = np.array(A_rows, dtype=float) if A_rows else np.empty((0, n_ranks), dtype=float)
    b_ub = np.array(b_rows, dtype=float) if b_rows else np.empty((0,), dtype=float)
    A_eq = np.array([rank_sizes], dtype=float)
    b_eq = np.array([1.0], dtype=float)

    x0 = _phase_one_feasible_point(A_ub, b_ub, A_eq, b_eq)

    rng = np.random.default_rng()
    burn_in = min(max(1000, 10 * n_ranks), 5000)
    thinning = max(2, min(10, 2 * n_ranks))
    rank_samples = _hit_and_run_polytope(
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        x0=x0,
        n_samples=int(max(1, n_samples)),
        burn_in=burn_in,
        thinning=thinning,
        rng=rng,
        progress_stage='sampling',
        progress_message='Sampling feasible solutions'
    )

    crit_indices = list(criteria_cards.index)
    sample_matrix = np.zeros((rank_samples.shape[0], len(crit_indices)), dtype=float)
    for j, crit_idx in enumerate(crit_indices):
        rnk = int(criteria_cards.loc[crit_idx, 'rank'])
        sample_matrix[:, j] = rank_samples[:, rank_pos[rnk]]

    if normalized:
        sample_matrix *= 100.0

    hitrun_df = pd.DataFrame(sample_matrix, columns=crit_indices)
    hitrun_df.index = [f'sample_{idx + 1}' for idx in range(len(hitrun_df))]
    return hitrun_df


def _estimate_linear_spacing_e0_anchor(e_value):
    """
    Returns a numeric e0 anchor used to evaluate linear-spacing ratio consistency.
    Supports exact, interval, HFL, and belief-distribution e0 encodings.
    """
    if isinstance(e_value, dict):
        e0_cloud = _extract_probability_pairs(
            e_value,
            value_prefix='e-value-0-',
            beta_prefix='e-beta-0-'
        )
        if e0_cloud:
            normalized_cloud = _normalize_probability_cloud(e0_cloud)
            expected_e0 = sum(val * beta for val, beta in normalized_cloud.items())
            return max(0.0, float(expected_e0))

        if 'e0' in e_value:
            try:
                return max(0.0, float(e_value.get('e0', 0)))
            except (TypeError, ValueError):
                return 0.0

        if 'rmin_0' in e_value or 'rmax_0' in e_value:
            try:
                r_min_term = int(e_value.get('rmin_0', e_value.get('rmax_0', HFL_CARD_MIN_TERM)))
                r_max_term = int(e_value.get('rmax_0', e_value.get('rmin_0', r_min_term)))
                r_min = _map_hfl_card_term(min(r_min_term, r_max_term))
                r_max = _map_hfl_card_term(max(r_min_term, r_max_term))
                return max(0.0, float((r_min + r_max) / 2.0))
            except (TypeError, ValueError):
                return 0.0

        if 'emin_0' in e_value or 'emax_0' in e_value:
            try:
                e_min = float(e_value.get('emin_0', e_value.get('emax_0', 0)))
                e_max = float(e_value.get('emax_0', e_value.get('emin_0', e_min)))
                return max(0.0, float((min(e_min, e_max) + max(e_min, e_max)) / 2.0))
            except (TypeError, ValueError):
                return 0.0

        return 0.0

    try:
        return max(0.0, float(e_value))
    except (TypeError, ValueError):
        return 0.0


def _input_parameter(key,
                     label,
                     current,
                     direction,
                     value_type='int',
                     lower=None,
                     upper=None,
                     rank_pair=None,
                     ratio=None):
    """
    Describes the user input that a relaxable EI issue asks to change.

    `ratio` is set for ratio inputs. It names the model constraints encoding the
    input so the restoration search can compute the input's feasible range exactly.
    """
    return {
        'key': key,
        'label': label,
        'current': current,
        'direction': direction,
        'value_type': value_type,
        'lower': lower,
        'upper': upper,
        'rank_pair': rank_pair,
        'ratio': ratio,
    }


def _add_ratio_issue(model,
                     issue_vars,
                     issue_meta,
                     issue_id,
                     num_var,
                     den_var,
                     value,
                     bound,
                     parameter,
                     floor_value=None,
                     ceiling_value=None):
    """
    Adds a relaxable `num >= value * den` (bound='lb') or `num <= value * den`
    (bound='ub') issue. The relaxation stops at the admissible input range, so a
    lowered ratio never goes below `floor_value` and a raised one never above
    `ceiling_value`. When the input cannot change (`parameter` is None) or already
    sits at that limit, the constraint is added as a hard constraint and None is
    returned.
    """
    expr = num_var - value * den_var
    if parameter is None:
        model.addConstr(expr >= 0 if bound == 'lb' else expr <= 0, f"ei_hard_{issue_id}")
        return None
    residual_cap = None
    if bound == 'lb' and floor_value is not None:
        if value <= floor_value + 1e-9:
            model.addConstr(expr >= 0, f"ei_hard_{issue_id}")
            return None
        residual_cap = (value - floor_value) * den_var
    elif bound == 'ub' and ceiling_value is not None:
        if value >= ceiling_value - 1e-9:
            model.addConstr(expr <= 0, f"ei_hard_{issue_id}")
            return None
        residual_cap = (ceiling_value - value) * den_var

    return _add_relaxable_issue(
        model, issue_vars, issue_meta,
        issue_id=issue_id,
        expr=expr,
        bound=bound,
        metadata={'parameter': parameter},
        residual_cap=residual_cap
    )


def _linear_spacing_e0_parameters(e_value):
    """
    Returns the e0 inputs that lower and raise the SRF-II implied ratio as
    (raise-e0 parameter, lower-e0 parameter). A larger e0 flattens the implied
    ratio towards 1. Either entry is None when that input is already at its limit.
    """
    if not isinstance(e_value, dict):
        current = int(round(_estimate_linear_spacing_e0_anchor(e_value)))
        label = "e0 value (SRF-II)"
        return (
            _input_parameter(('e0',), label, current, 'increase', upper=current + INCONSISTENCY_MAX_E0_INCREASE),
            _input_parameter(('e0',), label, current, 'decrease', lower=0) if current > 0 else None
        )

    e0_cloud = _extract_probability_pairs(e_value, value_prefix='e-value-0-', beta_prefix='e-beta-0-')
    if e0_cloud:
        e0_cloud = _normalize_probability_cloud(e0_cloud)
        low = int(np.floor(min(e0_cloud.keys())))
        high = int(np.ceil(max(e0_cloud.keys())))
        return (
            _input_parameter(('e_support', 0, 'max'), "largest e0 value in the belief distribution", high,
                             'increase', upper=high + INCONSISTENCY_MAX_E0_INCREASE),
            _input_parameter(('e_support', 0, 'min'), "smallest e0 value in the belief distribution", low,
                             'decrease', lower=0) if low > 0 else None
        )

    if 'e0' in e_value:
        current = int(float(e_value['e0']))
        label = "e0 value"
        return (
            _input_parameter(('e_key', 'e0'), label, current, 'increase', upper=current + INCONSISTENCY_MAX_E0_INCREASE),
            _input_parameter(('e_key', 'e0'), label, current, 'decrease', lower=0) if current > 0 else None
        )

    if 'rmin_0' in e_value or 'rmax_0' in e_value:
        r_min = int(e_value.get('rmin_0', e_value.get('rmax_0', HFL_CARD_MIN_TERM)))
        r_max = int(e_value.get('rmax_0', r_min))
        return (
            _input_parameter(('e_key', 'rmax_0'), "HFL upper e0 term", r_max, 'increase',
                             upper=HFL_CARD_MAX_TERM) if r_max < HFL_CARD_MAX_TERM else None,
            _input_parameter(('e_key', 'rmin_0'), "HFL lower e0 term", r_min, 'decrease',
                             lower=HFL_CARD_MIN_TERM) if r_min > HFL_CARD_MIN_TERM else None
        )

    if 'emin_0' in e_value or 'emax_0' in e_value:
        e_min = int(float(e_value.get('emin_0', e_value.get('emax_0', 0))))
        e_max = int(float(e_value.get('emax_0', e_min)))
        return (
            _input_parameter(('e_key', 'emax_0'), "maximum e0 value", e_max, 'increase',
                             upper=e_max + INCONSISTENCY_MAX_E0_INCREASE),
            _input_parameter(('e_key', 'emin_0'), "minimum e0 value", e_min, 'decrease',
                             lower=0) if e_min > 0 else None
        )

    return None, None


def _z_scale_parameters(z_value):
    """
    Classical belief-degree SRF: parameters rescaling every z value of the belief
    distribution proportionally around 1 (z -> 1 + k * (z - 1)), identified by the
    resulting expected z. Rescaling keeps the betas and the order of the values
    and, unlike an equal shift, can always move a wide distribution towards 1.
    Returns (decrease parameter, increase parameter); either is None at its limit.
    """
    cloud, z_min, z_max, expected = _probability_cloud_stats(
        _extract_probability_pairs(z_value, value_prefix='z-value-', beta_prefix='z-beta-')
    )
    if z_min <= 1.0:
        return None, None

    # Every rescaled z value must stay inside the admissible z range.
    lower = 1.0 + (INCONSISTENCY_Z_MIN - 1.0) * (expected - 1.0) / (z_min - 1.0)
    upper = 1.0 + (INCONSISTENCY_Z_MAX - 1.0) * (expected - 1.0) / (z_max - 1.0)
    label = "expected z of the belief distribution"
    ratio = {'kind': 'expected'}

    parameters = []
    for direction, available in (('decrease', expected > lower + 1e-9), ('increase', expected < upper - 1e-9)):
        if not available:
            parameters.append(None)
            continue
        parameter = _input_parameter(('z_scale',), label, expected, direction, value_type='float',
                                     lower=lower, upper=upper, ratio=ratio)
        parameter['support_values'] = sorted(cloud.keys())
        parameters.append(parameter)
    return tuple(parameters)


def _e_shift_parameters(e_value, prev_rank, pair_label, rank_pair):
    """
    Classical belief-degree SRF: parameters moving every blank-card value of one
    gap distribution by the same whole number, identified by the resulting
    smallest value. Returns (decrease parameter, increase parameter).
    """
    cloud = {}
    if isinstance(e_value, dict):
        cloud = _extract_probability_pairs(
            e_value,
            value_prefix=f"e-value-{prev_rank}-",
            beta_prefix=f"e-beta-{prev_rank}-",
        )
    if not cloud:
        return None, None

    support = sorted(_normalize_probability_cloud(cloud).keys())
    smallest = int(round(support[0]))
    key = ('e_shift', int(prev_rank))
    label = f"blank-card value in the belief distribution between {pair_label}"
    decrease = _input_parameter(key, label, smallest, 'decrease', lower=0, rank_pair=rank_pair) if smallest > 0 else None
    increase = _input_parameter(key, label, smallest, 'increase',
                                upper=smallest + INCONSISTENCY_MAX_GAP_INCREASE, rank_pair=rank_pair)
    for parameter in (decrease, increase):
        if parameter is not None:
            parameter['support_values'] = support
    return decrease, increase


def _shift_probability_values(values, value_prefix, beta_prefix, target_smallest):
    """
    Adds the same amount to every value of a flat (value, beta) distribution so
    that its smallest value becomes `target_smallest`.
    """
    cloud = _extract_probability_pairs(values, value_prefix=value_prefix, beta_prefix=beta_prefix)
    if not cloud:
        raise ValueError("No probability distribution is available to shift.")
    _normalized, smallest, _largest, _expected = _probability_cloud_stats(cloud)
    shift = float(target_smallest) - smallest
    for key in list(values.keys()):
        if key.startswith(value_prefix) and f"{beta_prefix}{key[len(value_prefix):]}" in values:
            values[key] = float(values[key]) + shift


def _scale_probability_values(values, value_prefix, beta_prefix, target_expected):
    """
    Rescales every value of a flat (value, beta) ratio distribution around 1 so
    that its expected value becomes `target_expected`.
    """
    cloud = _extract_probability_pairs(values, value_prefix=value_prefix, beta_prefix=beta_prefix)
    if not cloud:
        raise ValueError("No probability distribution is available to rescale.")
    _normalized, _smallest, _largest, expected = _probability_cloud_stats(cloud)
    factor = (float(target_expected) - 1.0) / (expected - 1.0)
    for key in list(values.keys()):
        if key.startswith(value_prefix) and f"{beta_prefix}{key[len(value_prefix):]}" in values:
            values[key] = 1.0 + (float(values[key]) - 1.0) * factor


def _add_expected_belief_issues(model,
                                issue_vars,
                                issue_meta,
                                cards_arrangement,
                                z_value,
                                e_value,
                                criteria_cards,
                                rank_white_count,
                                comp_rule_within,
                                normalized,
                                extra_cond,
                                min_delta,
                                z_scale_parameters,
                                e_shift_parameters):
    """
    Classical belief-degree SRF reports the weights of the expected inputs, so the
    EI model also contains that solution. Its issues use the same distribution
    parameters as the support model, so one proposed edit is judged against both.
    """
    expected_z, expected_e = _build_belief_expected_inputs(cards_arrangement, z_value, e_value)
    weights = {
        idx: model.addVar(lb=0, name=f"w_expected_{idx}")
        for idx in criteria_cards.index
    }
    spacing_scale = model.addVar(lb=max(float(min_delta), 1e-6), name="ei_expected_scale")

    rank_groups = {}
    for rank in criteria_cards['rank'].unique():
        rank_groups[rank] = criteria_cards[criteria_cards['rank'] == rank].index.tolist()
    sorted_ranks = sorted(rank_groups.keys())

    if comp_rule_within == 'equal':
        for rank, indices in rank_groups.items():
            for i in range(1, len(indices)):
                model.addConstr(
                    weights[indices[0]] == weights[indices[i]],
                    f"ei_expected_equal_within_rank_{rank}_{i}"
                )

    for i in range(1, len(sorted_ranks)):
        prev_rank = sorted_ranks[i - 1]
        curr_rank = sorted_ranks[i]
        diff_expr = weights[rank_groups[curr_rank][0]] - weights[rank_groups[prev_rank][0]]
        expected_gap_key = f'emin_{prev_rank}'
        if expected_gap_key not in expected_e:
            # Gaps without blank cards keep the deck spacing in the expected model.
            model.addConstr(
                diff_expr == spacing_scale * rank_white_count[prev_rank],
                f"ei_expected_gap_fixed_{prev_rank}"
            )
            continue

        expr = diff_expr - spacing_scale * (float(expected_e[expected_gap_key]) + 1.0)
        decrease, increase = e_shift_parameters.get(int(prev_rank), (None, None))
        y_decrease = None
        y_increase = None
        if decrease is not None:
            # Values cannot drop below zero, so the expected gap shrinks by at most
            # the smallest value of the distribution.
            y_decrease = _add_relaxable_issue(
                model, issue_vars, issue_meta,
                issue_id=f"expected_gap_{prev_rank}_plus",
                expr=expr,
                bound='lb',
                metadata={'parameter': decrease},
                residual_cap=spacing_scale * decrease['current']
            )
        else:
            model.addConstr(expr >= 0, f"ei_hard_expected_gap_{prev_rank}_plus")
        if increase is not None:
            y_increase = _add_relaxable_issue(
                model, issue_vars, issue_meta,
                issue_id=f"expected_gap_{prev_rank}_minus",
                expr=expr,
                bound='ub',
                metadata={'parameter': increase}
            )
        else:
            model.addConstr(expr <= 0, f"ei_hard_expected_gap_{prev_rank}_minus")
        if y_decrease is not None and y_increase is not None:
            model.addConstr(y_decrease + y_increase <= 1, f"ei_expected_gap_{prev_rank}_exclusive")

    num_var, den_var = _ratio_pair_variables(('global',), weights, rank_groups)
    expected_ratio = float(expected_z['zmin'])
    decrease, increase = z_scale_parameters
    y_decrease = _add_ratio_issue(
        model, issue_vars, issue_meta, "expected_z_min",
        num_var, den_var, expected_ratio, 'lb', decrease,
        floor_value=decrease['lower'] if decrease is not None else None
    )
    y_increase = _add_ratio_issue(
        model, issue_vars, issue_meta, "expected_z_max",
        num_var, den_var, expected_ratio, 'ub', increase,
        ceiling_value=increase['upper'] if increase is not None else None
    )
    if y_decrease is not None and y_increase is not None:
        model.addConstr(y_decrease + y_increase <= 1, "ei_expected_z_exclusive")

    if normalized:
        model.addConstr(gp.quicksum(weights.values()) == 100, "ei_expected_normalization")
    if extra_cond is not None:
        _add_optional_extra_constraints(model, weights, criteria_cards, extra_cond, name_prefix="ei_expected_extra")


def _set_probability_support(values, value_prefix, beta_prefix, which, new_value):
    """
    Moves the smallest (`which='min'`) or largest (`which='max'`) value of a
    flat (value, beta) distribution to `new_value`.
    """
    support = {}
    for key, raw_value in values.items():
        if not key.startswith(value_prefix):
            continue
        beta_key = f"{beta_prefix}{key[len(value_prefix):]}"
        if beta_key in values and float(values[beta_key]) > 0:
            support[key] = float(raw_value)
    if not support:
        raise ValueError("No probability support is available to adjust.")

    target = min(support.values()) if which == 'min' else max(support.values())
    for key, support_value in support.items():
        if abs(support_value - target) <= 1e-12:
            values[key] = new_value


def _apply_input_adjustments(z_value, e_value, adjustments):
    """
    Returns copies of the z/e inputs plus blank-card overrides with `adjustments`
    ({parameter key: new value}) applied.
    """
    z_adjusted = dict(z_value) if isinstance(z_value, dict) else z_value
    e_adjusted = dict(e_value) if isinstance(e_value, dict) else e_value
    gap_overrides = {}

    for key, value in adjustments.items():
        kind = key[0]
        if kind == 'gap':
            gap_overrides[int(key[1])] = int(value)
        elif kind == 'z':
            z_adjusted = float(value)
        elif kind == 'e0':
            e_adjusted = int(value)
        elif kind == 'z_key':
            z_adjusted[key[1]] = value
        elif kind == 'e_key':
            e_adjusted[key[1]] = value
        elif kind == 'z_support':
            _set_probability_support(z_adjusted, 'z-value-', 'z-beta-', key[1], value)
        elif kind == 'e_support':
            _set_probability_support(
                e_adjusted, f"e-value-{key[1]}-", f"e-beta-{key[1]}-", key[2], value
            )
        elif kind == 'z_scale':
            _scale_probability_values(z_adjusted, 'z-value-', 'z-beta-', value)
        elif kind == 'e_shift':
            _shift_probability_values(
                e_adjusted, f"e-value-{key[1]}-", f"e-beta-{key[1]}-", int(value)
            )
        else:
            raise ValueError(f"Unknown input adjustment: {key}")

    return z_adjusted, e_adjusted, gap_overrides


def _build_adjusted_model(context, adjustments, drop_constraints=(), expected_inputs=False):
    """
    Builds the weight-calculation model for the original inputs with `adjustments`
    applied, removing the constraints named in `drop_constraints`.
    """
    config = context['config']
    z_adjusted, e_adjusted, gap_overrides = _apply_input_adjustments(
        context['z_value'], context['e_value'], adjustments
    )
    comp_rule_successive = config['comp_rule_successive']
    ratio_mode = config['ratio_mode']
    if expected_inputs:
        # Classical belief-degree SRF reports the solution for the expected inputs.
        z_adjusted, e_adjusted = _build_belief_expected_inputs(
            context['cards_arrangement'], z_adjusted, e_adjusted
        )
        comp_rule_successive = 'interval-constrained'
        ratio_mode = 'interval-total'

    model, weights, rank_groups, _criteria_cards, _delta = _build_srf_model(
        context['cards_arrangement'],
        z_adjusted,
        e_adjusted,
        comp_rule_within=config['comp_rule_within'],
        comp_rule_successive=comp_rule_successive,
        ratio_mode=ratio_mode,
        normalized=config['normalized'],
        extra_cond=context['extra_constraints'],
        min_delta=context['min_delta'],
        launch_smaa=False,
        gap_overrides=gap_overrides or None,
        conditional_gap_milp=config['conditional_gap_milp'],
        dynamic_unit_weight=config['dynamic_unit_weight']
    )
    for name in drop_constraints:
        del model._problem.constraints[name]
    return model, weights, rank_groups


def _is_adjustment_feasible(context, adjustments):
    """
    Returns True when the weight calculation is feasible after applying `adjustments`.
    """
    model_variants = [False, True] if context['check_expected_model'] else [False]
    for expected_inputs in model_variants:
        context['remaining_checks'] -= 1
        try:
            model, _weights, _rank_groups = _build_adjusted_model(
                context, adjustments, expected_inputs=expected_inputs
            )
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            return False
        _optimize_model(model)
        if model.status != GRB.OPTIMAL:
            return False
    return True


def _ratio_pair_variables(pair, weights, rank_groups):
    """
    Returns the (numerator, denominator) weight variables of a ratio input.
    """
    if pair[0] == 'successive':
        rank = pair[1]
        return weights[rank_groups[rank + 1][0]], weights[rank_groups[rank][0]]

    sorted_ranks = sorted(rank_groups.keys())
    return weights[rank_groups[sorted_ranks[-1]][0]], weights[rank_groups[sorted_ranks[0]][0]]


def _ratio_extremes_linear(context, model, num_var, den_var):
    """
    Smallest and largest num/den over an LP feasible region via the
    Charnes-Cooper transformation y = t * x, t = 1 / den.
    """
    context['remaining_checks'] -= 2
    variables, A_ub, b_ub, A_eq, b_eq = _extract_freeopt_polytope(model)
    n_vars = len(variables)
    num_pos = next(idx for idx, var in enumerate(variables) if var is num_var)
    den_pos = next(idx for idx, var in enumerate(variables) if var is den_var)

    A_ub_cc = np.hstack([A_ub, -b_ub.reshape(-1, 1)])
    den_row = np.zeros((1, n_vars + 1))
    den_row[0, den_pos] = 1.0
    A_eq_cc = np.vstack([np.hstack([A_eq, -b_eq.reshape(-1, 1)]), den_row])
    b_eq_cc = np.zeros(len(A_eq_cc))
    b_eq_cc[-1] = 1.0
    bounds = [(None, None)] * n_vars + [(0, None)]

    extremes = []
    for sense in (1.0, -1.0):
        objective = np.zeros(n_vars + 1)
        objective[num_pos] = sense
        result = linprog(
            objective,
            A_ub=A_ub_cc if len(A_ub_cc) else None,
            b_ub=np.zeros(len(A_ub_cc)) if len(A_ub_cc) else None,
            A_eq=A_eq_cc,
            b_eq=b_eq_cc,
            bounds=bounds,
            method='highs'
        )
        if result.status == 2:
            return None
        if result.status == 3:
            extremes.append(-sense * np.inf)
        elif result.status == 0:
            extremes.append(sense * float(result.fun))
        else:
            return None
    return extremes[0], extremes[1]


def _ratio_extremes_by_bisection(context, model, num_var, den_var, iterations=24):
    """
    Smallest and largest num/den over a MILP feasible region by bisection on
    probe constraints. Both results are rounded towards the inside of the range.
    """
    probe_name = "ei_ratio_probe"

    def ratio_reachable(threshold, at_least):
        model._problem.constraints.pop(probe_name, None)
        probe = num_var - threshold * den_var
        model.addConstr(probe >= 0 if at_least else probe <= 0, probe_name)
        context['remaining_checks'] -= 1
        _optimize_model(model)
        return model.status == GRB.OPTIMAL

    context['remaining_checks'] -= 1
    _optimize_model(model)
    if model.status != GRB.OPTIMAL:
        return None

    if ratio_reachable(INCONSISTENCY_Z_MAX, at_least=True):
        ratio_max = np.inf
    else:
        low, high = 0.0, INCONSISTENCY_Z_MAX
        for _ in range(iterations):
            middle = (low + high) / 2.0
            if ratio_reachable(middle, at_least=True):
                low = middle
            else:
                high = middle
        ratio_max = low

    if ratio_reachable(0.0, at_least=False):
        ratio_min = 0.0
    elif not ratio_reachable(INCONSISTENCY_Z_MAX, at_least=False):
        ratio_min = np.inf
    else:
        low, high = 0.0, INCONSISTENCY_Z_MAX
        for _ in range(iterations):
            middle = (low + high) / 2.0
            if ratio_reachable(middle, at_least=False):
                high = middle
            else:
                low = middle
        ratio_min = high

    model._problem.constraints.pop(probe_name, None)
    return ratio_min, ratio_max


def _ratio_range_without(context, adjustments, constraints, pair, expected_inputs=False):
    """
    Returns the (min, max) ratio of `pair` in the adjusted model once the named
    ratio constraints are removed, or None when that model is infeasible.
    """
    try:
        model, weights, rank_groups = _build_adjusted_model(
            context, adjustments, drop_constraints=constraints, expected_inputs=expected_inputs
        )
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return None

    num_var, den_var = _ratio_pair_variables(pair, weights, rank_groups)
    if all(_lp_var_is_continuous(var) for var in model._vars):
        return _ratio_extremes_linear(context, model, num_var, den_var)
    return _ratio_extremes_by_bisection(context, model, num_var, den_var)


def _feasible_ratio_parameter_interval(context, adjustments, parameter):
    """
    Returns the (low, high) values of a ratio input that make the model feasible
    once `adjustments` are applied, restricted to the requested side of the
    current value. Returns None when no such value exists or when the current
    value already works (so this input does not need to change).
    """
    ratio = parameter['ratio']
    if ratio.get('kind') == 'expected':
        bound_constraints = ['z_ratio_constraint_min', 'z_ratio_constraint_max']
        support_range = _ratio_range_without(context, adjustments, bound_constraints, ('global',))
        expected_range = _ratio_range_without(
            context, adjustments, bound_constraints, ('global',), expected_inputs=True
        )
        if support_range is None or expected_range is None:
            return None
        expected = float(parameter['current'])
        # Rescaled to expected value v, the support becomes
        # [1 + low_factor * (v - 1), 1 + high_factor * (v - 1)].
        low_factor = (min(parameter['support_values']) - 1.0) / (expected - 1.0)
        high_factor = (max(parameter['support_values']) - 1.0) / (expected - 1.0)
        # The expected z must be attainable in the expected-input model, and the
        # rescaled support must overlap the attainable ratios of the support model.
        low = max(expected_range[0], 1.0 + (support_range[0] - 1.0) / high_factor)
        high = min(expected_range[1], 1.0 + (support_range[1] - 1.0) / low_factor)
    else:
        extremes = _ratio_range_without(context, adjustments, ratio['constraints'], ratio['pair'])
        if extremes is None:
            return None
        ratio_min, ratio_max = extremes

        # A lower bound on the ratio must not exceed its largest attainable value, an
        # upper bound must not fall below its smallest one, and an exact ratio must
        # lie between both.
        low = ratio_min if ratio['bound'] in {'eq', 'ub'} else -np.inf
        high = ratio_max if ratio['bound'] in {'eq', 'lb'} else np.inf
    current = float(parameter['current'])
    if parameter['direction'] == 'decrease':
        if high >= current - 1e-6:
            return None
        low = max(low, float(parameter['lower']))
    else:
        if low <= current + 1e-6:
            return None
        high = min(high, float(parameter['upper']))

    if low > high:
        return None
    return low, high


def _verified_ratio_bound(context, adjustments, key, value, inward_step, limit, attempts=3):
    """
    Returns `value`, or the first value moved inward by `inward_step`, that the
    exact model accepts. This absorbs LP round-off at the ends of a range.
    """
    for _ in range(attempts):
        if (inward_step > 0 and value > limit + 1e-12) or (inward_step < 0 and value < limit - 1e-12):
            return None
        candidate = float(round(value, 6))
        if _is_adjustment_feasible(context, {**adjustments, key: candidate}):
            return candidate
        value += inward_step
    return None


def _pick_verified_ratio_values(context, adjustments, parameter, low, high):
    """
    Rounds a feasible ratio range inward to display precision, verifies both shown
    bounds against the exact model, and picks an example value. The example is a
    one-decimal value (matching the input step) closest to the current input
    whenever the range allows it. Returns (suggested, shown_low, shown_high) or None.
    """
    key = parameter['key']
    slack = 1e-4
    for decimals in (2, 3, 4):
        scale = 10 ** decimals
        shown_low = float(np.ceil(low * scale - slack) / scale)
        shown_high = float(np.floor(high * scale + slack) / scale)
        if shown_low > shown_high:
            continue

        verified_low = _verified_ratio_bound(context, adjustments, key, shown_low, 1.0 / scale, shown_high)
        if verified_low is None:
            continue
        verified_high = _verified_ratio_bound(context, adjustments, key, shown_high, -1.0 / scale, verified_low)
        if verified_high is None:
            continue

        # Rescaled belief distributions are displayed with two decimals, so their
        # example keeps enough distance from the range ends to absorb that rounding.
        margin = INCONSISTENCY_DISTRIBUTION_ROUNDING_MARGIN if key[0] == 'z_scale' else 0.0
        example_low = verified_low + margin
        example_high = verified_high - margin
        if example_low > example_high:
            example_low = example_high = (verified_low + verified_high) / 2.0

        # Any value between two verified bounds is feasible: the feasible values
        # of one ratio input form an interval.
        if parameter['direction'] == 'decrease':
            one_decimal = float(np.floor(example_high * 10 + slack) / 10)
            fallback = max(verified_low, float(np.floor(example_high * scale) / scale))
        else:
            one_decimal = float(np.ceil(example_low * 10 - slack) / 10)
            fallback = min(verified_high, float(np.ceil(example_low * scale) / scale))
        if (example_low <= one_decimal <= example_high
                and _is_adjustment_feasible(context, {**adjustments, key: one_decimal})):
            return one_decimal, verified_low, verified_high
        return fallback, verified_low, verified_high

    return None


def _format_input_value(value, value_type):
    if value_type == 'int':
        return str(int(round(float(value))))
    return str(float(round(float(value), 4)))


def _adjusted_distribution_text(parameter, new_anchor):
    """
    Renders "a, b become c, d" for a belief distribution edited so that its
    anchor (expected z or smallest blank-card value) becomes `new_anchor`.
    """
    current = float(parameter['current'])
    if parameter['key'][0] == 'z_scale':
        factor = (float(new_anchor) - 1.0) / (current - 1.0)
        adjust = lambda value: 1.0 + (value - 1.0) * factor
    else:
        adjust = lambda value: value + float(new_anchor) - current

    value_type = parameter['value_type']
    old_values = ', '.join(_format_input_value(value, value_type) for value in parameter['support_values'])
    new_values = ', '.join(
        _format_input_value(round(adjust(value), 2), value_type) for value in parameter['support_values']
    )
    return f"{old_values} become {new_values}"


def _integer_recommendation(parameter, value, extended_value=None):
    """
    Builds the recommendation for an integer input moved to `value`. When
    `extended_value` is given, every value between both was verified as well.
    """
    low, high = sorted((value, value if extended_value is None else extended_value))
    current = int(parameter['current'])
    if parameter['key'][0] == 'e_shift':
        text = (
            f"{parameter['direction'].capitalize()} every {parameter['label']} by "
            f"{abs(value - current)} (values {_adjusted_distribution_text(parameter, value)}"
        )
        if high != low:
            smallest_shift, largest_shift = sorted((abs(low - current), abs(high - current)))
            text += f"; any shift from {smallest_shift} to {largest_shift} works"
        text += ")"
    else:
        text = (
            f"{parameter['direction'].capitalize()} {parameter['label']} from "
            f"{_format_input_value(current, 'int')} to {value}"
        )
        if high != low:
            text += f" (any value from {low} to {high} works)"
    return {
        'issue_id': parameter['issue_id'],
        'direction': parameter['direction'],
        'rank_pair': parameter['rank_pair'],
        'current_value': parameter['current'],
        'suggested_value': value,
        'feasible_min': low,
        'feasible_max': high,
        'recommendation': text + ".",
    }


def _ratio_recommendation(parameter, suggested, shown_low=None, shown_high=None):
    """
    Builds the recommendation for a ratio input moved to `suggested`, optionally
    stating the verified range [shown_low, shown_high] of values that also work.
    """
    fmt = lambda value: _format_input_value(value, 'float')
    text = (
        f"{parameter['direction'].capitalize()} {parameter['label']} from "
        f"{fmt(parameter['current'])} to {fmt(suggested)}"
    )
    notes = []
    if parameter['key'][0] == 'z_scale':
        text += " by rescaling all z values proportionally around 1"
        notes.append(f"z values {_adjusted_distribution_text(parameter, suggested)}")
    if shown_low is not None and shown_high is not None and shown_high > shown_low:
        if parameter['direction'] == 'decrease' and shown_low <= float(parameter['lower']) + 1e-9:
            notes.append(f"any value of {fmt(shown_high)} or lower works")
        elif parameter['direction'] == 'increase' and shown_high >= float(parameter['upper']) - 1e-9:
            notes.append(f"any value of {fmt(shown_low)} or higher works")
        else:
            notes.append(f"any value from {fmt(shown_low)} to {fmt(shown_high)} works")
    if notes:
        text += f" ({'; '.join(notes)})"
    return {
        'issue_id': parameter['issue_id'],
        'direction': parameter['direction'],
        'rank_pair': parameter['rank_pair'],
        'current_value': parameter['current'],
        'suggested_value': suggested,
        'feasible_min': shown_low,
        'feasible_max': shown_high,
        'recommendation': text + ".",
    }


def _iter_positive_compositions(total, n_parts):
    """
    Yields ordered compositions of `total` into `n_parts` strictly positive integers.
    """
    if n_parts == 1:
        if total >= 1:
            yield [total]
        return

    max_first = total - (n_parts - 1)
    for first in range(1, max_first + 1):
        for tail in _iter_positive_compositions(total - first, n_parts - 1):
            yield [first] + tail


def _iter_integer_adjustments(parameters):
    """
    Yields candidate values for integer inputs ordered by total change, with every
    input moved by at least one unit in its required direction.
    """
    if not parameters:
        yield ()
        return

    limits = []
    for parameter in parameters:
        if parameter['direction'] == 'decrease':
            limits.append(int(parameter['current']) - int(parameter['lower']))
        else:
            limits.append(int(parameter['upper']) - int(parameter['current']))
    if min(limits) < 1:
        return

    max_total = sum(limits)
    if len(parameters) > 1:
        max_total = min(max_total, len(parameters) + INCONSISTENCY_MAX_COMBINED_CHANGE)

    for total in range(len(parameters), max_total + 1):
        for deltas in _iter_positive_compositions(total, len(parameters)):
            if any(delta > limit for delta, limit in zip(deltas, limits)):
                continue
            yield tuple(
                int(parameter['current']) + (delta if parameter['direction'] == 'increase' else -delta)
                for parameter, delta in zip(parameters, deltas)
            )


def _extend_integer_range(context, parameter, value):
    """
    Continues past the nearest restoring value of a single integer input and
    returns the farthest value that is still verified as feasible.
    """
    step = 1 if parameter['direction'] == 'increase' else -1
    limit = parameter['upper'] if step > 0 else parameter['lower']
    extended = value
    for _ in range(INCONSISTENCY_RANGE_SCAN_STEPS):
        candidate = extended + step
        if (step > 0 and candidate > limit) or (step < 0 and candidate < limit):
            break
        if not _is_adjustment_feasible(context, {parameter['key']: candidate}):
            break
        extended = candidate
    return extended


def _restore_from_ratio_seeds(context, parameters, ratio_seeds):
    """
    Restoration for sets changing several ratio inputs at once. Their joint range
    has no closed form, so the ratios observed in the EI solution are rounded
    outward and checked against the exact model.
    """
    adjustments = {}
    for parameter in parameters:
        seed = ratio_seeds.get(parameter['key'])
        if seed is None or not np.isfinite(seed):
            return None
        bound = parameter['ratio']['bound']
        if bound == 'lb':
            value = float(np.floor(seed * 100) / 100)
        elif bound == 'ub':
            value = float(np.ceil(seed * 100) / 100)
        else:
            value = float(round(seed, 3))

        if parameter['direction'] == 'decrease':
            value = max(value, float(parameter['lower']))
            if value >= float(parameter['current']):
                return None
        else:
            value = min(value, float(parameter['upper']))
            if value <= float(parameter['current']):
                return None
        adjustments[parameter['key']] = value

    if not _is_adjustment_feasible(context, adjustments):
        return None
    return [_ratio_recommendation(parameter, adjustments[parameter['key']]) for parameter in parameters]


def _search_restoration(context, parameters, ratio_seeds):
    """
    Finds the smallest verified change of `parameters` that makes the weight
    calculation feasible. Returns its recommendations, or None.
    """
    integer_parameters = [parameter for parameter in parameters if parameter['ratio'] is None]
    ratio_parameters = [parameter for parameter in parameters if parameter['ratio'] is not None]
    if len(ratio_parameters) > 1:
        if integer_parameters:
            return None
        return _restore_from_ratio_seeds(context, ratio_parameters, ratio_seeds)
    ratio_parameter = ratio_parameters[0] if ratio_parameters else None
    stop_below = max(0, context['remaining_checks'] - INCONSISTENCY_MAX_CHECKS_PER_SET)

    for integer_values in _iter_integer_adjustments(integer_parameters):
        if context['remaining_checks'] <= stop_below:
            return None
        adjustments = {
            parameter['key']: value
            for parameter, value in zip(integer_parameters, integer_values)
        }

        if ratio_parameter is None:
            if not _is_adjustment_feasible(context, adjustments):
                continue
            if len(integer_parameters) == 1:
                extended = _extend_integer_range(context, integer_parameters[0], integer_values[0])
                return [_integer_recommendation(integer_parameters[0], integer_values[0], extended)]
            return [
                _integer_recommendation(parameter, value)
                for parameter, value in zip(integer_parameters, integer_values)
            ]

        interval = _feasible_ratio_parameter_interval(context, adjustments, ratio_parameter)
        if interval is None:
            continue
        picked = _pick_verified_ratio_values(context, adjustments, ratio_parameter, *interval)
        if picked is None:
            continue
        suggested, shown_low, shown_high = picked
        recommendations = [
            _integer_recommendation(parameter, value)
            for parameter, value in zip(integer_parameters, integer_values)
        ]
        recommendations.append(_ratio_recommendation(ratio_parameter, suggested, shown_low, shown_high))
        return recommendations

    return None


def _merge_issue_parameters(active_issues, issue_meta):
    """
    Collects the distinct inputs changed by one EI issue set, or returns None when
    the set cannot be turned into one consistent edit of the user inputs.
    """
    parameters = {}
    for issue_id in active_issues:
        parameter = issue_meta[issue_id].get('parameter')
        if parameter is None:
            return None
        existing = parameters.get(parameter['key'])
        if existing is None:
            parameters[parameter['key']] = {**parameter, 'issue_id': issue_id}
        elif existing['direction'] != parameter['direction']:
            return None
    return list(parameters.values())


def identify_inconsistency_recommendations(cards_arrangement,
                                           z_value,
                                           e_value,
                                           srf_method,
                                           extra_constraints=None,
                                           min_delta=1.0,
                                           max_suggestions=3,
                                           modular_options=None,
                                           modular_profile=None):
    """
    Iterative E^I-style inconsistency identification with verified restoration.

    A MILP relaxation proposes minimal-cardinality sets of inputs to change. Each
    proposed set is then searched against the exact model used by the weight
    calculation, and only changes that make that model feasible are reported, so
    applying any one suggestion restores consistency in a single step. Rejected
    sets do not use up suggestion slots. Returns up to `max_suggestions` alternatives.
    """
    max_suggestions = int(max_suggestions) if str(max_suggestions).strip() != '' else 3
    max_suggestions = max(1, min(max_suggestions, 20))

    config = _resolve_model_configuration(
        srf_method,
        modular_options=modular_options,
        modular_profile=modular_profile
    )
    comp_rule_within = config['comp_rule_within']
    ratio_mode = config['ratio_mode']
    extra_cond = extra_constraints if isinstance(extra_constraints, dict) else None

    # With dynamic unit weights, imprecise gap inputs only keep the minimum
    # blank-card spacing (see `_build_srf_model`), i.e. the fully-flexible rule.
    if config['dynamic_unit_weight'] and config['comp_rule_successive'] in {
        'interval-constrained',
        'probability-distribution',
        'hfl-linguistic-interval',
    }:
        gap_rule = 'fully-flexible'
    else:
        gap_rule = config['comp_rule_successive']

    model = gp.Model("SRF_Inconsistency_Identification")
    model.setParam("OutputFlag", 0)

    criteria_cards = cards_arrangement[cards_arrangement['class'] == 'criterion'].sort_values('rank')
    weights = {
        idx: model.addVar(lb=0, name=f"w_{idx}")
        for idx in criteria_cards.index
    }

    # Shared scale variable used for fixed/interval spacing styles.
    if gap_rule in ['fixed-spacing', 'interval-constrained', 'probability-distribution']:
        spacing_scale = model.addVar(lb=max(float(min_delta), 1e-6), name="ei_spacing_scale")
    elif gap_rule == 'hfl-linguistic-interval':
        spacing_scale = model.addVar(lb=1e-6, name="ei_hfl_scale")
    else:
        spacing_scale = None

    rank_white_count = {}
    for rank in cards_arrangement['rank'].unique():
        rank_white_count[rank] = cards_arrangement[cards_arrangement['rank'] == rank]['class'].to_list().count('white') + 1

    rank_groups = {}
    for rank in criteria_cards['rank'].unique():
        rank_groups[rank] = criteria_cards[criteria_cards['rank'] == rank].index.tolist()
    sorted_ranks = sorted(rank_groups.keys())

    # Hard constraints: within-rank equalities.
    if comp_rule_within == 'equal':
        for rank, indices in rank_groups.items():
            for i in range(1, len(indices)):
                model.addConstr(weights[indices[0]] == weights[indices[i]], f"ei_equal_within_rank_{rank}_{i}")

    issue_vars = {}
    issue_meta = {}

    # Classical belief-degree SRF reports the solution of the expected inputs, so
    # each distribution is edited as a whole (z values rescaled around 1,
    # blank-card values shifted).
    expected_model_check = srf_method == 'belief_degree_imprecise_srf'
    z_scale_parameters = _z_scale_parameters(z_value) if expected_model_check else (None, None)
    e_shift_parameters = {}

    # Successive-rank constraints as relaxable EI issues. A decrease is capped at
    # the smallest admissible input (e.g. zero blank cards), so the relaxation
    # never relies on a change the user cannot make; an input already at its
    # limit keeps its constraint hard.
    for i in range(1, len(sorted_ranks)):
        prev_rank = sorted_ranks[i - 1]
        curr_rank = sorted_ranks[i]
        prev_idx = rank_groups[prev_rank][0]
        curr_idx = rank_groups[curr_rank][0]
        rank_pair = [int(prev_rank), int(curr_rank)]
        pair_label = f"Rank {prev_rank} and Rank {curr_rank}"
        units = int(rank_white_count[prev_rank])
        observed_gap = units - 1
        diff_expr = weights[curr_idx] - weights[prev_idx]
        if expected_model_check:
            e_shift_parameters[int(prev_rank)] = _e_shift_parameters(e_value, prev_rank, pair_label, rank_pair)

        e_cloud = {}
        if gap_rule == 'probability-distribution' and isinstance(e_value, dict):
            e_cloud = _extract_probability_pairs(
                e_value,
                value_prefix=f"e-value-{prev_rank}-",
                beta_prefix=f"e-beta-{prev_rank}-",
            )
        has_interval_bounds = (
            gap_rule == 'interval-constrained'
            and isinstance(e_value, dict)
            and f'emin_{prev_rank}' in e_value
            and f'emax_{prev_rank}' in e_value
        )

        if gap_rule == 'fully-flexible':
            if units > 1:
                _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"gap_{prev_rank}_min",
                    expr=diff_expr - (min_delta * units),
                    bound='lb',
                    metadata={'parameter': _input_parameter(
                        ('gap', int(prev_rank)),
                        f"minimum blank cards gap between {pair_label}",
                        observed_gap, 'decrease', lower=0, rank_pair=rank_pair
                    )},
                    residual_cap=min_delta * observed_gap
                )
            else:
                model.addConstr(diff_expr >= min_delta * units, f"ei_hard_gap_{prev_rank}_min")

        elif has_interval_bounds or e_cloud:
            if has_interval_bounds:
                e_min = int(e_value[f'emin_{prev_rank}'])
                e_max = int(e_value[f'emax_{prev_rank}'])
                issue_prefix = "gap"
                min_parameter = _input_parameter(
                    ('e_key', f'emin_{prev_rank}'), f"minimum blank cards between {pair_label}",
                    e_min, 'decrease', lower=0, rank_pair=rank_pair
                ) if e_min > 0 else None
                max_parameter = _input_parameter(
                    ('e_key', f'emax_{prev_rank}'), f"maximum blank cards between {pair_label}",
                    e_max, 'increase', upper=e_max + INCONSISTENCY_MAX_GAP_INCREASE, rank_pair=rank_pair
                )
            else:
                e_cloud = _normalize_probability_cloud(e_cloud)
                e_min = int(np.floor(min(e_cloud.keys())))
                e_max = int(np.ceil(max(e_cloud.keys())))
                issue_prefix = "gap_support"
                if expected_model_check:
                    min_parameter, max_parameter = e_shift_parameters[int(prev_rank)]
                else:
                    min_parameter = _input_parameter(
                        ('e_support', int(prev_rank), 'min'),
                        f"smallest blank-card value in the belief distribution between {pair_label}",
                        e_min, 'decrease', lower=0, rank_pair=rank_pair
                    ) if e_min > 0 else None
                    max_parameter = _input_parameter(
                        ('e_support', int(prev_rank), 'max'),
                        f"largest blank-card value in the belief distribution between {pair_label}",
                        e_max, 'increase', upper=e_max + INCONSISTENCY_MAX_GAP_INCREASE, rank_pair=rank_pair
                    )

            if min_parameter is not None:
                _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"{issue_prefix}_min_{prev_rank}",
                    expr=diff_expr - spacing_scale * (e_min + 1),
                    bound='lb',
                    metadata={'parameter': min_parameter},
                    residual_cap=spacing_scale * e_min
                )
            else:
                model.addConstr(diff_expr >= spacing_scale * (e_min + 1), f"ei_hard_{issue_prefix}_min_{prev_rank}")
            if max_parameter is not None:
                _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"{issue_prefix}_max_{prev_rank}",
                    expr=diff_expr - spacing_scale * (e_max + 1),
                    bound='ub',
                    metadata={'parameter': max_parameter}
                )
            else:
                model.addConstr(diff_expr <= spacing_scale * (e_max + 1), f"ei_hard_{issue_prefix}_max_{prev_rank}")

        elif gap_rule == 'hfl-linguistic-interval':
            r_min_term = int(e_value.get(f'rmin_{prev_rank}', 1))
            r_max_term = int(e_value.get(f'rmax_{prev_rank}', r_min_term))
            r_min = _map_hfl_card_term(r_min_term)
            r_max = _map_hfl_card_term(r_max_term)

            if r_min > HFL_CARD_MIN_TERM:
                _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"hfl_gap_min_{prev_rank}",
                    expr=diff_expr - spacing_scale * r_min,
                    bound='lb',
                    metadata={'parameter': _input_parameter(
                        ('e_key', f'rmin_{prev_rank}'),
                        f"HFL lower gap term between {pair_label}",
                        r_min_term, 'decrease', lower=HFL_CARD_MIN_TERM, rank_pair=rank_pair
                    )},
                    residual_cap=spacing_scale * (r_min - HFL_CARD_MIN_TERM)
                )
            else:
                model.addConstr(diff_expr >= spacing_scale * r_min, f"ei_hard_hfl_gap_min_{prev_rank}")
            if r_max < HFL_CARD_MAX_TERM:
                _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"hfl_gap_max_{prev_rank}",
                    expr=diff_expr - spacing_scale * r_max,
                    bound='ub',
                    metadata={'parameter': _input_parameter(
                        ('e_key', f'rmax_{prev_rank}'),
                        f"HFL upper gap term between {pair_label}",
                        r_max_term, 'increase', upper=HFL_CARD_MAX_TERM, rank_pair=rank_pair
                    )},
                    residual_cap=spacing_scale * (HFL_CARD_MAX_TERM - r_max)
                )
            else:
                model.addConstr(diff_expr <= spacing_scale * r_max, f"ei_hard_hfl_gap_max_{prev_rank}")

        else:
            # Exact spacing: fixed-spacing, or interval/probability rules without
            # explicit bounds for this gap.
            label = f"blank cards between {pair_label}"
            expr = diff_expr - spacing_scale * units
            y_increase = _add_relaxable_issue(
                model, issue_vars, issue_meta,
                issue_id=f"gap_{prev_rank}_minus",
                expr=expr,
                bound='ub',
                metadata={'parameter': _input_parameter(
                    ('gap', int(prev_rank)), label, observed_gap, 'increase',
                    upper=observed_gap + INCONSISTENCY_MAX_GAP_INCREASE, rank_pair=rank_pair
                )}
            )
            if units > 1:
                y_decrease = _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"gap_{prev_rank}_plus",
                    expr=expr,
                    bound='lb',
                    metadata={'parameter': _input_parameter(
                        ('gap', int(prev_rank)), label, observed_gap, 'decrease',
                        lower=0, rank_pair=rank_pair
                    )},
                    residual_cap=spacing_scale * observed_gap
                )
                model.addConstr(y_decrease + y_increase <= 1, f"ei_gap_{prev_rank}_exclusive")
            else:
                model.addConstr(expr >= 0, f"ei_hard_gap_{prev_rank}_plus")

    # Ratio issues
    min_index = rank_groups[min(sorted_ranks)][0]
    max_index = rank_groups[max(sorted_ranks)][0]
    min_weight_var = weights[min_index]
    max_weight_var = weights[max_index]
    global_pair = ('global',)

    if ratio_mode == 'exact-ratio':
        z_exact = float(z_value)
        ratio = {'constraints': ['z_ratio_constraint'], 'pair': global_pair, 'bound': 'eq'}
        y_decrease = _add_ratio_issue(
            model, issue_vars, issue_meta, "z_exact_plus",
            max_weight_var, min_weight_var, z_exact, 'lb',
            _input_parameter(('z',), "z ratio", z_exact, 'decrease', value_type='float',
                             lower=INCONSISTENCY_Z_MIN, ratio=ratio),
            floor_value=INCONSISTENCY_Z_MIN
        )
        y_increase = _add_ratio_issue(
            model, issue_vars, issue_meta, "z_exact_minus",
            max_weight_var, min_weight_var, z_exact, 'ub',
            _input_parameter(('z',), "z ratio", z_exact, 'increase', value_type='float',
                             upper=INCONSISTENCY_Z_MAX, ratio=ratio),
            ceiling_value=INCONSISTENCY_Z_MAX
        )
        if y_decrease is not None and y_increase is not None:
            model.addConstr(y_decrease + y_increase <= 1, "ei_z_exact_exclusive")

    elif ratio_mode == 'linear-spacing':
        e0_anchor = _estimate_linear_spacing_e0_anchor(e_value)
        bar_sum = (
            (cards_arrangement['rank'].max() - 1)
            + cards_arrangement['class'].to_list().count('white')
        )
        z_linear = (bar_sum + (e0_anchor + 1)) / (e0_anchor + 1)
        raise_e0, lower_e0 = _linear_spacing_e0_parameters(e_value)
        y_decrease = None
        y_increase = None
        if raise_e0 is not None:
            y_decrease = _add_ratio_issue(
                model, issue_vars, issue_meta, "z_linear_plus",
                max_weight_var, min_weight_var, z_linear, 'lb', raise_e0,
                floor_value=1.0
            )
        else:
            model.addConstr(max_weight_var - z_linear * min_weight_var >= 0, "ei_hard_z_linear_plus")
        if lower_e0 is not None:
            y_increase = _add_ratio_issue(
                model, issue_vars, issue_meta, "z_linear_minus",
                max_weight_var, min_weight_var, z_linear, 'ub', lower_e0,
                ceiling_value=float(bar_sum + 1)
            )
        else:
            model.addConstr(max_weight_var - z_linear * min_weight_var <= 0, "ei_hard_z_linear_minus")
        if y_decrease is not None and y_increase is not None:
            model.addConstr(y_decrease + y_increase <= 1, "ei_z_linear_exclusive")

    elif ratio_mode in {'interval-total', 'probability-cloud'}:
        if ratio_mode == 'interval-total':
            z_min = float(z_value['zmin'])
            z_max = float(z_value['zmax'])
            min_key, max_key = ('z_key', 'zmin'), ('z_key', 'zmax')
            min_label, max_label = "z lower bound", "z upper bound"
        else:
            z_cloud = _normalize_probability_cloud(_extract_probability_pairs(
                z_value,
                value_prefix='z-value-',
                beta_prefix='z-beta-'
            ))
            z_min = float(min(z_cloud.keys()))
            z_max = float(max(z_cloud.keys()))
            min_key, max_key = ('z_support', 'min'), ('z_support', 'max')
            min_label = "smallest z value in the belief distribution"
            max_label = "largest z value in the belief distribution"

        min_parameter = _input_parameter(
            min_key, min_label, z_min, 'decrease', value_type='float', lower=INCONSISTENCY_Z_MIN,
            ratio={'constraints': ['z_ratio_constraint_min'], 'pair': global_pair, 'bound': 'lb'}
        )
        max_parameter = _input_parameter(
            max_key, max_label, z_max, 'increase', value_type='float', upper=INCONSISTENCY_Z_MAX,
            ratio={'constraints': ['z_ratio_constraint_max'], 'pair': global_pair, 'bound': 'ub'}
        )
        if expected_model_check:
            min_parameter, max_parameter = z_scale_parameters

        _add_ratio_issue(
            model, issue_vars, issue_meta, "z_interval_min",
            max_weight_var, min_weight_var, z_min, 'lb', min_parameter,
            floor_value=INCONSISTENCY_Z_MIN
        )
        _add_ratio_issue(
            model, issue_vars, issue_meta, "z_interval_max",
            max_weight_var, min_weight_var, z_max, 'ub', max_parameter,
            ceiling_value=INCONSISTENCY_Z_MAX
        )

    elif ratio_mode == 'interval-successive':
        for rank in range(1, cards_arrangement['rank'].max()):
            pair = ('successive', rank)
            num_var, den_var = _ratio_pair_variables(pair, weights, rank_groups)
            z_min = float(z_value[f'zmin_{rank}'])
            z_max = float(z_value[f'zmax_{rank}'])
            _add_ratio_issue(
                model, issue_vars, issue_meta, f"z_successive_min_{rank}",
                num_var, den_var, z_min, 'lb',
                _input_parameter(('z_key', f'zmin_{rank}'), f"z lower bound for Rank {rank + 1} / Rank {rank}",
                                 z_min, 'decrease', value_type='float', lower=INCONSISTENCY_Z_MIN,
                                 rank_pair=[rank, rank + 1],
                                 ratio={'constraints': [f'z_ratio_constraint_min_{rank}'], 'pair': pair, 'bound': 'lb'}),
                floor_value=INCONSISTENCY_Z_MIN
            )
            _add_ratio_issue(
                model, issue_vars, issue_meta, f"z_successive_max_{rank}",
                num_var, den_var, z_max, 'ub',
                _input_parameter(('z_key', f'zmax_{rank}'), f"z upper bound for Rank {rank + 1} / Rank {rank}",
                                 z_max, 'increase', value_type='float', upper=INCONSISTENCY_Z_MAX,
                                 rank_pair=[rank, rank + 1],
                                 ratio={'constraints': [f'z_ratio_constraint_max_{rank}'], 'pair': pair, 'bound': 'ub'}),
                ceiling_value=INCONSISTENCY_Z_MAX
            )

    elif ratio_mode == 'hfl-ratio-interval' and isinstance(z_value, dict):
        min_key = 'emin' if 'emin' in z_value else 'zmin'
        max_key = 'emax' if 'emax' in z_value else 'zmax'
        z_min_term = int(z_value.get(min_key, HFL_Z_MIN_TERM))
        z_max_term = int(z_value.get(max_key, z_min_term))
        z_min = _map_hfl_z_term(z_min_term)
        z_max = _map_hfl_z_term(z_max_term)

        _add_ratio_issue(
            model, issue_vars, issue_meta, "hfl_z_min",
            max_weight_var, min_weight_var, z_min, 'lb',
            _input_parameter(('z_key', min_key), "HFL lower z term", z_min_term, 'decrease',
                             lower=HFL_Z_MIN_TERM),
            floor_value=HFL_Z_MIN_TERM
        )
        _add_ratio_issue(
            model, issue_vars, issue_meta, "hfl_z_max",
            max_weight_var, min_weight_var, z_max, 'ub',
            _input_parameter(('z_key', max_key), "HFL upper z term", z_max_term, 'increase',
                             upper=HFL_Z_MAX_TERM),
            ceiling_value=HFL_Z_MAX_TERM
        )

    if expected_model_check:
        _add_expected_belief_issues(
            model, issue_vars, issue_meta,
            cards_arrangement, z_value, e_value,
            criteria_cards, rank_white_count, comp_rule_within,
            config['normalized'], extra_cond, min_delta,
            z_scale_parameters, e_shift_parameters
        )

    # Hard normalization.
    if config['normalized']:
        model.addConstr(gp.quicksum(weights.values()) == 100, "ei_normalization")

    # Keep optional requirements hard in EI analysis when enabled.
    if extra_cond is not None:
        _add_optional_extra_constraints(model, weights, criteria_cards, extra_cond)

    if not issue_vars:
        return {
            'detected': False,
            'message': 'No relaxable inconsistency checks are defined for the selected method.',
            'requested_suggestions': max_suggestions,
            'suggestions': []
        }

    issue_cardinality_expr = gp.quicksum(issue_vars.values())
    residual_expr = gp.quicksum(
        meta['residual_var'] for meta in issue_meta.values()
        if meta.get('residual_var') is not None
    )
    model.setObjective(
        INCONSISTENCY_CARDINALITY_WEIGHT * issue_cardinality_expr + residual_expr,
        GRB.MINIMIZE
    )

    context = {
        'cards_arrangement': cards_arrangement,
        'z_value': z_value,
        'e_value': e_value,
        'config': config,
        'extra_constraints': extra_cond,
        'min_delta': min_delta,
        'check_expected_model': expected_model_check,
        'remaining_checks': INCONSISTENCY_MAX_FEASIBILITY_CHECKS,
    }

    suggestions = []
    restoring_parameter_sets = []
    for idx in range(INCONSISTENCY_MAX_CANDIDATE_SETS):
        if len(suggestions) >= max_suggestions or context['remaining_checks'] <= 0:
            break

        _optimize_model(model)
        if model.status != GRB.OPTIMAL:
            break

        active_issues = sorted(
            issue_id for issue_id, var in issue_vars.items()
            if var.X > 0.5
        )
        if not active_issues:
            break

        recommendations = None
        parameters = _merge_issue_parameters(active_issues, issue_meta)
        if parameters is not None:
            parameter_keys = frozenset(parameter['key'] for parameter in parameters)
            already_covered = any(keys <= parameter_keys for keys in restoring_parameter_sets)
            if not already_covered:
                ratio_seeds = {}
                for parameter in parameters:
                    if parameter['ratio'] is None or 'pair' not in parameter['ratio']:
                        continue
                    num_var, den_var = _ratio_pair_variables(parameter['ratio']['pair'], weights, rank_groups)
                    if den_var.X > 1e-9:
                        ratio_seeds[parameter['key']] = num_var.X / den_var.X
                recommendations = _search_restoration(context, parameters, ratio_seeds)

        if recommendations:
            suggestions.append({
                'suggestion_id': len(suggestions) + 1,
                'minimal_changes': len(recommendations),
                'recommendations': recommendations
            })
            restoring_parameter_sets.append(parameter_keys)
            # Any superset of a restoring set is not minimal.
            model.addConstr(
                gp.quicksum(issue_vars[issue_id] for issue_id in active_issues) <= len(active_issues) - 1,
                f"ei_nogood_{idx + 1}"
            )
        else:
            # Only this exact set is ruled out: a larger set containing these
            # changes may still restore consistency.
            model.addConstr(
                gp.quicksum(issue_vars[issue_id] for issue_id in active_issues)
                - gp.quicksum(var for issue_id, var in issue_vars.items() if issue_id not in active_issues)
                <= len(active_issues) - 1,
                f"ei_reject_{idx + 1}"
            )

    return {
        'detected': len(suggestions) > 0,
        'message': (
            "Input preferences are inconsistent. "
            "Apply all changes of one suggestion below, keep the other inputs unchanged, and re-run."
            if suggestions else
            "No actionable inconsistency recommendation could be generated."
        ),
        'requested_suggestions': max_suggestions,
        'returned_suggestions': len(suggestions),
        'minimal_inconsistency_size': (
            min(suggestion['minimal_changes'] for suggestion in suggestions)
            if suggestions else None
        ),
        'suggestions': suggestions
    }


def _extract_gap_unit_bounds(sorted_ranks, rank_white_count, e_value, comp_rule_successive):
    """
    Extract integer unit-gap bounds for each successive rank gap.

    Unit scale:
      - interval/probability spacing uses e + 1
      - HFL spacing uses linguistic term index r
    """
    gap_bounds = {}
    for i in range(1, len(sorted_ranks)):
        prev_rank = sorted_ranks[i - 1]
        default_units = int(max(1, rank_white_count.get(prev_rank, 1)))

        lower_units = default_units
        upper_units = default_units

        if comp_rule_successive == 'interval-constrained' and isinstance(e_value, dict):
            e_min_key = f'emin_{prev_rank}'
            e_max_key = f'emax_{prev_rank}'
            if e_min_key in e_value and e_max_key in e_value:
                lower_units = int(np.floor(float(e_value[e_min_key]))) + 1
                upper_units = int(np.ceil(float(e_value[e_max_key]))) + 1

        elif comp_rule_successive == 'probability-distribution' and isinstance(e_value, dict):
            cloud = _extract_probability_pairs(
                e_value,
                value_prefix=f"e-value-{prev_rank}-",
                beta_prefix=f"e-beta-{prev_rank}-",
            )
            if cloud:
                lower_units = int(np.floor(min(cloud.keys()))) + 1
                upper_units = int(np.ceil(max(cloud.keys()))) + 1

        elif comp_rule_successive == 'hfl-linguistic-interval' and isinstance(e_value, dict):
            r_min_term = int(e_value.get(f'rmin_{prev_rank}', 1))
            r_max_term = int(e_value.get(f'rmax_{prev_rank}', r_min_term))
            lower_units = _map_hfl_card_term(r_min_term)
            upper_units = _map_hfl_card_term(r_max_term)

        lower_units = max(1, int(lower_units))
        upper_units = max(lower_units, int(upper_units))
        gap_bounds[int(prev_rank)] = (lower_units, upper_units)

    return gap_bounds


def _add_conditional_gap_order_constraints(model,
                                           weights,
                                           rank_groups,
                                           sorted_ranks,
                                           gap_unit_bounds,
                                           min_delta=1.0):
    """
    Add MILP constraints linking uncertain gap units to robust ordering of
    successive weight differences.
    """
    if len(sorted_ranks) < 3:
        return

    min_delta = float(max(min_delta, 1e-9))
    big_m = float(max(INCONSISTENCY_BIG_M, 1000.0))

    gap_diff_expr = {}
    for i in range(1, len(sorted_ranks)):
        prev_rank = int(sorted_ranks[i - 1])
        curr_rank = sorted_ranks[i]
        prev_idx = rank_groups[prev_rank][0]
        curr_idx = rank_groups[curr_rank][0]
        gap_diff_expr[prev_rank] = weights[curr_idx] - weights[prev_idx]

    gap_unit_expr = {}
    for prev_rank, (lower_units, upper_units) in gap_unit_bounds.items():
        if lower_units < upper_units:
            gap_unit_expr[prev_rank] = model.addVar(
                lb=lower_units,
                ub=upper_units,
                vtype=GRB.INTEGER,
                name=f"gap_units_{prev_rank}"
            )
        else:
            gap_unit_expr[prev_rank] = float(lower_units)

    gap_ids = list(gap_diff_expr.keys())
    for i in range(len(gap_ids) - 1):
        left_gap = gap_ids[i]
        for j in range(i + 1, len(gap_ids)):
            right_gap = gap_ids[j]
            left_lo, left_hi = gap_unit_bounds[left_gap]
            right_lo, right_hi = gap_unit_bounds[right_gap]

            can_left_gt = left_hi > right_lo
            can_right_gt = right_hi > left_lo
            can_equal = not (left_hi < right_lo or right_hi < left_lo)

            left_diff = gap_diff_expr[left_gap]
            right_diff = gap_diff_expr[right_gap]
            left_units = gap_unit_expr[left_gap]
            right_units = gap_unit_expr[right_gap]

            if can_left_gt and not can_right_gt and not can_equal:
                model.addConstr(
                    left_diff - right_diff >= min_delta,
                    f"gap_rel_det_left_gt_{left_gap}_{right_gap}"
                )
                continue
            if can_right_gt and not can_left_gt and not can_equal:
                model.addConstr(
                    right_diff - left_diff >= min_delta,
                    f"gap_rel_det_right_gt_{left_gap}_{right_gap}"
                )
                continue
            if can_equal and not can_left_gt and not can_right_gt:
                model.addConstr(
                    left_diff == right_diff,
                    f"gap_rel_det_equal_{left_gap}_{right_gap}"
                )
                continue

            selectors = []
            b_left_gt = None
            b_right_gt = None
            b_equal = None

            if can_left_gt:
                b_left_gt = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"gap_rel_left_gt_{left_gap}_{right_gap}"
                )
                selectors.append(b_left_gt)
            if can_right_gt:
                b_right_gt = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"gap_rel_right_gt_{left_gap}_{right_gap}"
                )
                selectors.append(b_right_gt)
            if can_equal:
                b_equal = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"gap_rel_equal_{left_gap}_{right_gap}"
                )
                selectors.append(b_equal)

            if not selectors:
                continue

            model.addConstr(
                gp.quicksum(selectors) == 1,
                f"gap_rel_select_one_{left_gap}_{right_gap}"
            )

            if b_left_gt is not None:
                model.addConstr(
                    left_units - right_units >= 1 - big_m * (1 - b_left_gt),
                    f"gap_units_left_gt_{left_gap}_{right_gap}"
                )
                model.addConstr(
                    left_diff - right_diff >= min_delta - big_m * (1 - b_left_gt),
                    f"gap_weights_left_gt_{left_gap}_{right_gap}"
                )

            if b_right_gt is not None:
                model.addConstr(
                    right_units - left_units >= 1 - big_m * (1 - b_right_gt),
                    f"gap_units_right_gt_{left_gap}_{right_gap}"
                )
                model.addConstr(
                    right_diff - left_diff >= min_delta - big_m * (1 - b_right_gt),
                    f"gap_weights_right_gt_{left_gap}_{right_gap}"
                )

            if b_equal is not None:
                model.addConstr(
                    left_units - right_units <= big_m * (1 - b_equal),
                    f"gap_units_equal_ub1_{left_gap}_{right_gap}"
                )
                model.addConstr(
                    right_units - left_units <= big_m * (1 - b_equal),
                    f"gap_units_equal_ub2_{left_gap}_{right_gap}"
                )
                model.addConstr(
                    left_diff - right_diff <= big_m * (1 - b_equal),
                    f"gap_weights_equal_ub1_{left_gap}_{right_gap}"
                )
                model.addConstr(
                    right_diff - left_diff <= big_m * (1 - b_equal),
                    f"gap_weights_equal_ub2_{left_gap}_{right_gap}"
                )


def _build_srf_model(cards_arrangement,
                     z_value,
                     e_value,
                     comp_rule_within,
                     comp_rule_successive,
                     ratio_mode,
                     normalized,
                     extra_cond=None,
                     min_delta=1.0,
                     launch_smaa=False,
                     gap_overrides=None,
                     conditional_gap_milp=False,
                     dynamic_unit_weight=False):
    """
    Helper function to build the SRF LP model with a free MILP solver (CBC via PuLP).

    This function encapsulates the common model building logic that can be flexibly reused by other functions.

    Args:
        cards_arrangement (pd.DataFrame): Preprocessed card arrangement data
        z_value: Ratio between first and last rank or successive ranks
        e_value: Spacing between cards
        comp_rule_within (str): Rule for comparing weights within an ex aequo set
        comp_rule_successive (str): Rule for comparing weights between successive sets
        ratio_mode (float): Target ratio between most and least important criteria
        normalized (bool): Whether to normalize weights to sum to 100
        extra_cond (callable, optional): Additional constraints function
        min_delta (float): Minimum difference between successive rank weights
        gap_overrides (dict, optional): Mapping prev_rank -> blank-card count override.
        conditional_gap_milp (bool): Whether to add conditional robust-imprecise gap ordering constraints.
        dynamic_unit_weight (bool): Whether to allow rank-pair-specific gap-scale variables.

    Returns:
        tuple: (model, weights, rank_groups, delta) - The optimization model, variables and related data
    """

    """
    MODEL INITIALIZATION AND INPUT PREPARATION
    """

    # Filter criteria cards
    criteria_cards = cards_arrangement[cards_arrangement['class'] == 'criterion'].sort_values('rank')

    # Create the optimization model
    model = gp.Model("SRF_Weights")
    model.setParam("OutputFlag", 0)  # Suppress output

    # Create variables for weights (k_r)
    weights = {}
    for idx in criteria_cards.index:
        # For finding feasible solutions, no objective coefficient needed
        weights[idx] = model.addVar(lb=0, name=f"weight_{idx}")

    use_dynamic_numeric_scale = bool(
        dynamic_unit_weight
        and comp_rule_successive in {'interval-constrained', 'probability-distribution'}
    )
    use_dynamic_hfl_scale = bool(
        dynamic_unit_weight and comp_rule_successive == 'hfl-linguistic-interval'
    )

    if comp_rule_successive in ['fixed-spacing', 'interval-constrained', 'probability-distribution'] and not use_dynamic_numeric_scale:
        # Fixed-C model: one shared spacing scale across all rank gaps.
        delta = model.addVar(lb=min_delta, name="delta")
    else:
        delta = None

    # HFL-SRF: linguistic scaling variable and objective variable (Model I)
    if comp_rule_successive == 'hfl-linguistic-interval':
        epsilon = model.addVar(lb=0, name="hfl_epsilon")
        if use_dynamic_hfl_scale:
            t_scale = None
        else:
            t_scale = model.addVar(lb=1e-6, name="hfl_t")
            model.addConstr(t_scale >= epsilon, "hfl_t_ge_epsilon")
    else:
        t_scale = None
        epsilon = None

    # Prepare for e_r calculation (white cards per rank)
    rank_white_count = {}
    for rank in cards_arrangement['rank'].unique():
        rank_white_count[rank] = cards_arrangement[cards_arrangement['rank'] == rank]['class'].to_list().count('white') + 1

    if isinstance(gap_overrides, dict):
        for rank_key, blank_cards in gap_overrides.items():
            try:
                rank_int = int(rank_key)
                blank_int = int(blank_cards)
            except (TypeError, ValueError):
                continue
            if rank_int in rank_white_count:
                rank_white_count[rank_int] = max(0, blank_int) + 1

    # Identify groups of criteria with the same rank
    rank_groups = {}
    for rank in criteria_cards['rank'].unique():
        rank_groups[rank] = criteria_cards[criteria_cards['rank'] == rank].index.tolist()
    sorted_ranks = sorted(rank_groups.keys())

    dynamic_gap_deltas = {}
    dynamic_gap_t_scales = {}
    if use_dynamic_numeric_scale:
        for i in range(1, len(sorted_ranks)):
            prev_rank = int(sorted_ranks[i - 1])
            curr_rank = int(sorted_ranks[i])
            dynamic_gap_deltas[prev_rank] = model.addVar(
                lb=min_delta,
                name=f"delta_{prev_rank}_{curr_rank}",
            )
    if use_dynamic_hfl_scale:
        for i in range(1, len(sorted_ranks)):
            prev_rank = int(sorted_ranks[i - 1])
            curr_rank = int(sorted_ranks[i])
            local_t = model.addVar(lb=1e-6, name=f"hfl_t_{prev_rank}_{curr_rank}")
            dynamic_gap_t_scales[prev_rank] = local_t
            model.addConstr(local_t >= epsilon, f"hfl_t_ge_epsilon_{prev_rank}_{curr_rank}")

    # Precompute probability clouds for belief-degree SRF to avoid repeated parsing.
    e_probability_cloud = {}
    if comp_rule_successive == 'probability-distribution':
        for i in range(1, len(sorted_ranks)):
            prev_rank = sorted_ranks[i - 1]
            cloud = _extract_probability_pairs(
                e_value,
                value_prefix=f"e-value-{prev_rank}-",
                beta_prefix=f"e-beta-{prev_rank}-",
            )
            # Fallback to observed deck spacing if no pair is provided for this rank gap.
            if not cloud and rank_white_count[prev_rank] > 1:
                cloud = {float(rank_white_count[prev_rank] - 1): 1.0}
            if cloud:
                e_probability_cloud[prev_rank] = _normalize_probability_cloud(cloud)

    """
    ADD CONSTRAINTS
    """
    # 1. Within ex aequo constraints (same rank = same weight)
    if comp_rule_within == 'equal':
        for rank, indices in rank_groups.items():
            for i in range(1, len(indices)):
                model.addConstr(weights[indices[0]] == weights[indices[i]], f"equal_within_rank_{rank}_{i}")

    # 2. Between successive ex aequo constraints

    for i in range(1, len(sorted_ranks)):
        prev_rank = sorted_ranks[i-1]
        curr_rank = sorted_ranks[i]
        prev_indices = rank_groups[prev_rank]
        curr_indices = rank_groups[curr_rank]

        prev_index = prev_indices[0]  # representative of previous rank
        curr_index = curr_indices[0]  # representative of current rank
        gap_delta_scale = dynamic_gap_deltas.get(int(prev_rank), delta)
        gap_t_scale = dynamic_gap_t_scales.get(int(prev_rank), t_scale)

        # Add constraint based on comp_rule_successive
        if comp_rule_successive == 'fixed-spacing':
            model.addConstr(
                weights[curr_index] - weights[prev_index] == delta * rank_white_count[prev_rank],
                f"successive_fixed_{prev_rank}_{curr_rank}"
            )
        elif comp_rule_successive == 'fully-flexible':
            model.addConstr(
                weights[curr_index] - weights[prev_index] >= min_delta * rank_white_count[prev_rank],
                f"successive_flexible_{prev_rank}_{curr_rank}"
            )
        elif comp_rule_successive == 'interval-constrained':
            if dynamic_unit_weight:
                model.addConstr(
                    weights[curr_index] - weights[prev_index] >= min_delta * rank_white_count[prev_rank],
                    f"successive_interval_dynamic_lb_{prev_rank}_{curr_rank}"
                )
            else:
                has_interval_bounds = (
                    isinstance(e_value, dict)
                    and f'emin_{prev_rank}' in e_value
                    and f'emax_{prev_rank}' in e_value
                )
                if has_interval_bounds:
                    min_units = float(e_value[f'emin_{prev_rank}']) + 1.0
                    max_units = float(e_value[f'emax_{prev_rank}']) + 1.0
                    model.addConstr(
                        weights[curr_index] - weights[prev_index] >= gap_delta_scale * min_units,
                        f"successive_interval_lb_{prev_rank}_{curr_rank}"
                    )
                    model.addConstr(
                        weights[curr_index] - weights[prev_index] <= gap_delta_scale * max_units,
                        f"successive_interval_ub_{prev_rank}_{curr_rank}"
                    )
                else:
                    model.addConstr(
                        weights[curr_index] - weights[prev_index] == gap_delta_scale * rank_white_count[prev_rank],
                        f"successive_interval_fixed_{prev_rank}_{curr_rank}"
                    )
        elif comp_rule_successive == 'probability-distribution':
            if dynamic_unit_weight:
                model.addConstr(
                    weights[curr_index] - weights[prev_index] >= min_delta * rank_white_count[prev_rank],
                    f"successive_probabilistic_dynamic_lb_{prev_rank}_{curr_rank}"
                )
            else:
                e_values_rank = e_probability_cloud.get(prev_rank, {})
                if e_values_rank:
                    min_units = float(min(e_values_rank.keys())) + 1.0
                    max_units = float(max(e_values_rank.keys())) + 1.0
                    model.addConstr(
                        weights[curr_index] - weights[prev_index] >= gap_delta_scale * min_units,
                        f"successive_probabilistic_lb_{prev_rank}_{curr_rank}"
                    )
                    model.addConstr(
                        weights[curr_index] - weights[prev_index] <= gap_delta_scale * max_units,
                        f"successive_probabilistic_ub_{prev_rank}_{curr_rank}"
                    )
                else:
                    model.addConstr(
                        weights[curr_index] - weights[prev_index] == gap_delta_scale * rank_white_count[prev_rank],
                        f"successive_probabilistic_fixed_{prev_rank}_{curr_rank}"
                    )
        elif comp_rule_successive == 'hfl-linguistic-interval':
            r_min_term = e_value.get(f'rmin_{prev_rank}', 1)
            r_max_term = e_value.get(f'rmax_{prev_rank}', r_min_term)

            if r_max_term < r_min_term:
                raise ValueError(
                    f"Invalid HFL interval for rank pair {prev_rank}-{curr_rank}: r_min > r_max."
                )
            r_min = _map_hfl_card_term(r_min_term)
            r_max = _map_hfl_card_term(r_max_term)

            if dynamic_unit_weight:
                model.addConstr(
                    weights[curr_index] - weights[prev_index] >= min_delta * rank_white_count[prev_rank],
                    f"successive_hfl_dynamic_lb_{prev_rank}_{curr_rank}"
                )
            else:
                model.addConstr(
                    weights[curr_index] - weights[prev_index] >= r_min * gap_t_scale,
                    f"successive_hfl_lb_{prev_rank}_{curr_rank}"
                )
                model.addConstr(
                    weights[curr_index] - weights[prev_index] <= r_max * gap_t_scale,
                    f"successive_hfl_ub_{prev_rank}_{curr_rank}"
                )
        else:
            raise ValueError('Invalid rule for comparison of successive ranks')

        # >>> SMAA <<<
        if launch_smaa:
            if comp_rule_successive == 'fully-flexible':
                if ratio_mode != 'interval-successive':
                    model.addConstr(
                        weights[curr_index] - weights[prev_index] == np.random.uniform(min_delta * rank_white_count[prev_rank], 100),
                        f"successive_smaa_{prev_rank}_{curr_rank}"
                    )
            elif comp_rule_successive == 'interval-constrained':
                has_interval_bounds = (
                    isinstance(e_value, dict)
                    and f'emin_{prev_rank}' in e_value
                    and f'emax_{prev_rank}' in e_value
                )
                if has_interval_bounds:
                    e_value_sample = np.random.uniform(e_value[f'emin_{prev_rank}'], e_value[f'emax_{prev_rank}'])
                    model.addConstr(
                        weights[curr_index] - weights[prev_index] == gap_delta_scale * (e_value_sample + 1),
                        f"successive_smaa_{prev_rank}_{curr_rank}"
                    )
                else:
                    model.addConstr(
                        weights[curr_index] - weights[prev_index] == gap_delta_scale * rank_white_count[prev_rank],
                        f"successive_smaa_{prev_rank}_{curr_rank}"
                    )
            elif comp_rule_successive == 'probability-distribution':
                e_values_rank = e_probability_cloud.get(prev_rank, {})
                if e_values_rank:
                    e_value_sample = np.random.uniform(min(e_values_rank.keys()), max(e_values_rank.keys()))
                else:
                    e_value_sample = float(rank_white_count[prev_rank] - 1)
                model.addConstr(
                    weights[curr_index] - weights[prev_index] == gap_delta_scale * (e_value_sample + 1),
                    f"successive_smaa_{prev_rank}_{curr_rank}"
                )
            elif comp_rule_successive == 'hfl-linguistic-interval':
                r_min_term = int(e_value.get(f'rmin_{prev_rank}', 1))
                r_max_term = int(e_value.get(f'rmax_{prev_rank}', r_min_term))
                if r_max_term < r_min_term:
                    raise ValueError(
                        f"Invalid HFL interval for rank pair {prev_rank}-{curr_rank}: r_min > r_max."
                    )
                r_min = _map_hfl_card_term(r_min_term)
                r_max = _map_hfl_card_term(r_max_term)
                r_value_sample = random.randint(r_min, r_max)
                model.addConstr(
                    weights[curr_index] - weights[prev_index] == r_value_sample * gap_t_scale,
                    f"successive_smaa_{prev_rank}_{curr_rank}"
                )

    # Optional MILP layer for modular robust+imprecise combinations:
    # enforce conditional ordering/equality relations between successive gaps.
    if conditional_gap_milp and comp_rule_successive in {
        'interval-constrained',
        'probability-distribution',
        'hfl-linguistic-interval',
    }:
        gap_unit_bounds = _extract_gap_unit_bounds(
            sorted_ranks=sorted_ranks,
            rank_white_count=rank_white_count,
            e_value=e_value,
            comp_rule_successive=comp_rule_successive,
        )
        _add_conditional_gap_order_constraints(
            model=model,
            weights=weights,
            rank_groups=rank_groups,
            sorted_ranks=sorted_ranks,
            gap_unit_bounds=gap_unit_bounds,
            min_delta=min_delta,
        )

    # 3. Z-ratio constraint
    min_index = rank_groups[min(sorted_ranks)][0]
    max_index = rank_groups[max(sorted_ranks)][0]
    z_values_pb = None
    e0_values_pb = None
    e0_interval_bounds = None
    if ratio_mode == 'exact-ratio':
        model.addConstr(
            weights[max_index] == z_value * weights[min_index],
            "z_ratio_constraint"
        )
    elif ratio_mode == 'linear-spacing':
        default_bar_sum = sum(float(max(1, rank_white_count[rank])) for rank in sorted_ranks[:-1])

        def _bar_e_bounds_for_gap(prev_rank):
            # bar{e}_s = e_s + 1 in interval/probability settings, and mapped HFL term for fuzzy settings.
            default_bar = float(max(1, rank_white_count.get(prev_rank, 1)))
            if not isinstance(e_value, dict):
                return default_bar, default_bar

            if comp_rule_successive == 'interval-constrained':
                e_min_key = f'emin_{prev_rank}'
                e_max_key = f'emax_{prev_rank}'
                if e_min_key in e_value and e_max_key in e_value:
                    low = float(e_value[e_min_key]) + 1.0
                    high = float(e_value[e_max_key]) + 1.0
                    if high < low:
                        raise ValueError(
                            f"Invalid rank-gap interval for rank {prev_rank}: emin > emax."
                        )
                    return max(1.0, low), max(1.0, high)

            if comp_rule_successive == 'probability-distribution':
                cloud = _extract_probability_pairs(
                    e_value,
                    value_prefix=f"e-value-{prev_rank}-",
                    beta_prefix=f"e-beta-{prev_rank}-",
                )
                if cloud:
                    low = float(min(cloud.keys())) + 1.0
                    high = float(max(cloud.keys())) + 1.0
                    return max(1.0, low), max(1.0, high)

            if comp_rule_successive == 'hfl-linguistic-interval':
                if f'rmin_{prev_rank}' in e_value or f'rmax_{prev_rank}' in e_value:
                    r_min_term = int(e_value.get(f'rmin_{prev_rank}', e_value.get(f'rmax_{prev_rank}', 1)))
                    r_max_term = int(e_value.get(f'rmax_{prev_rank}', e_value.get(f'rmin_{prev_rank}', r_min_term)))
                    if r_max_term < r_min_term:
                        raise ValueError(
                            f"Invalid HFL rank-gap interval for rank {prev_rank}: rmin > rmax."
                        )
                    return float(_map_hfl_card_term(r_min_term)), float(_map_hfl_card_term(r_max_term))

            return default_bar, default_bar

        bar_sum_low = 0.0
        bar_sum_high = 0.0
        for prev_rank in sorted_ranks[:-1]:
            gap_low, gap_high = _bar_e_bounds_for_gap(prev_rank)
            bar_sum_low += gap_low
            bar_sum_high += gap_high

        def _sample_bar_sum():
            sampled_sum = 0.0
            for prev_rank in sorted_ranks[:-1]:
                default_bar = float(max(1, rank_white_count.get(prev_rank, 1)))
                if not isinstance(e_value, dict):
                    sampled_sum += default_bar
                    continue

                if comp_rule_successive == 'interval-constrained':
                    e_min_key = f'emin_{prev_rank}'
                    e_max_key = f'emax_{prev_rank}'
                    if e_min_key in e_value and e_max_key in e_value:
                        low = float(e_value[e_min_key]) + 1.0
                        high = float(e_value[e_max_key]) + 1.0
                        sampled_sum += np.random.uniform(min(low, high), max(low, high))
                        continue

                if comp_rule_successive == 'probability-distribution':
                    cloud = e_probability_cloud.get(prev_rank, {})
                    if cloud:
                        support = np.array(list(cloud.keys()), dtype=float)
                        probs = np.array(list(cloud.values()), dtype=float)
                        sampled_e = float(np.random.choice(support, p=probs))
                        sampled_sum += sampled_e + 1.0
                        continue

                if comp_rule_successive == 'hfl-linguistic-interval':
                    if f'rmin_{prev_rank}' in e_value or f'rmax_{prev_rank}' in e_value:
                        r_min_term = int(e_value.get(f'rmin_{prev_rank}', e_value.get(f'rmax_{prev_rank}', 1)))
                        r_max_term = int(e_value.get(f'rmax_{prev_rank}', e_value.get(f'rmin_{prev_rank}', r_min_term)))
                        r_low = _map_hfl_card_term(min(r_min_term, r_max_term))
                        r_high = _map_hfl_card_term(max(r_min_term, r_max_term))
                        sampled_sum += float(random.randint(r_low, r_high))
                        continue

                sampled_sum += default_bar
            return sampled_sum

        def _z_bounds_from_intervals(bar_low, bar_high, e0_low, e0_high):
            e0_low = float(e0_low)
            e0_high = float(e0_high)
            if e0_low < 0 or e0_high < 0:
                raise ValueError("e0 bounds must be non-negative for linear-spacing mode.")
            if e0_high < e0_low:
                raise ValueError("Invalid e0 interval: emin_0 > emax_0.")
            z_low = (float(bar_low) + e0_high + 1.0) / (e0_high + 1.0)
            z_high = (float(bar_high) + e0_low + 1.0) / (e0_low + 1.0)
            if z_high < z_low:
                z_low, z_high = z_high, z_low
            return z_low, z_high

        e0_exact = None
        if isinstance(e_value, dict):
            if comp_rule_successive == 'probability-distribution':
                e0_cloud = _extract_probability_pairs(
                    e_value,
                    value_prefix='e-value-0-',
                    beta_prefix='e-beta-0-',
                )
                if e0_cloud:
                    e0_values_pb = _normalize_probability_cloud(e0_cloud)
                    e0_min = min(e0_values_pb.keys())
                    e0_max = max(e0_values_pb.keys())
                    e0_interval_bounds = (float(e0_min), float(e0_max))
                elif 'e0' in e_value:
                    e0_exact = float(e_value.get('e0', 0))
            if e0_exact is None and comp_rule_successive == 'hfl-linguistic-interval' and ('rmin_0' in e_value or 'rmax_0' in e_value):
                r_min_term = int(e_value.get('rmin_0', 1))
                r_max_term = int(e_value.get('rmax_0', r_min_term))
                if r_max_term < r_min_term:
                    raise ValueError('Invalid HFL e0 interval: rmin_0 > rmax_0.')
                e0_min = float(_map_hfl_card_term(r_min_term))
                e0_max = float(_map_hfl_card_term(r_max_term))
                e0_interval_bounds = (e0_min, e0_max)
            if e0_exact is None and ('emin_0' in e_value or 'emax_0' in e_value):
                e0_min = float(e_value.get('emin_0', e_value.get('emax_0', 0)))
                e0_max = float(e_value.get('emax_0', e_value.get('emin_0', e0_min)))
                if e0_max < e0_min:
                    raise ValueError('Invalid e0 interval: emin_0 > emax_0.')
                e0_interval_bounds = (e0_min, e0_max)
            if e0_exact is None and e0_interval_bounds is None:
                e0_exact = float(e_value.get('e0', 0))
        else:
            e0_exact = float(e_value)

        if e0_interval_bounds is not None:
            e0_low, e0_high = e0_interval_bounds
            z_low, z_high = _z_bounds_from_intervals(
                bar_sum_low,
                bar_sum_high,
                e0_low,
                e0_high,
            )
            model.addConstr(
                weights[max_index] >= z_low * weights[min_index],
                "z_ratio_constraint_min"
            )
            model.addConstr(
                weights[max_index] <= z_high * weights[min_index],
                "z_ratio_constraint_max"
            )
        else:
            e0_anchor = float(e0_exact)
            if e0_anchor < 0:
                raise ValueError("e0 must be non-negative for linear-spacing mode.")
            if abs(bar_sum_high - bar_sum_low) > 1e-9:
                z_low, z_high = _z_bounds_from_intervals(
                    bar_sum_low,
                    bar_sum_high,
                    e0_anchor,
                    e0_anchor,
                )
                model.addConstr(
                    weights[max_index] >= z_low * weights[min_index],
                    "z_ratio_constraint_min"
                )
                model.addConstr(
                    weights[max_index] <= z_high * weights[min_index],
                    "z_ratio_constraint_max"
                )
            else:
                z_value = (bar_sum_low + e0_anchor + 1.0) / (e0_anchor + 1.0)
                model.addConstr(
                    weights[max_index] == z_value * weights[min_index],
                    "z_ratio_constraint"
                )
    elif ratio_mode == 'interval-successive':
        for rank in range(1, cards_arrangement['rank'].max()):
            curr_rank = rank_groups[rank][0]
            next_rank = rank_groups[rank + 1][0]

            model.addConstr(
                weights[next_rank] >= z_value[f'zmin_{rank}'] * weights[curr_rank],
                f"z_ratio_constraint_min_{rank}"
            )

            model.addConstr(
                weights[next_rank] <= z_value[f'zmax_{rank}'] * weights[curr_rank],
                f"z_ratio_constraint_max_{rank}"
            )
    elif ratio_mode == 'interval-total':
        model.addConstr(
            weights[max_index] >= z_value['zmin'] * weights[min_index],
            "z_ratio_constraint_min"
        )

        model.addConstr(
            weights[max_index] <= z_value['zmax'] * weights[min_index],
            "z_ratio_constraint_max"
        )
    elif ratio_mode == 'probability-cloud':
        z_values_pb = _extract_probability_pairs(
            z_value,
            value_prefix='z-value-',
            beta_prefix='z-beta-',
        )
        if not z_values_pb:
            raise ValueError("No valid (z, beta) pairs were provided for belief-degree SRF.")
        z_values_pb = _normalize_probability_cloud(z_values_pb)
        model.addConstr(
            weights[max_index] >= min(z_values_pb.keys()) * weights[min_index],
            "z_ratio_constraint_min"
        )

        model.addConstr(
            weights[max_index] <= max(z_values_pb.keys()) * weights[min_index],
            "z_ratio_constraint_max"
        )
    elif ratio_mode == 'hfl-ratio-interval':
        if isinstance(z_value, dict):
            e_min_term = float(z_value.get('emin', z_value.get('zmin', 1.0)))
            e_max_term = float(z_value.get('emax', z_value.get('zmax', e_min_term)))
        else:
            e_min_term = float(z_value)
            e_max_term = float(z_value)

        if e_max_term < e_min_term:
            raise ValueError('Invalid HFL global ratio interval: e_min > e_max.')
        e_min = _map_hfl_z_term(e_min_term)
        e_max = _map_hfl_z_term(e_max_term)

        model.addConstr(
            weights[max_index] >= e_min * weights[min_index],
            "z_ratio_hfl_min"
        )
        model.addConstr(
            weights[max_index] <= e_max * weights[min_index],
            "z_ratio_hfl_max"
        )
    else:
        raise ValueError('Invalid z ratio mode')

    # >>> SMAA <<<
    if launch_smaa:
        if ratio_mode == 'interval-successive':
            for rank in range(1, cards_arrangement['rank'].max()):
                curr_rank = rank_groups[rank][0]
                next_rank = rank_groups[rank + 1][0]

                z_value_sample = np.random.uniform(z_value[f'zmin_{rank}'], z_value[f'zmax_{rank}'])
                model.addConstr(
                    weights[next_rank] == z_value_sample * weights[curr_rank],
                    f"z_ratio_constraint_smaa_{rank}"
                )
        elif ratio_mode == 'interval-total':
            z_value_sample = np.random.uniform(z_value['zmin'], z_value['zmax'])
            model.addConstr(
                weights[max_index] == z_value_sample * weights[min_index],
                "z_ratio_constraint_smaa"
            )
        elif ratio_mode == 'probability-cloud':
            z_value_sample = np.random.uniform(min(z_values_pb.keys()), max(z_values_pb.keys()))
            model.addConstr(
                weights[max_index] == z_value_sample * weights[min_index],
                "z_ratio_constraint_smaa"
            )
        elif ratio_mode == 'hfl-ratio-interval':
            if isinstance(z_value, dict):
                e_min_term = int(z_value.get('emin', z_value.get('zmin', 1)))
                e_max_term = int(z_value.get('emax', z_value.get('zmax', e_min_term)))
            else:
                e_min_term = int(float(z_value))
                e_max_term = int(float(z_value))

            if e_max_term < e_min_term:
                raise ValueError('Invalid HFL global ratio interval: e_min > e_max.')
            e_min = _map_hfl_z_term(e_min_term)
            e_max = _map_hfl_z_term(e_max_term)
            e_value_sample = random.randint(e_min, e_max)
            model.addConstr(
                weights[max_index] == e_value_sample * weights[min_index],
                "z_ratio_constraint_smaa"
            )
        elif ratio_mode == 'linear-spacing':
            bar_sum_sample = _sample_bar_sum() if '_sample_bar_sum' in locals() else default_bar_sum
            if e0_values_pb:
                e0_support = np.array(list(e0_values_pb.keys()), dtype=float)
                e0_probs = np.array(list(e0_values_pb.values()), dtype=float)
                e0_value_sample = np.random.choice(e0_support, p=e0_probs)
                z_value_sample = (bar_sum_sample + (e0_value_sample + 1.0)) / (e0_value_sample + 1.0)
                model.addConstr(
                    weights[max_index] == z_value_sample * weights[min_index],
                    "z_ratio_constraint_smaa"
                )
            elif e0_interval_bounds is not None:
                e0_low, e0_high = e0_interval_bounds
                e0_value_sample = np.random.uniform(e0_low, e0_high)
                z_value_sample = (bar_sum_sample + (e0_value_sample + 1.0)) / (e0_value_sample + 1.0)
                model.addConstr(
                    weights[max_index] == z_value_sample * weights[min_index],
                    "z_ratio_constraint_smaa"
                )
            elif e0_exact is not None:
                e0_value_sample = float(e0_exact)
                z_value_sample = (bar_sum_sample + (e0_value_sample + 1.0)) / (e0_value_sample + 1.0)
                model.addConstr(
                    weights[max_index] == z_value_sample * weights[min_index],
                    "z_ratio_constraint_smaa"
                )

    # 4. Add normalization constraint if required.
    # For SMAA-style fully-flexible sampling we usually skip normalization to keep model generation broad.
    # But minimum-weight requirements are absolute on the normalized 0-100 scale, so normalization must stay active.
    min_weight_enabled = (
        isinstance(extra_cond, dict)
        and isinstance(extra_cond.get('minimum_weight', {}), dict)
        and bool(extra_cond.get('minimum_weight', {}).get('enabled'))
    )
    skip_normalization = (
        launch_smaa
        and comp_rule_successive == 'fully-flexible'
        and not min_weight_enabled
    )
    if normalized and not skip_normalization:
        target_sum = 100
        model.addConstr(
            gp.quicksum(weights.values()) == target_sum,
            "normalization_constraint"
        )

    # 5. Apply any extra conditions
    if extra_cond is not None:
        _add_optional_extra_constraints(model, weights, criteria_cards, extra_cond)

    """
    MODEL CONFIGURATION
    """
    if comp_rule_successive == 'hfl-linguistic-interval' and epsilon is not None and not launch_smaa:
        # HFL-SRF Model I objective: maximize epsilon while preserving feasibility.
        model.setObjective(epsilon, GRB.MAXIMIZE)
    else:
        # Set the objective to zero to find any feasible solution.
        model.setObjective(0, GRB.MINIMIZE)

    # Turn off pool search for a single solution
    model.setParam("PoolSearchMode", 0)

    return model, weights, rank_groups, criteria_cards, delta


def calc_srf_modular(cards_arrangement,
                     z_value,
                     e_value,
                     comp_rule_within='equal',
                     comp_rule_successive='fixed-spacing',
                     ratio_mode=None,
                     normalized=True,
                     extra_cond=None,
                     w_value=1,
                     min_delta=1.0,
                     conditional_gap_milp=False,
                     dynamic_unit_weight=False):
    """
    Calculates criteria weights using Linear Programming with a free MILP solver
    
    Args:
        cards_arrangement (pd.DataFrame): Preprocessed card arrangement data
        z_value: Ratio between first and last rank or successive ranks
        e_value: Spacing between cards
        comp_rule_within (str): Rule for comparing weights within an ex aequo set
        comp_rule_successive (str): Rule for comparing weights between successive sets
        ratio_mode (float): Target ratio between most and least important criteria
        normalized (bool): Whether to normalize weights to sum to 100
        extra_cond (callable, optional): Additional constraints function
        w_value (int): Decimal precision for weight normalization
        min_delta (float): Minimum difference between successive rank weights
        conditional_gap_milp (bool): Whether to add conditional robust-imprecise gap MILP constraints.
        dynamic_unit_weight (bool): Whether to allow rank-pair-specific gap-scale variables.
        
    Returns:
        pd.DataFrame: Calculated criteria weights
    """
    # Build the model using the shared helper function
    model, weights, rank_groups, criteria_cards, delta = _build_srf_model(
        cards_arrangement,
        z_value,
        e_value,
        comp_rule_within, 
        comp_rule_successive, 
        ratio_mode,
        normalized, 
        extra_cond, 
        min_delta,
        conditional_gap_milp=conditional_gap_milp,
        dynamic_unit_weight=dynamic_unit_weight
    )
    
    # Solve the model
    _optimize_model(model)
    
    # Check if a solution was found
    if model.status != GRB.OPTIMAL:
        raise ValueError(f"No optimal solution found. Status: {model.status}")
    
    # Create results DataFrame
    simos_calc_results = pd.DataFrame(columns=['r', 'name', 'k_i'],
                                      index=criteria_cards.index[::-1])
    
    simos_calc_results['r'] = criteria_cards['rank']
    simos_calc_results['name'] = criteria_cards['name']
    
    # Extract and process weights
    for idx in simos_calc_results.index:
        simos_calc_results.loc[idx, 'k_i'] = weights[idx].X
    
    # If normalization with rounding is required, use round_up_selected
    if normalized and w_value > 0:
        target_sum = 100
        # We still use round_up_selected to handle potential rounding issues that might affect the sum
        simos_calc_results['k_i'] = round_up_selected(simos_calc_results['k_i'], w_value, target_sum=target_sum)
    
    return simos_calc_results


"""
FUNCTIONS FOR RANDOM SAMPLING OF CRITERIA WEIGHTS
"""


def calc_srf_rand_samples(cards_arrangement,
                          z_value,
                          e_value,
                          comp_rule_within=None,
                          comp_rule_successive=None,
                          ratio_mode=None,
                          normalized=False,
                          extra_cond=None,
                          min_delta=1.0,
                          n_samples=100,
                          conditional_gap_milp=False,
                          dynamic_unit_weight=False):
    """
    Generates feasible samples of criteria weights for variability analysis.

    For continuous SRF models, the sampler uses hit-and-run to target the
    uniform distribution over the feasible region. Mixed-integer cases fall back
    to feasible-solution exploration when uniform polytope sampling is not
    directly available.

    Args:
        cards_arrangement (pd.DataFrame): Card arrangement data
        z_value: Ratio between first and last rank or successive ranks
        e_value: Spacing between cards
        comp_rule_within (str): Rule for comparing weights within ex aequo sets
        comp_rule_successive (str): Rule for comparing weights between successive sets
        ratio_mode (float): Target ratio between max and min weights
        normalized (bool): Whether to normalize weights to sum to 100
        extra_cond (callable, optional): Additional constraints function
        min_delta (float): Minimum delta for random sampling
        n_samples (int): Number of random samples to generate
        conditional_gap_milp (bool): Whether to add conditional robust-imprecise gap MILP constraints.
        dynamic_unit_weight (bool): Whether to allow rank-pair-specific gap-scale variables.

    Returns:
        pd.DataFrame: Matrix of random weight samples
    """

    results = []
    allow_convex_mixing = True
    update_calculation_progress(
        stage='sampling',
        message='Uniformly sampling feasible solutions...',
        current=0,
        total=n_samples,
        active=True,
        done=False
    )

    use_zero_dynamic_hit_and_run = bool(
        dynamic_unit_weight
        and comp_rule_successive == 'interval-constrained'
        and ratio_mode == 'linear-spacing'
        and isinstance(e_value, dict)
        and normalized
        and ({'emin_0', 'emax_0'} & set(e_value.keys()) or 'e0' in e_value)
    )
    if use_zero_dynamic_hit_and_run:
        hitrun_samples = _try_hit_and_run_zero_dynamic_samples(
            cards_arrangement=cards_arrangement,
            e_value=e_value,
            extra_cond=extra_cond,
            min_delta=min_delta,
            n_samples=n_samples,
            normalized=normalized,
            conditional_gap_milp=conditional_gap_milp
        )
        if isinstance(hitrun_samples, pd.DataFrame) and not hitrun_samples.empty:
            hitrun_samples.rename(columns=cards_arrangement['name']).to_json(
                str(SRF_SAMPLES_PATH), orient='records'
            )
            update_calculation_progress(
                stage='sampling',
                message='Uniformly sampling feasible solutions...',
                current=n_samples,
                total=n_samples,
                active=True,
                done=False
            )
            return hitrun_samples, calc_asi(hitrun_samples)

    base_model = None
    base_weights = None
    base_criteria_cards = None
    base_delta = None
    if normalized:
        base_model, base_weights, _, base_criteria_cards, base_delta = _build_srf_model(
            cards_arrangement,
            z_value,
            e_value,
            comp_rule_within,
            comp_rule_successive,
            ratio_mode,
            normalized,
            extra_cond,
            min_delta,
            launch_smaa=False,
            conditional_gap_milp=conditional_gap_milp,
            dynamic_unit_weight=dynamic_unit_weight
        )
        allow_convex_mixing = all(
            _lp_var_is_continuous(var)
            for var in getattr(base_model, '_vars', [])
        )
        hitrun_samples = _try_hit_and_run_model_samples(
            model=base_model,
            weights=base_weights,
            criteria_cards=base_criteria_cards,
            n_samples=n_samples,
            normalized=normalized,
            progress_message='Uniformly sampling feasible solutions'
        )
        if isinstance(hitrun_samples, pd.DataFrame) and not hitrun_samples.empty:
            hitrun_samples.rename(columns=cards_arrangement['name']).to_json(
                str(SRF_SAMPLES_PATH), orient='records'
            )
            update_calculation_progress(
                stage='sampling',
                message='Uniformly sampling feasible solutions...',
                current=n_samples,
                total=n_samples,
                active=True,
                done=False
            )
            return hitrun_samples, calc_asi(hitrun_samples)

    def _solution_signature(solution):
        return tuple(
            round(v, 8)
            for _, v in sorted(solution.items(), key=lambda kv: str(kv[0]))
        )

    seen_signatures = set()
    # HFL sampling based on rebuilding randomized SMAA models can be very slow.
    # For HFL, sample the feasible region by random objectives on one fixed model
    # (plus interior convex mixing below), which still yields a rich PCA cloud.
    use_smaa_phase = (
        comp_rule_successive != 'hfl-linguistic-interval'
        and not (
            dynamic_unit_weight
            and comp_rule_successive in {'interval-constrained', 'probability-distribution', 'hfl-linguistic-interval'}
        )
    )
    if use_smaa_phase:
        attempts = 0
        max_attempts = max(6 * n_samples, n_samples + 25)
        while len(results) < n_samples and attempts < max_attempts:
            attempts += 1
            # Build the model using the shared helper function
            model, weights, rank_groups, criteria_cards, delta = _build_srf_model(
                cards_arrangement,
                z_value,
                e_value,
                comp_rule_within,
                comp_rule_successive,
                ratio_mode,
                normalized,
                extra_cond,
                min_delta,
                launch_smaa=True,
                conditional_gap_milp=conditional_gap_milp,
                dynamic_unit_weight=dynamic_unit_weight
            )

            # Solve to get a solution
            model.optimize()

            if model.status == GRB.OPTIMAL:
                # Extract and record weights for this solution
                solution = {idx: weights[idx].X for idx in criteria_cards.index}
                if normalized:
                    solution = {k: (v / sum(solution.values())) * 100 for k, v in solution.items()}
                signature = _solution_signature(solution)
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    results.append(solution)
                    if _should_emit_progress(len(results), n_samples):
                        update_calculation_progress(
                            stage='sampling',
                            message='Sampling feasible solutions...',
                            current=len(results),
                            total=n_samples,
                            active=True,
                            done=False
                        )

    # Fallback/top-up: if probabilistic draws are sparse, sample feasible points directly.
    if len(results) < n_samples:
        if base_model is None or base_weights is None or base_criteria_cards is None:
            model, weights, rank_groups, criteria_cards, delta = _build_srf_model(
                cards_arrangement,
                z_value,
                e_value,
                comp_rule_within,
                comp_rule_successive,
                ratio_mode,
                normalized,
                extra_cond,
                min_delta,
                launch_smaa=False,
                conditional_gap_milp=conditional_gap_milp,
                dynamic_unit_weight=dynamic_unit_weight
            )
        else:
            model = base_model
            weights = base_weights
            criteria_cards = base_criteria_cards
            delta = base_delta
        if model is not None:
            allow_convex_mixing = all(
                _lp_var_is_continuous(var)
                for var in getattr(model, '_vars', [])
            )
        aux_scale_vars = [
            var for var in getattr(model, '_vars', [])
            if isinstance(getattr(var, 'name', None), str)
            and (var.name.startswith('delta_') or var.name.startswith('hfl_t_'))
        ]

        top_up_attempts = 0
        max_top_up_attempts = max(20 * (n_samples - len(results)), (n_samples - len(results)) + 50)
        while len(results) < n_samples and top_up_attempts < max_top_up_attempts:
            top_up_attempts += 1
            obj_expr = gp.quicksum(
                np.random.uniform(-1.0, 1.0) * weights[idx]
                for idx in weights
            )
            if delta is not None:
                obj_expr += np.random.uniform(-1.0, 1.0) * delta
            for scale_var in aux_scale_vars:
                obj_expr += np.random.uniform(-1.0, 1.0) * scale_var
            objective_sense = GRB.MAXIMIZE if np.random.rand() > 0.5 else GRB.MINIMIZE
            model.setObjective(obj_expr, objective_sense)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                solution = {idx: weights[idx].X for idx in criteria_cards.index}
                if normalized:
                    solution = {k: (v / sum(solution.values())) * 100 for k, v in solution.items()}
                signature = _solution_signature(solution)
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    results.append(solution)
                    if _should_emit_progress(len(results), n_samples):
                        update_calculation_progress(
                            stage='sampling',
                            message='Sampling feasible solutions...',
                            current=len(results),
                            total=n_samples,
                            active=True,
                            done=False
                        )

    # Interior-point enrichment for belief-degree and HFL:
    # LP-based sampling tends to return many vertices; mix feasible solutions to
    # guarantee points inside the polyhedron for PCA clouds.
    if (allow_convex_mixing
            and comp_rule_successive in ['probability-distribution', 'hfl-linguistic-interval']
            and len(results) >= 2):
        base_results = results.copy()
        mixed_results = []
        mixed_signatures = set()
        target_interior = min(max(n_samples // 2, 10), n_samples)
        interior_attempts = 0
        max_interior_attempts = max(40 * target_interior, target_interior + 200)

        while len(mixed_results) < target_interior and interior_attempts < max_interior_attempts:
            interior_attempts += 1
            s1, s2 = random.sample(base_results, 2)
            alpha = float(np.random.uniform(0.05, 0.95))
            mixed = {key: alpha * s1[key] + (1 - alpha) * s2[key] for key in s1}
            signature = _solution_signature(mixed)

            if signature not in mixed_signatures:
                mixed_signatures.add(signature)
                mixed_results.append(mixed)

        if mixed_results:
            keep_base = max(0, n_samples - len(mixed_results))
            results = mixed_results + base_results[:keep_base]
            results = results[:n_samples]
            if _should_emit_progress(len(results), n_samples):
                update_calculation_progress(
                    stage='sampling',
                    message='Refining interior samples...',
                    current=len(results),
                    total=n_samples,
                    active=True,
                    done=False
                )

    # Final densification: if still sparse, keep adding interior convex combinations.
    seen_signatures = {_solution_signature(solution) for solution in results}
    if allow_convex_mixing and len(results) < n_samples and len(results) >= 2:
        densify_attempts = 0
        max_densify_attempts = max(30 * (n_samples - len(results)), (n_samples - len(results)) + 100)
        while len(results) < n_samples and densify_attempts < max_densify_attempts:
            densify_attempts += 1
            s1, s2 = random.sample(results, 2)
            alpha = random.random()
            solution = {
                key: alpha * s1[key] + (1 - alpha) * s2[key]
                for key in s1
            }
            signature = _solution_signature(solution)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                results.append(solution)
                if _should_emit_progress(len(results), n_samples):
                    update_calculation_progress(
                        stage='sampling',
                        message='Refining interior samples...',
                        current=len(results),
                        total=n_samples,
                        active=True,
                        done=False
                    )

    # Convert to DataFrame and export into a JSON file
    srf_samples = pd.DataFrame(results)
    srf_samples.index = [f'sample_{idx + 1}' for idx in range(len(srf_samples))]
    srf_samples.rename(columns=cards_arrangement['name']).to_json(
        str(SRF_SAMPLES_PATH), orient='records'
    )

    update_calculation_progress(
        stage='sampling',
        message='Sampling feasible solutions...',
        current=n_samples if n_samples else len(srf_samples),
        total=n_samples if n_samples else max(1, len(srf_samples)),
        active=True,
        done=False
    )

    # Calculate the ASI value
    asi_srf_samples = calc_asi(srf_samples) if len(srf_samples) else 0

    return srf_samples, asi_srf_samples


def calc_srf_vertices(cards_arrangement,
                      z_value,
                      e_value,
                      comp_rule_within=None,
                      comp_rule_successive=None,
                      ratio_mode=None,
                      normalized=True,
                      extra_cond=None,
                      min_delta=1.0,
                      n_samples=100,
                      conditional_gap_milp=False,
                      dynamic_unit_weight=False):
    """
    Identifies vertices of the solution space polyhedron (feasible region P).
    
    This function explores the vertices of the polyhedron by solving the LP problem with 
    randomly perturbed objective functions. For LP problems, the optimal solutions lie at 
    the vertices of the feasible region. By using different objective functions, we can find 
    different vertices of the polyhedron.
    
    Note: Due to the randomized approach, this may not find all vertices, but provides
    a representative sample of the solution space boundaries.

    Args:
        cards_arrangement (pd.DataFrame): Card arrangement data
        z_value: Ratio between first and last rank or successive ranks
        e_value: Spacing between cards
        comp_rule_within (str): Rule for comparing weights within ex aequo sets
        comp_rule_successive (str): Rule for comparing weights between successive sets
        ratio_mode (float): Target ratio between max and min weights
        normalized (bool): Whether to normalize weights to sum to 100
        extra_cond (callable, optional): Additional constraints function
        min_delta (float): Minimum delta for random sampling
        n_samples (int): Number of random objective functions to try
        conditional_gap_milp (bool): Whether to add conditional robust-imprecise gap MILP constraints.
        dynamic_unit_weight (bool): Whether to allow rank-pair-specific gap-scale variables.

    Returns:
        pd.DataFrame: Matrix of unique vertices (distinct solutions) found
    """
    # Build the model using the shared helper function
    model, weights, rank_groups, criteria_cards, delta = _build_srf_model(
        cards_arrangement,
        z_value,
        e_value,
        comp_rule_within,
        comp_rule_successive,
        ratio_mode,
        normalized,
        extra_cond,
        min_delta,
        conditional_gap_milp=conditional_gap_milp,
        dynamic_unit_weight=dynamic_unit_weight
    )
    aux_scale_vars = [
        var for var in getattr(model, '_vars', [])
        if isinstance(getattr(var, 'name', None), str)
        and (var.name.startswith('delta_') or var.name.startswith('hfl_t_'))
    ]

    update_calculation_progress(
        stage='extreme',
        message='Exploring extreme scenarios...',
        current=0,
        total=n_samples,
        active=True,
        done=False
    )

    results = []
    for iteration in range(1, n_samples + 1):
        # For finding diverse solutions, add a small random perturbation to the objective function
        # This helps the solver explore different parts of the feasible region
        for idx in weights:
            weights[idx].Obj = np.random.uniform(0, 100)

        if delta is not None:
            delta.Obj = np.random.uniform(min_delta, 100)
        for scale_var in aux_scale_vars:
            scale_var.Obj = np.random.uniform(min_delta, 100)

        # Solve to get a solution
        model.optimize()

        if model.status == GRB.OPTIMAL:
            # Extract and record weights for this solution
            solution = {idx: weights[idx].X for idx in criteria_cards.index}
            results.append(solution)

        if _should_emit_progress(iteration, n_samples):
            update_calculation_progress(
                stage='extreme',
                message='Exploring extreme scenarios...',
                current=iteration,
                total=n_samples,
                active=True,
                done=False
            )

    # Convert to DataFrame and remove the duplicates
    srf_vertices = pd.DataFrame(results)
    srf_vertices = srf_vertices.round(decimals=2).drop_duplicates().reset_index(drop=True)
    srf_vertices = srf_vertices.rename(index={idx: f'vertex_{idx}' for idx in srf_vertices.index})

    # Calculate the ASI value
    asi_srf_vertices = calc_asi(srf_vertices)

    return srf_vertices, asi_srf_vertices


def calc_srf_min_max(cards_arrangement,
                     z_value,
                     e_value,
                     comp_rule_within=None,
                     comp_rule_successive=None,
                     ratio_mode=None,
                     normalized=True,
                     extra_cond=None,
                     min_delta=1.0,
                     conditional_gap_milp=False,
                     dynamic_unit_weight=False):
    """
    Computes the variation range (min/max) of weights for each criterion.
    
    This function solves 2n linear programs (where n is the number of criteria):
        Min pj and Max pj, for j = 1, 2, ..., n
        subject to p in P (the feasible region)
    
    By finding the minimum and maximum possible weight for each criterion within
    the feasible region, we can understand the full range of possible values and
    the flexibility allowed by the constraints.

    Args:
        cards_arrangement (pd.DataFrame): Card arrangement data
        z_value: Ratio between first and last rank or successive ranks
        e_value: Spacing between cards
        comp_rule_within (str): Rule for comparing weights within ex aequo sets
        comp_rule_successive (str): Rule for comparing weights between successive sets
        ratio_mode (float): Target ratio between max and min weights
        normalized (bool): Whether to normalize weights to sum to 100
        extra_cond (callable, optional): Additional constraints function
        min_delta (float): Minimum delta between successive ranks
        conditional_gap_milp (bool): Whether to add conditional robust-imprecise gap MILP constraints.
        dynamic_unit_weight (bool): Whether to allow rank-pair-specific gap-scale variables.

    Returns:
        pd.DataFrame: Matrix containing solutions that define the min/max bounds for each criterion
    """
    # Build the model using the shared helper function
    model, weights, rank_groups, criteria_cards, delta = _build_srf_model(
        cards_arrangement,
        z_value,
        e_value,
        comp_rule_within,
        comp_rule_successive,
        ratio_mode,
        normalized,
        extra_cond,
        min_delta,
        conditional_gap_milp=conditional_gap_milp,
        dynamic_unit_weight=dynamic_unit_weight
    )

    results = []
    scenario_labels = []
    total_runs = 2 * len(weights)
    progress_step = 0
    update_calculation_progress(
        stage='extreme',
        message='Computing extreme scenario bounds...',
        current=0,
        total=total_runs,
        active=True,
        done=False
    )
    for idx in weights:
        # Solve for a minimum criterion weight
        model.setObjective(weights[idx], GRB.MINIMIZE)
        model.optimize()
        progress_step += 1

        if model.status == GRB.OPTIMAL:
            # Extract and record weights for this solution
            solution = {idx: weights[idx].X for idx in criteria_cards.index}
            results.append(solution)
            scenario_labels.append(f"{criteria_cards.loc[idx, 'name']} minimized")
        if _should_emit_progress(progress_step, total_runs):
            update_calculation_progress(
                stage='extreme',
                message='Computing extreme scenario bounds...',
                current=progress_step,
                total=total_runs,
                active=True,
                done=False
            )

        # Solve for a maximum criterion weight
        model.setObjective(weights[idx], GRB.MAXIMIZE)
        model.optimize()
        progress_step += 1

        if model.status == GRB.OPTIMAL:
            # Extract and record weights for this solution
            solution = {idx: weights[idx].X for idx in criteria_cards.index}
            results.append(solution)
            scenario_labels.append(f"{criteria_cards.loc[idx, 'name']} maximized")
        if _should_emit_progress(progress_step, total_runs):
            update_calculation_progress(
                stage='extreme',
                message='Computing extreme scenario bounds...',
                current=progress_step,
                total=total_runs,
                active=True,
                done=False
            )

    # Convert to DataFrame
    srf_min_max = pd.DataFrame(results, index=scenario_labels[:len(results)])

    # Calculate the ASI value
    asi_srf_min_max = calc_asi(srf_min_max)

    return srf_min_max, asi_srf_min_max

