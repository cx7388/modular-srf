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
DATA_DIR.mkdir(parents=True, exist_ok=True)
INCONSISTENCY_BIG_M = 10_000.0
INCONSISTENCY_CARDINALITY_WEIGHT = 1_000_000_000.0
INCONSISTENCY_RESTORATION_BETA = 1.0
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
    if normalized['output_type'] != 'variability':
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


def _observed_gap_counts_by_prev_rank(cards_arrangement):
    """
    Returns observed blank-card counts for each successive rank gap keyed by previous rank.
    """
    criteria_cards = cards_arrangement[cards_arrangement['class'] == 'criterion'].sort_values('rank')

    rank_white_count = {}
    for rank in cards_arrangement['rank'].unique():
        rank_white_count[rank] = cards_arrangement[cards_arrangement['rank'] == rank]['class'].to_list().count('white') + 1

    rank_groups = {}
    for rank in criteria_cards['rank'].unique():
        rank_groups[rank] = criteria_cards[criteria_cards['rank'] == rank].index.tolist()
    sorted_ranks = sorted(rank_groups.keys())

    return {
        int(prev_rank): int(max(0, rank_white_count[prev_rank] - 1))
        for prev_rank in sorted_ranks[:-1]
    }


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


def _add_optional_extra_constraints(model, weights, criteria_cards, extra_cond):
    """
    Adds optional extra constraints:
      - minimum-weight requirement for all criteria
      - anti-dictatorship requirement (automatic for all criteria)
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
                f"extra_min_weight_{idx}"
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
                f"extra_anti_dictatorship_{idx}"
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
    match srf_method:
        case 'srf' | 'srf_ii' | 'belief_degree_imprecise_srf' | 'hfl_srf':
            srf_objective = None
        case 'robust_srf' | 'wap' | 'imprecise_srf':
            srf_objective = 'Maximize ASI'
        case _:
            raise ValueError('Invalid SRF method')

    match srf_method:
        case 'srf' | 'srf_ii' | 'robust_srf' | 'wap' | 'imprecise_srf' | 'belief_degree_imprecise_srf' | 'hfl_srf':
            comp_rule_within = 'equal'
        case _:
            raise ValueError('Invalid SRF method')

    match srf_method:
        case 'srf' | 'srf_ii':
            comp_rule_successive = 'fixed-spacing'
        case 'hfl_srf':
            comp_rule_successive = 'hfl-linguistic-interval'
        case 'robust_srf' | 'wap':
            comp_rule_successive = 'fully-flexible'
        case 'imprecise_srf':
            comp_rule_successive = 'interval-constrained'
        case 'belief_degree_imprecise_srf':
            comp_rule_successive = 'probability-distribution'
        case _:
            raise ValueError('Invalid SRF method')

    match srf_method:
        case 'srf' | 'robust_srf':
            ratio_mode = 'exact-ratio'
        case 'hfl_srf':
            ratio_mode = 'hfl-ratio-interval'
        case 'srf_ii':
            ratio_mode = 'linear-spacing'
        case 'wap':
            ratio_mode = 'interval-successive'
        case 'imprecise_srf':
            ratio_mode = 'interval-total'
        case 'belief_degree_imprecise_srf':
            ratio_mode = 'probability-cloud'
        case _:
            raise ValueError('Invalid SRF method')

    match srf_method:
        case 'srf' | 'srf_ii' | 'robust_srf' | 'wap' | 'imprecise_srf' | 'belief_degree_imprecise_srf' | 'hfl_srf':
            normalized = True
        case _:
            raise ValueError('Invalid SRF method')

    return srf_objective, comp_rule_within, comp_rule_successive, ratio_mode, normalized


def _resolve_inconsistency_structure(srf_method, modular_options=None, modular_profile=None):
    """
    Resolves EI model structure for both classical SRF methods and modular SRF.
    Returns components plus the effective restoration method used by exact fix helpers.
    """
    if srf_method == 'modular_srf':
        options, effective_method = resolve_modular_configuration(
            modular_options=modular_options,
            modular_profile=modular_profile
        )
        (_srf_objective,
         comp_rule_within,
         comp_rule_successive,
         ratio_mode,
         normalized) = _resolve_modular_structure(options)

        # Align EI with the modular dynamic unit-weight setting used in weight computation.
        if options.get('unit_weight') == 'dynamic' and comp_rule_successive == 'fixed-spacing':
            comp_rule_successive = 'fully-flexible'

        return comp_rule_within, comp_rule_successive, ratio_mode, normalized, effective_method

    (_srf_objective,
     comp_rule_within,
     comp_rule_successive,
     ratio_mode,
     normalized) = _resolve_method_structure(srf_method)
    return comp_rule_within, comp_rule_successive, ratio_mode, normalized, srf_method


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

    export_df = samples_df.copy()
    try:
        export_df = export_df.rename(columns=cards_arrangement['name'])
    except Exception:
        pass

    export_df.to_json(str(SRF_SAMPLES_PATH), orient='records')


def calc_srf_flat(cards_arrangement, z_value, e_value, w_value, srf_method,
                  modular_options=None,
                  modular_profile=None,
                  min_delta=1.0,
                  extra_constraints=None):
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

    Returns:
        tuple:
            - pd.DataFrame: Calculated criteria weights
            - float: ASI (Average Stability Index) value
    """
    n_crit_cards = cards_arrangement['class'].to_list().count('criterion')
    is_modular = srf_method == 'modular_srf'

    if is_modular:
        # Modular SRF first resolves questionnaire answers into structural choices,
        # while classical methods already encode those choices in the method name.
        resolved_modular_options, effective_method = resolve_modular_configuration(
            modular_options=modular_options,
            modular_profile=modular_profile
        )
        (srf_objective,
         comp_rule_within,
         comp_rule_successive,
         ratio_mode,
         normalized) = _resolve_modular_structure(resolved_modular_options)
    else:
        resolved_modular_options = None
        effective_method = srf_method
        (srf_objective,
         comp_rule_within,
         comp_rule_successive,
         ratio_mode,
         normalized) = _resolve_method_structure(effective_method)

    modular_output_variability = bool(
        is_modular and resolved_modular_options.get('output_type') == 'variability'
    )
    modular_variability_method = (
        resolved_modular_options.get('variability_method', 'sampling')
        if is_modular else 'sampling'
    )
    if modular_variability_method not in {'sampling', 'extreme'}:
        modular_variability_method = 'sampling'
    modular_sampling_size = None
    if is_modular:
        raw_modular_options = modular_options if isinstance(modular_options, dict) else {}
        raw_sampling_size = (
            raw_modular_options.get('sampling_size')
            if 'sampling_size' in raw_modular_options
            else raw_modular_options.get('sample_size')
        )
        try:
            parsed_sampling_size = int(raw_sampling_size)
            if parsed_sampling_size > 0:
                modular_sampling_size = min(parsed_sampling_size, 20000)
        except (TypeError, ValueError):
            modular_sampling_size = None
    modular_dynamic_unit = bool(
        is_modular and resolved_modular_options.get('unit_weight') == 'dynamic'
    )
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
    # Imprecise distance variants sometimes need extra binary logic so gap bounds are
    # enforced conditionally instead of with one global spacing parameter.
    conditional_gap_milp = bool(
        is_modular
        and modular_output_variability
        and resolved_modular_options.get('procedure') in {'standard', 'zero'}
        and resolved_modular_options.get('distance_type') == 'imprecise'
        and comp_rule_successive in {
            'interval-constrained',
            'probability-distribution',
            'hfl-linguistic-interval',
        }
        and not (
            resolved_modular_options.get('procedure') == 'zero'
            and modular_dynamic_unit
            and modular_variability_method == 'sampling'
        )
    )

    # Dynamic unit weight (Q12b) is modeled through fully-flexible successive constraints.
    if modular_dynamic_unit and comp_rule_successive == 'fixed-spacing':
        comp_rule_successive = 'fully-flexible'

    """
    [O6] Extra Constraints
    """
    # Additional constraints are introduced here.
    match srf_method:
        case 'srf' | 'srf_ii' | 'robust_srf' | 'wap' | 'imprecise_srf' | 'belief_degree_imprecise_srf' | 'hfl_srf' | 'modular_srf':
            extra_cond = extra_constraints if isinstance(extra_constraints, dict) else None
        case _:
            raise ValueError('Invalid SRF method')

    if _is_extra_constraints_enabled(extra_cond):
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
            sample_size_hint=modular_sampling_size if is_modular else None
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

            srf_samples, asi_srf_samples = calc_srf_rand_samples(cards_arrangement,
                                                                 z_value,
                                                                 e_value,
                                                                 comp_rule_within=comp_rule_within,
                                                                 comp_rule_successive=comp_rule_successive,
                                                                 ratio_mode=ratio_mode,
                                                                 normalized=normalized,
                                                                 extra_cond=extra_cond,
                                                                 min_delta=min_delta,
                                                                 n_samples=50 * n_crit_cards,
                                                                 conditional_gap_milp=conditional_gap_milp,
                                                                 dynamic_unit_weight=modular_dynamic_unit)

            robustness_rules = {
                asi_srf_min_max: srf_min_max.mean() if srf_min_max is not None else None,
                asi_srf_vertices: srf_vertices.mean() if srf_vertices is not None else None,
                asi_srf_samples: srf_samples.mean() if srf_samples is not None else None
            }
        else:
            # Modular runs may need either full variability outputs or only enough
            # samples to compute a representative center solution.
            needs_distribution = modular_output_variability or modular_center_single_required
            if needs_distribution:
                if modular_output_variability and modular_variability_method == 'extreme':
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
                                                        n_samples=max(12 * n_crit_cards, 40),
                                                        conditional_gap_milp=conditional_gap_milp,
                                                        dynamic_unit_weight=modular_dynamic_unit)
                    if isinstance(srf_vertices, pd.DataFrame) and not srf_vertices.empty:
                        srf_samples = srf_vertices.copy()
                    elif isinstance(srf_min_max, pd.DataFrame) and not srf_min_max.empty:
                        srf_samples = srf_min_max.copy()
                else:
                    probabilistic_or_hfl = (
                        comp_rule_successive in ['probability-distribution', 'hfl-linguistic-interval']
                        or ratio_mode in ['probability-cloud', 'hfl-ratio-interval']
                    )
                    if probabilistic_or_hfl:
                        sample_budget = min(max(12 * n_crit_cards, 80), 300)
                        vertex_budget = min(max(5 * n_crit_cards, 20), 120)
                    else:
                        sample_budget = max(25 * n_crit_cards, 120) if modular_output_variability else max(15 * n_crit_cards, 80)
                        vertex_budget = max(10 * n_crit_cards, 40)

                    if modular_output_variability and isinstance(modular_sampling_size, int):
                        sample_budget = modular_sampling_size

                    if modular_output_variability:
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

                # Sampling can under-cover broad feasible regions in modular imprecise runs.
                # Compute exact min/max envelopes once so reported bounds reflect the true space.
                if (modular_output_variability
                        and modular_variability_method == 'sampling'
                        and resolved_modular_options.get('distance_type') == 'imprecise'
                        and comp_rule_successive in {'interval-constrained', 'probability-distribution', 'hfl-linguistic-interval'}
                        and not (isinstance(srf_min_max, pd.DataFrame) and not srf_min_max.empty)):
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
    else:
        if comp_rule_successive in ['fixed-spacing']:
            pass
        elif srf_method in ['belief_degree_imprecise_srf', 'hfl_srf']:
            # For belief-degree and HFL variants, generate distributions for ASI/PCA
            # with bounded budgets to keep response time practical.
            vertex_budget = min(max(5 * n_crit_cards, 20), 120)
            sample_budget = min(max(12 * n_crit_cards, 80), 300)

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
        else:
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
                                                               n_samples=20 * n_crit_cards,
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
                                                                 n_samples=50 * n_crit_cards,
                                                                 conditional_gap_milp=conditional_gap_milp,
                                                                 dynamic_unit_weight=modular_dynamic_unit)

            # Store ASI values and barycenters of each robustness rule.
            robustness_rules = {
                asi_srf_min_max: srf_min_max.mean() if srf_min_max is not None else None,
                asi_srf_vertices: srf_vertices.mean() if srf_vertices is not None else None,
                asi_srf_samples: srf_samples.mean() if srf_samples is not None else None
            }

    """
    SRF Calculations
    """
    if is_modular and modular_maximize_asi_equivalent:
        simos_calc_results = pd.DataFrame(columns=['r', 'name', 'k_i'],
                                          index=cards_arrangement[cards_arrangement['class'] == 'criterion'].index[::-1])
        simos_calc_results['r'] = cards_arrangement['rank']
        simos_calc_results['name'] = cards_arrangement['name']

        # Select the mean criteria weight based on the max ASI of the three methods.
        asi_value = max(robustness_rules)
        simos_calc_results['k_i'] = robustness_rules[asi_value]

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
        asi_value = max(robustness_rules)
        simos_calc_results['k_i'] = robustness_rules[asi_value]

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
            summary_decimals = max(2, int(w_value))
        except (TypeError, ValueError):
            summary_decimals = 2
        simos_calc_results = _attach_solution_summary_columns(
            simos_calc_results,
            srf_samples=srf_samples,
            srf_min_max=srf_min_max,
            decimals=summary_decimals
        )

    if should_attach_distribution_summary:
        if isinstance(srf_samples, pd.DataFrame) and not srf_samples.empty:
            _persist_distribution_samples(cards_arrangement, srf_samples)
        elif isinstance(srf_min_max, pd.DataFrame) and not srf_min_max.empty:
            _persist_distribution_samples(cards_arrangement, srf_min_max)

    # Export the 2D projection consumed by Plotly. Extreme-only modular mode skips
    # PCA because it does not generate the dense cloud that plot expects.
    skip_pca = bool(is_modular and modular_output_variability and modular_variability_method == 'extreme')
    if not skip_pca:
        pca_vertices = srf_vertices
        if (isinstance(srf_samples, pd.DataFrame)
                and isinstance(srf_vertices, pd.DataFrame)
                and not srf_samples.empty
                and not srf_vertices.empty):
            overlapping_index = set(srf_samples.index).intersection(set(srf_vertices.index))
            if srf_samples is srf_vertices or overlapping_index:
                pca_vertices = None
        calc_pca(srf_samples, selected=simos_calc_results['k_i'], vertices=pca_vertices)

    return simos_calc_results, asi_value


def _add_relaxable_issue(model, issue_vars, issue_meta, issue_id, expr, bound, metadata):
    """
    Adds one relaxable inconsistency issue linked to a binary variable.
    bound='lb' encodes expr >= 0, bound='ub' encodes expr <= 0.
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


def _hit_and_run_polytope(A_ub, b_ub, A_eq, x0, n_samples, burn_in, thinning, rng):
    """
    Hit-and-run sampler over a polytope defined by A_ub x <= b_ub and A_eq x = const.
    """
    basis = _nullspace_matrix(A_eq)
    if basis.size == 0:
        return np.repeat(np.asarray(x0, dtype=float)[None, :], repeats=n_samples, axis=0)

    x_vec = np.asarray(x0, dtype=float).copy()
    samples = []
    eps = 1e-12
    n_target_steps = int(burn_in + thinning * n_samples)
    max_steps = max(n_target_steps * 3, n_target_steps + 500)
    steps = 0

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

    if not samples:
        return np.repeat(np.asarray(x0, dtype=float)[None, :], repeats=n_samples, axis=0)

    if len(samples) < n_samples:
        pad = np.repeat(np.asarray(samples[-1])[None, :], repeats=n_samples - len(samples), axis=0)
        return np.vstack([np.asarray(samples), pad])

    return np.asarray(samples[:n_samples])


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
        for left in range(len(gap_bounds) - 1):
            for right in range(left + 1, len(gap_bounds)):
                left_low, left_high = gap_bounds[left]
                right_low, right_high = gap_bounds[right]
                if left_low > right_high:
                    row = np.zeros(n_ranks)
                    row[left] += 1.0
                    row[left + 1] += -1.0
                    row[right] += -1.0
                    row[right + 1] += 1.0
                    A_rows.append(row)
                    b_rows.append(-delta_frac)
                elif right_low > left_high:
                    row = np.zeros(n_ranks)
                    row[right] += 1.0
                    row[right + 1] += -1.0
                    row[left] += -1.0
                    row[left + 1] += 1.0
                    A_rows.append(row)
                    b_rows.append(-delta_frac)

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
        rng=rng
    )

    crit_indices = list(criteria_cards.index)
    sample_matrix = np.zeros((rank_samples.shape[0], len(crit_indices)), dtype=float)
    for j, crit_idx in enumerate(crit_indices):
        rnk = int(criteria_cards.loc[crit_idx, 'rank'])
        sample_matrix[:, j] = rank_samples[:, rank_pos[rnk]]

    if normalized:
        sample_matrix *= 100.0

    return pd.DataFrame(sample_matrix, columns=crit_indices)


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


def _build_inconsistency_recommendation(issue_id, meta, rank_groups):
    """
    Builds one human-readable recommendation from an active EI issue.
    """
    expr_val = float(meta['expr'].getValue())
    residual = float(meta.get('residual_var').X) if meta.get('residual_var') is not None else abs(expr_val)
    bound = meta['bound']
    direction = 'decrease' if bound == 'lb' else 'increase'
    label = meta.get('label', issue_id)
    current_value = meta.get('current_value')
    if residual <= 1e-9:
        return None

    if meta.get('scale_source') == 'gap':
        scale_ref = _safe_positive(meta.get('scale_var').X if meta.get('scale_var') is not None else 1.0)
        delta_value = max(1, int(np.ceil(residual / scale_ref)))
    elif meta.get('scale_source') == 'ratio':
        den_var = meta.get('den_var')
        den_val = _safe_positive(den_var.X if den_var is not None else 1.0)
        ratio_delta = max(0.0, residual / den_val)
        if meta.get('value_type') == 'int':
            delta_value = max(1, int(np.ceil(ratio_delta)))
        else:
            delta_value = ratio_delta
    else:
        delta_value = 1

    proposed_value = None
    if current_value is not None:
        if direction == 'decrease':
            proposed_value = current_value - delta_value
        else:
            proposed_value = current_value + delta_value

        min_value = meta.get('min_value')
        max_value = meta.get('max_value')
        if direction == 'decrease' and min_value is not None and current_value <= min_value + 1e-9:
            return None
        if direction == 'increase' and max_value is not None and current_value >= max_value - 1e-9:
            return None
        if min_value is not None:
            proposed_value = max(min_value, proposed_value)
        if max_value is not None:
            proposed_value = min(max_value, proposed_value)

        if meta.get('value_type') == 'int':
            proposed_value = int(round(proposed_value))
        else:
            proposed_value = float(round(proposed_value, 3))

    if proposed_value is not None and abs(float(proposed_value) - float(current_value)) <= 1e-9:
        return None

    recommendation_text = (
        f"{direction.capitalize()} {label}"
        + (
            f" from {current_value} to about {proposed_value}."
            if proposed_value is not None else "."
        )
    )

    return {
        'issue_id': issue_id,
        'direction': direction,
        'rank_pair': meta.get('rank_pair'),
        'recommendation': recommendation_text
    }


def _is_configuration_feasible(cards_arrangement,
                               z_value,
                               e_value,
                               srf_method,
                               extra_constraints=None,
                               min_delta=1.0,
                               gap_overrides=None):
    """
    Feasibility oracle on the base SRF model for a concrete configuration.
    """
    (_srf_objective,
     comp_rule_within,
     comp_rule_successive,
     ratio_mode,
     normalized) = _resolve_method_structure(srf_method)

    extra_cond = extra_constraints if isinstance(extra_constraints, dict) else None
    model, weights, rank_groups, criteria_cards, delta = _build_srf_model(
        cards_arrangement,
        z_value,
        e_value,
        comp_rule_within=comp_rule_within,
        comp_rule_successive=comp_rule_successive,
        ratio_mode=ratio_mode,
        normalized=normalized,
        extra_cond=extra_cond,
        min_delta=min_delta,
        launch_smaa=False,
        gap_overrides=gap_overrides
    )
    _optimize_model(model)
    return model.status == GRB.OPTIMAL


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


def _format_exact_recommendation(issue_id, meta, new_value):
    """
    Builds one recommendation record from an exact, feasibility-checked target value.
    """
    bound = meta.get('bound')
    direction = 'decrease' if bound == 'lb' else 'increase'
    label = meta.get('label', issue_id)
    current_value = meta.get('current_value')
    if current_value is not None and abs(float(new_value) - float(current_value)) <= 1e-9:
        return None

    if meta.get('value_type') == 'int':
        current_render = int(round(float(current_value))) if current_value is not None else current_value
        new_render = int(round(float(new_value)))
    else:
        current_render = float(round(float(current_value), 3)) if current_value is not None else current_value
        new_render = float(round(float(new_value), 3))

    return {
        'issue_id': issue_id,
        'direction': direction,
        'rank_pair': meta.get('rank_pair'),
        'recommendation': (
            f"{direction.capitalize()} {label} from {current_render} "
            f"to about {new_render}."
        )
    }


def _find_issue_set_exact_recommendations(cards_arrangement,
                                          z_value,
                                          e_value,
                                          srf_method,
                                          active_issues,
                                          issue_meta,
                                          extra_constraints=None,
                                          min_delta=1.0,
                                          max_total_change=250):
    """
    Solves a discrete E^R-style restoration for one EI subset.
    Currently specialized for SRF-II (gap/e0 changes), with beta=1.
    """
    if srf_method != 'srf_ii' or not active_issues:
        return None

    descriptors = []
    used_keys = set()
    for issue_id in sorted(active_issues):
        meta = issue_meta.get(issue_id, {})
        bound = meta.get('bound')
        if bound not in {'lb', 'ub'}:
            return None
        sign = -1 if bound == 'lb' else 1

        gap_match = re.match(r'^gap_(\d+)_(plus|minus)$', issue_id)
        if gap_match:
            prev_rank = int(gap_match.group(1))
            key = ('gap', prev_rank)
            if key in used_keys:
                return None
            used_keys.add(key)
            descriptors.append({
                'issue_id': issue_id,
                'kind': 'gap',
                'prev_rank': prev_rank,
                'sign': sign,
                'meta': meta
            })
            continue

        if issue_id.startswith('z_linear_'):
            key = ('e0', None)
            if key in used_keys:
                return None
            used_keys.add(key)
            descriptors.append({
                'issue_id': issue_id,
                'kind': 'e0',
                'sign': sign,
                'meta': meta
            })
            continue

        return None

    if not descriptors:
        return None
    if len(descriptors) > 4:
        return None

    observed_gaps = _observed_gap_counts_by_prev_rank(cards_arrangement)
    current_e0 = int(e_value)
    n_parts = len(descriptors)
    min_total = n_parts  # each active EI issue must be changed at least by one unit
    max_total = max(min_total, int(max_total_change))
    beta = float(INCONSISTENCY_RESTORATION_BETA)

    for total_delta in range(min_total, max_total + 1):
        candidate_vectors = []
        for deltas in _iter_positive_compositions(total_delta, n_parts):
            # beta=1 by default, but keep the objective expression explicit.
            objective_value = 0.0
            for descriptor, delta_units in zip(descriptors, deltas):
                if descriptor['kind'] == 'e0':
                    objective_value += beta * float(delta_units)
                else:
                    objective_value += float(delta_units)
            candidate_vectors.append((objective_value, tuple(deltas)))

        candidate_vectors.sort(key=lambda item: (item[0], item[1]))
        for _, deltas in candidate_vectors:
            gap_overrides = {}
            candidate_values = {}
            candidate_e0 = current_e0
            invalid = False

            for descriptor, delta_units in zip(descriptors, deltas):
                meta = descriptor['meta']
                current_value = meta.get('current_value')
                if current_value is None:
                    if descriptor['kind'] == 'gap':
                        current_value = observed_gaps.get(descriptor['prev_rank'], 0)
                    elif descriptor['kind'] == 'e0':
                        current_value = current_e0
                    else:
                        current_value = 0

                proposed_value = float(current_value) + descriptor['sign'] * float(delta_units)
                min_value = meta.get('min_value')
                max_value = meta.get('max_value')
                if min_value is not None and proposed_value < float(min_value) - 1e-9:
                    invalid = True
                    break
                if max_value is not None and proposed_value > float(max_value) + 1e-9:
                    invalid = True
                    break

                if meta.get('value_type') == 'int':
                    proposed_value = int(round(proposed_value))

                if descriptor['kind'] == 'gap':
                    if int(proposed_value) < 0:
                        invalid = True
                        break
                    gap_overrides[int(descriptor['prev_rank'])] = int(proposed_value)
                elif descriptor['kind'] == 'e0':
                    if int(proposed_value) < 0:
                        invalid = True
                        break
                    candidate_e0 = int(proposed_value)
                else:
                    invalid = True
                    break

                candidate_values[descriptor['issue_id']] = proposed_value

            if invalid:
                continue

            if not _is_configuration_feasible(
                cards_arrangement,
                z_value,
                candidate_e0,
                srf_method='srf_ii',
                extra_constraints=extra_constraints,
                min_delta=min_delta,
                gap_overrides=gap_overrides if gap_overrides else None
            ):
                continue

            recommendations = []
            for descriptor in descriptors:
                issue_id = descriptor['issue_id']
                rec = _format_exact_recommendation(issue_id, descriptor['meta'], candidate_values[issue_id])
                if rec is not None:
                    recommendations.append(rec)
            if recommendations:
                return recommendations

    return None


def _find_single_issue_exact_recommendation(cards_arrangement,
                                            z_value,
                                            e_value,
                                            srf_method,
                                            issue_id,
                                            issue_meta,
                                            extra_constraints=None,
                                            min_delta=1.0):
    """
    For one-issue suggestions, find the nearest actually feasible fix by direct feasibility search.
    """
    meta = issue_meta.get(issue_id, {})
    bound = meta.get('bound')
    direction = 'decrease' if bound == 'lb' else 'increase'
    label = meta.get('label', issue_id)

    gap_match = re.match(r'^gap_(\d+)_(plus|minus)$', issue_id)
    if gap_match and srf_method in {'srf', 'srf_ii'}:
        prev_rank = int(gap_match.group(1))
        current_blank = int(meta.get('current_value', 0))
        for step in range(1, 201):
            candidate_blank = current_blank - step if direction == 'decrease' else current_blank + step
            if candidate_blank < 0:
                continue
            if _is_configuration_feasible(
                cards_arrangement,
                z_value,
                e_value,
                srf_method=srf_method,
                extra_constraints=extra_constraints,
                min_delta=min_delta,
                gap_overrides={prev_rank: candidate_blank}
            ):
                return {
                    'issue_id': issue_id,
                    'direction': direction,
                    'rank_pair': meta.get('rank_pair'),
                    'recommendation': (
                        f"{direction.capitalize()} {label} from {current_blank} "
                        f"to about {candidate_blank}."
                    )
                }
        return None

    if issue_id.startswith('z_exact_') and srf_method == 'srf':
        current_z = float(meta.get('current_value', z_value))
        step_size = 0.01
        max_steps = 20_000
        for step in range(1, max_steps + 1):
            candidate_z = current_z - step * step_size if direction == 'decrease' else current_z + step * step_size
            if candidate_z < 1.01:
                continue
            candidate_z = float(round(candidate_z, 3))
            if _is_configuration_feasible(
                cards_arrangement,
                candidate_z,
                e_value,
                srf_method=srf_method,
                extra_constraints=extra_constraints,
                min_delta=min_delta
            ):
                return {
                    'issue_id': issue_id,
                    'direction': direction,
                    'rank_pair': meta.get('rank_pair'),
                    'recommendation': (
                        f"{direction.capitalize()} {label} from {current_z} "
                        f"to about {candidate_z}."
                    )
                }
        return None

    if issue_id.startswith('z_linear_') and srf_method == 'srf_ii':
        current_e0 = int(meta.get('current_value', e_value))
        for step in range(1, 1001):
            candidate_e0 = current_e0 - step if direction == 'decrease' else current_e0 + step
            if candidate_e0 < 0:
                continue
            if _is_configuration_feasible(
                cards_arrangement,
                z_value,
                candidate_e0,
                srf_method=srf_method,
                extra_constraints=extra_constraints,
                min_delta=min_delta
            ):
                return {
                    'issue_id': issue_id,
                    'direction': direction,
                    'rank_pair': meta.get('rank_pair'),
                    'recommendation': (
                        f"{direction.capitalize()} {label} from {current_e0} "
                        f"to about {candidate_e0}."
                    )
                }
        return None

    return None


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
    Iterative E^I-style inconsistency identification and restoration hints.
    Returns up to `max_suggestions` minimal-cardinality alternatives.
    """
    max_suggestions = int(max_suggestions) if str(max_suggestions).strip() != '' else 3
    max_suggestions = max(1, min(max_suggestions, 20))

    (comp_rule_within,
     comp_rule_successive,
     ratio_mode,
     normalized,
     restoration_method) = _resolve_inconsistency_structure(
        srf_method=srf_method,
        modular_options=modular_options,
        modular_profile=modular_profile
    )

    model = gp.Model("SRF_Inconsistency_Identification")
    model.setParam("OutputFlag", 0)

    criteria_cards = cards_arrangement[cards_arrangement['class'] == 'criterion'].sort_values('rank')
    weights = {
        idx: model.addVar(lb=0, name=f"w_{idx}")
        for idx in criteria_cards.index
    }

    # Shared scale variable used for fixed/interval spacing styles.
    if comp_rule_successive in ['fixed-spacing', 'interval-constrained', 'probability-distribution']:
        spacing_scale = model.addVar(lb=max(float(min_delta), 1e-6), name="ei_spacing_scale")
    elif comp_rule_successive == 'hfl-linguistic-interval':
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
    original_z_value = z_value
    original_e_value = e_value

    # Successive-rank constraints as relaxable EI issues.
    for i in range(1, len(sorted_ranks)):
        prev_rank = sorted_ranks[i - 1]
        curr_rank = sorted_ranks[i]
        prev_idx = rank_groups[prev_rank][0]
        curr_idx = rank_groups[curr_rank][0]
        rank_pair = [int(prev_rank), int(curr_rank)]
        observed_gap = int(max(0, rank_white_count[prev_rank] - 1))
        diff_expr = weights[curr_idx] - weights[prev_idx]

        if comp_rule_successive == 'fixed-spacing':
            target = spacing_scale * rank_white_count[prev_rank]
            expr = diff_expr - target
            y_plus = _add_relaxable_issue(
                model, issue_vars, issue_meta,
                issue_id=f"gap_{prev_rank}_plus",
                expr=expr,
                bound='lb',
                metadata={
                    'label': f"blank cards between Rank {prev_rank} and Rank {curr_rank}",
                    'current_value': observed_gap,
                    'value_type': 'int',
                    'min_value': 0,
                    'rank_pair': rank_pair,
                    'scale_source': 'gap',
                    'scale_var': spacing_scale
                }
            )
            y_minus = _add_relaxable_issue(
                model, issue_vars, issue_meta,
                issue_id=f"gap_{prev_rank}_minus",
                expr=expr,
                bound='ub',
                metadata={
                    'label': f"blank cards between Rank {prev_rank} and Rank {curr_rank}",
                    'current_value': observed_gap,
                    'value_type': 'int',
                    'min_value': 0,
                    'rank_pair': rank_pair,
                    'scale_source': 'gap',
                    'scale_var': spacing_scale
                }
            )
            model.addConstr(y_plus + y_minus <= 1, f"ei_gap_{prev_rank}_exclusive")

        elif comp_rule_successive == 'fully-flexible':
            expr = diff_expr - (min_delta * rank_white_count[prev_rank])
            _add_relaxable_issue(
                model, issue_vars, issue_meta,
                issue_id=f"gap_{prev_rank}_min",
                expr=expr,
                bound='lb',
                metadata={
                    'label': f"minimum blank cards gap between Rank {prev_rank} and Rank {curr_rank}",
                    'current_value': observed_gap,
                    'value_type': 'int',
                    'min_value': 0,
                    'rank_pair': rank_pair,
                    'scale_source': 'none',
                    'scale_var': None
                }
            )

        elif comp_rule_successive == 'interval-constrained':
            if rank_white_count[prev_rank] == 1:
                target = spacing_scale * rank_white_count[prev_rank]
                expr = diff_expr - target
                y_plus = _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"gap_{prev_rank}_plus",
                    expr=expr,
                    bound='lb',
                    metadata={
                        'label': f"blank cards between Rank {prev_rank} and Rank {curr_rank}",
                        'current_value': observed_gap,
                        'value_type': 'int',
                        'min_value': 0,
                        'rank_pair': rank_pair,
                        'scale_source': 'gap',
                        'scale_var': spacing_scale
                    }
                )
                y_minus = _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"gap_{prev_rank}_minus",
                    expr=expr,
                    bound='ub',
                    metadata={
                        'label': f"blank cards between Rank {prev_rank} and Rank {curr_rank}",
                        'current_value': observed_gap,
                        'value_type': 'int',
                        'min_value': 0,
                        'rank_pair': rank_pair,
                        'scale_source': 'gap',
                        'scale_var': spacing_scale
                    }
                )
                model.addConstr(y_plus + y_minus <= 1, f"ei_gap_{prev_rank}_exclusive")
            else:
                e_min = int(e_value[f'emin_{prev_rank}'])
                e_max = int(e_value[f'emax_{prev_rank}'])
                expr_min = diff_expr - spacing_scale * (e_min + 1)
                expr_max = diff_expr - spacing_scale * (e_max + 1)
                _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"gap_min_{prev_rank}",
                    expr=expr_min,
                    bound='lb',
                    metadata={
                        'label': f"minimum blank cards between Rank {prev_rank} and Rank {curr_rank}",
                        'current_value': e_min,
                        'value_type': 'int',
                        'min_value': 0,
                        'rank_pair': rank_pair,
                        'scale_source': 'gap',
                        'scale_var': spacing_scale
                    }
                )
                _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"gap_max_{prev_rank}",
                    expr=expr_max,
                    bound='ub',
                    metadata={
                        'label': f"maximum blank cards between Rank {prev_rank} and Rank {curr_rank}",
                        'current_value': e_max,
                        'value_type': 'int',
                        'min_value': 0,
                        'rank_pair': rank_pair,
                        'scale_source': 'gap',
                        'scale_var': spacing_scale
                    }
                )

        elif comp_rule_successive == 'probability-distribution':
            if rank_white_count[prev_rank] == 1:
                target = spacing_scale * rank_white_count[prev_rank]
                expr = diff_expr - target
                y_plus = _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"gap_{prev_rank}_plus",
                    expr=expr,
                    bound='lb',
                    metadata={
                        'label': f"blank cards between Rank {prev_rank} and Rank {curr_rank}",
                        'current_value': observed_gap,
                        'value_type': 'int',
                        'min_value': 0,
                        'rank_pair': rank_pair,
                        'scale_source': 'gap',
                        'scale_var': spacing_scale
                    }
                )
                y_minus = _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"gap_{prev_rank}_minus",
                    expr=expr,
                    bound='ub',
                    metadata={
                        'label': f"blank cards between Rank {prev_rank} and Rank {curr_rank}",
                        'current_value': observed_gap,
                        'value_type': 'int',
                        'min_value': 0,
                        'rank_pair': rank_pair,
                        'scale_source': 'gap',
                        'scale_var': spacing_scale
                    }
                )
                model.addConstr(y_plus + y_minus <= 1, f"ei_gap_{prev_rank}_exclusive")
            else:
                e_cloud = _extract_probability_pairs(
                    e_value,
                    value_prefix=f"e-value-{prev_rank}-",
                    beta_prefix=f"e-beta-{prev_rank}-",
                )
                if not e_cloud:
                    e_cloud = {float(observed_gap): 1.0}
                e_cloud = _normalize_probability_cloud(e_cloud)
                e_min = int(np.floor(min(e_cloud.keys())))
                e_max = int(np.ceil(max(e_cloud.keys())))

                expr_min = diff_expr - spacing_scale * (e_min + 1)
                expr_max = diff_expr - spacing_scale * (e_max + 1)
                _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"gap_support_min_{prev_rank}",
                    expr=expr_min,
                    bound='lb',
                    metadata={
                        'label': f"minimum blank-card support between Rank {prev_rank} and Rank {curr_rank}",
                        'current_value': e_min,
                        'value_type': 'int',
                        'min_value': 0,
                        'rank_pair': rank_pair,
                        'scale_source': 'gap',
                        'scale_var': spacing_scale
                    }
                )
                _add_relaxable_issue(
                    model, issue_vars, issue_meta,
                    issue_id=f"gap_support_max_{prev_rank}",
                    expr=expr_max,
                    bound='ub',
                    metadata={
                        'label': f"maximum blank-card support between Rank {prev_rank} and Rank {curr_rank}",
                        'current_value': e_max,
                        'value_type': 'int',
                        'min_value': 0,
                        'rank_pair': rank_pair,
                        'scale_source': 'gap',
                        'scale_var': spacing_scale
                    }
                )

        elif comp_rule_successive == 'hfl-linguistic-interval':
            r_min_term = int(e_value.get(f'rmin_{prev_rank}', 1))
            r_max_term = int(e_value.get(f'rmax_{prev_rank}', r_min_term))
            r_min = _map_hfl_card_term(r_min_term)
            r_max = _map_hfl_card_term(r_max_term)

            expr_min = diff_expr - spacing_scale * r_min
            expr_max = diff_expr - spacing_scale * r_max
            _add_relaxable_issue(
                model, issue_vars, issue_meta,
                issue_id=f"hfl_gap_min_{prev_rank}",
                expr=expr_min,
                bound='lb',
                metadata={
                    'label': f"HFL lower gap term between Rank {prev_rank} and Rank {curr_rank}",
                    'current_value': r_min,
                    'value_type': 'int',
                    'min_value': HFL_CARD_MIN_TERM,
                    'max_value': HFL_CARD_MAX_TERM,
                    'rank_pair': rank_pair,
                    'scale_source': 'gap',
                    'scale_var': spacing_scale
                }
            )
            _add_relaxable_issue(
                model, issue_vars, issue_meta,
                issue_id=f"hfl_gap_max_{prev_rank}",
                expr=expr_max,
                bound='ub',
                metadata={
                    'label': f"HFL upper gap term between Rank {prev_rank} and Rank {curr_rank}",
                    'current_value': r_max,
                    'value_type': 'int',
                    'min_value': HFL_CARD_MIN_TERM,
                    'max_value': HFL_CARD_MAX_TERM,
                    'rank_pair': rank_pair,
                    'scale_source': 'gap',
                    'scale_var': spacing_scale
                }
            )

    # Ratio issues
    min_index = rank_groups[min(sorted_ranks)][0]
    max_index = rank_groups[max(sorted_ranks)][0]
    min_weight_var = weights[min_index]

    if ratio_mode == 'exact-ratio':
        z_exact = float(z_value)
        expr = weights[max_index] - z_exact * weights[min_index]
        y_plus = _add_relaxable_issue(
            model, issue_vars, issue_meta,
            issue_id="z_exact_plus",
            expr=expr,
            bound='lb',
            metadata={
                'label': "z ratio",
                'current_value': z_exact,
                'value_type': 'float',
                'min_value': 1.01,
                'scale_source': 'ratio',
                'den_var': min_weight_var
            }
        )
        y_minus = _add_relaxable_issue(
            model, issue_vars, issue_meta,
            issue_id="z_exact_minus",
            expr=expr,
            bound='ub',
            metadata={
                'label': "z ratio",
                'current_value': z_exact,
                'value_type': 'float',
                'min_value': 1.01,
                'scale_source': 'ratio',
                'den_var': min_weight_var
            }
        )
        model.addConstr(y_plus + y_minus <= 1, "ei_z_exact_exclusive")

    elif ratio_mode == 'linear-spacing':
        e0_anchor = _estimate_linear_spacing_e0_anchor(e_value)
        z_linear = (
            (cards_arrangement['rank'].max() - 1)
            + cards_arrangement['class'].to_list().count('white')
            + (e0_anchor + 1)
        ) / (e0_anchor + 1)
        expr = weights[max_index] - z_linear * weights[min_index]
        y_plus = _add_relaxable_issue(
            model, issue_vars, issue_meta,
            issue_id="z_linear_plus",
            expr=expr,
            bound='lb',
            metadata={
                'label': "e0 value (SRF-II)",
                'current_value': int(round(e0_anchor)),
                'value_type': 'int',
                'min_value': 0,
                'scale_source': 'none',
                'den_var': min_weight_var
            }
        )
        y_minus = _add_relaxable_issue(
            model, issue_vars, issue_meta,
            issue_id="z_linear_minus",
            expr=expr,
            bound='ub',
            metadata={
                'label': "e0 value (SRF-II)",
                'current_value': int(round(e0_anchor)),
                'value_type': 'int',
                'min_value': 0,
                'scale_source': 'none',
                'den_var': min_weight_var
            }
        )
        model.addConstr(y_plus + y_minus <= 1, "ei_z_linear_exclusive")

    elif ratio_mode == 'interval-total':
        z_min = float(z_value['zmin'])
        z_max = float(z_value['zmax'])
        expr_min = weights[max_index] - z_min * weights[min_index]
        expr_max = weights[max_index] - z_max * weights[min_index]
        _add_relaxable_issue(
            model, issue_vars, issue_meta,
            issue_id="z_interval_min",
            expr=expr_min,
            bound='lb',
            metadata={
                'label': "z lower bound",
                'current_value': z_min,
                'value_type': 'float',
                'min_value': 1.01,
                'scale_source': 'ratio',
                'den_var': min_weight_var
            }
        )
        _add_relaxable_issue(
            model, issue_vars, issue_meta,
            issue_id="z_interval_max",
            expr=expr_max,
            bound='ub',
            metadata={
                'label': "z upper bound",
                'current_value': z_max,
                'value_type': 'float',
                'min_value': 1.01,
                'scale_source': 'ratio',
                'den_var': min_weight_var
            }
        )

    elif ratio_mode == 'interval-successive':
        for rank in range(1, cards_arrangement['rank'].max()):
            curr_rank = rank_groups[rank][0]
            next_rank = rank_groups[rank + 1][0]
            z_min = float(z_value[f'zmin_{rank}'])
            z_max = float(z_value[f'zmax_{rank}'])
            den_var = weights[curr_rank]
            expr_min = weights[next_rank] - z_min * weights[curr_rank]
            expr_max = weights[next_rank] - z_max * weights[curr_rank]
            _add_relaxable_issue(
                model, issue_vars, issue_meta,
                issue_id=f"z_successive_min_{rank}",
                expr=expr_min,
                bound='lb',
                metadata={
                    'label': f"z lower bound for Rank {rank + 1} / Rank {rank}",
                    'current_value': z_min,
                    'value_type': 'float',
                    'min_value': 1.01,
                    'rank_pair': [rank, rank + 1],
                    'scale_source': 'ratio',
                    'den_var': den_var
                }
            )
            _add_relaxable_issue(
                model, issue_vars, issue_meta,
                issue_id=f"z_successive_max_{rank}",
                expr=expr_max,
                bound='ub',
                metadata={
                    'label': f"z upper bound for Rank {rank + 1} / Rank {rank}",
                    'current_value': z_max,
                    'value_type': 'float',
                    'min_value': 1.01,
                    'rank_pair': [rank, rank + 1],
                    'scale_source': 'ratio',
                    'den_var': den_var
                }
            )

    elif ratio_mode == 'probability-cloud':
        z_cloud = _extract_probability_pairs(
            z_value,
            value_prefix='z-value-',
            beta_prefix='z-beta-'
        )
        z_cloud = _normalize_probability_cloud(z_cloud)
        z_min = float(min(z_cloud.keys()))
        z_max = float(max(z_cloud.keys()))
        expr_min = weights[max_index] - z_min * weights[min_index]
        expr_max = weights[max_index] - z_max * weights[min_index]
        _add_relaxable_issue(
            model, issue_vars, issue_meta,
            issue_id="z_support_min",
            expr=expr_min,
            bound='lb',
            metadata={
                'label': "minimum z support",
                'current_value': z_min,
                'value_type': 'float',
                'min_value': 1.01,
                'scale_source': 'ratio',
                'den_var': min_weight_var
            }
        )
        _add_relaxable_issue(
            model, issue_vars, issue_meta,
            issue_id="z_support_max",
            expr=expr_max,
            bound='ub',
            metadata={
                'label': "maximum z support",
                'current_value': z_max,
                'value_type': 'float',
                'min_value': 1.01,
                'scale_source': 'ratio',
                'den_var': min_weight_var
            }
        )

    elif ratio_mode == 'hfl-ratio-interval':
        if isinstance(z_value, dict):
            z_min_term = int(z_value.get('emin', z_value.get('zmin', HFL_Z_MIN_TERM)))
            z_max_term = int(z_value.get('emax', z_value.get('zmax', z_min_term)))
        else:
            z_min_term = int(float(z_value))
            z_max_term = int(float(z_value))
        z_min = _map_hfl_z_term(z_min_term)
        z_max = _map_hfl_z_term(z_max_term)

        expr_min = weights[max_index] - z_min * weights[min_index]
        expr_max = weights[max_index] - z_max * weights[min_index]
        _add_relaxable_issue(
            model, issue_vars, issue_meta,
            issue_id="hfl_z_min",
            expr=expr_min,
            bound='lb',
            metadata={
                'label': "HFL lower z term",
                'current_value': z_min,
                'value_type': 'int',
                'min_value': HFL_Z_MIN_TERM,
                'max_value': HFL_Z_MAX_TERM,
                'scale_source': 'ratio',
                'den_var': min_weight_var
            }
        )
        _add_relaxable_issue(
            model, issue_vars, issue_meta,
            issue_id="hfl_z_max",
            expr=expr_max,
            bound='ub',
            metadata={
                'label': "HFL upper z term",
                'current_value': z_max,
                'value_type': 'int',
                'min_value': HFL_Z_MIN_TERM,
                'max_value': HFL_Z_MAX_TERM,
                'scale_source': 'ratio',
                'den_var': min_weight_var
            }
        )

    # Hard normalization.
    if normalized:
        model.addConstr(gp.quicksum(weights.values()) == 100, "ei_normalization")

    # Keep optional requirements hard in EI analysis when enabled.
    if extra_constraints is not None:
        _add_optional_extra_constraints(model, weights, criteria_cards, extra_constraints)

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

    suggestions = []
    first_cardinality = None
    for idx in range(max_suggestions):
        _optimize_model(model)
        if model.status != GRB.OPTIMAL:
            break

        active_issues = [
            issue_id for issue_id, var in issue_vars.items()
            if var.X > 0.5
        ]
        cardinality_value = len(active_issues)
        if first_cardinality is None:
            first_cardinality = cardinality_value

        if cardinality_value == 0:
            break

        exact_issue_set_recommendations = _find_issue_set_exact_recommendations(
            cards_arrangement=cards_arrangement,
            z_value=original_z_value,
            e_value=original_e_value,
            srf_method=restoration_method,
            active_issues=sorted(active_issues),
            issue_meta=issue_meta,
            extra_constraints=extra_constraints,
            min_delta=min_delta
        )
        if restoration_method == 'srf_ii':
            # For SRF-II, only keep feasibility-checked restoration suggestions.
            recommendations = exact_issue_set_recommendations or []
        elif exact_issue_set_recommendations is None:
            recommendations = [
                _build_inconsistency_recommendation(issue_id, issue_meta[issue_id], rank_groups)
                for issue_id in sorted(active_issues)
            ]

            if len(active_issues) == 1:
                exact_rec = _find_single_issue_exact_recommendation(
                    cards_arrangement=cards_arrangement,
                    z_value=original_z_value,
                    e_value=original_e_value,
                    srf_method=restoration_method,
                    issue_id=active_issues[0],
                    issue_meta=issue_meta,
                    extra_constraints=extra_constraints,
                    min_delta=min_delta
                )
                if exact_rec is not None:
                    recommendations = [exact_rec]
        else:
            recommendations = exact_issue_set_recommendations
        recommendations = [rec for rec in recommendations if rec is not None]

        if recommendations:
            suggestions.append({
                'suggestion_id': len(suggestions) + 1,
                'minimal_changes': cardinality_value,
                'recommendations': recommendations
            })

        model.addConstr(
            gp.quicksum(issue_vars[issue_id] for issue_id in active_issues) <= len(active_issues) - 1,
            f"ei_nogood_{idx + 1}"
        )

    return {
        'detected': len(suggestions) > 0,
        'message': (
            "Input preferences are inconsistent. "
            "Apply one of the minimal adjustment sets below and re-run."
            if suggestions else
            "No actionable inconsistency recommendation could be generated."
        ),
        'requested_suggestions': max_suggestions,
        'returned_suggestions': len(suggestions),
        'minimal_inconsistency_size': int(first_cardinality) if first_cardinality is not None else None,
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
        match comp_rule_successive:
            case 'fixed-spacing':
                model.addConstr(
                    weights[curr_index] - weights[prev_index] == delta * rank_white_count[prev_rank],
                    f"successive_fixed_{prev_rank}_{curr_rank}"
                )
            case 'fully-flexible':
                model.addConstr(
                    weights[curr_index] - weights[prev_index] >= min_delta * rank_white_count[prev_rank],
                    f"successive_flexible_{prev_rank}_{curr_rank}"
                )
            case 'interval-constrained':
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
            case 'probability-distribution':
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
            case 'hfl-linguistic-interval':
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
            case _:
                raise ValueError('Invalid rule for comparison of successive ranks')

        # >>> SMAA <<<
        if launch_smaa:
            match comp_rule_successive:
                case 'fully-flexible':
                    if ratio_mode != 'interval-successive':
                        model.addConstr(
                            weights[curr_index] - weights[prev_index] == np.random.uniform(min_delta * rank_white_count[prev_rank], 100),
                            f"successive_smaa_{prev_rank}_{curr_rank}"
                        )
                case 'interval-constrained':
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
                case 'probability-distribution':
                    e_values_rank = e_probability_cloud.get(prev_rank, {})
                    if e_values_rank:
                        e_value_sample = np.random.uniform(min(e_values_rank.keys()), max(e_values_rank.keys()))
                    else:
                        e_value_sample = float(rank_white_count[prev_rank] - 1)
                    model.addConstr(
                        weights[curr_index] - weights[prev_index] == gap_delta_scale * (e_value_sample + 1),
                        f"successive_smaa_{prev_rank}_{curr_rank}"
                    )
                case 'hfl-linguistic-interval':
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
    match ratio_mode:
        case 'exact-ratio':
            model.addConstr(
                weights[max_index] == z_value * weights[min_index],
                "z_ratio_constraint"
            )
        case 'linear-spacing':
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
        case 'interval-successive':
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
        case 'interval-total':
            model.addConstr(
                weights[max_index] >= z_value['zmin'] * weights[min_index],
                "z_ratio_constraint_min"
            )

            model.addConstr(
                weights[max_index] <= z_value['zmax'] * weights[min_index],
                "z_ratio_constraint_max"
            )
        case 'probability-cloud':
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
        case 'hfl-ratio-interval':
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
        case _:
            raise ValueError('Invalid z ratio mode')

    # >>> SMAA <<<
    if launch_smaa:
        match ratio_mode:
            case 'interval-successive':
                for rank in range(1, cards_arrangement['rank'].max()):
                    curr_rank = rank_groups[rank][0]
                    next_rank = rank_groups[rank + 1][0]

                    z_value_sample = np.random.uniform(z_value[f'zmin_{rank}'], z_value[f'zmax_{rank}'])
                    model.addConstr(
                        weights[next_rank] == z_value_sample * weights[curr_rank],
                        f"z_ratio_constraint_smaa_{rank}"
                    )
            case 'interval-total':
                z_value_sample = np.random.uniform(z_value['zmin'], z_value['zmax'])
                model.addConstr(
                    weights[max_index] == z_value_sample * weights[min_index],
                    "z_ratio_constraint_smaa"
                )
            case 'probability-cloud':
                z_value_sample = np.random.uniform(min(z_values_pb.keys()), max(z_values_pb.keys()))
                model.addConstr(
                    weights[max_index] == z_value_sample * weights[min_index],
                    "z_ratio_constraint_smaa"
                )
            case 'hfl-ratio-interval':
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
            case 'linear-spacing':
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
    Generates random samples of criteria weights using repeated solves on a free MILP solver.

    Instead of solving the problem multiple times in parallel, this approach:
    1. Builds the constraint model once
    2. Uses repeated randomized objectives to generate multiple feasible solutions
    3. Returns these as a DataFrame of weight samples

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

    # Fallback/top-up: if probabilistic draws are sparse, sample feasible points directly.
    if len(results) < n_samples:
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

    # Interior-point enrichment for belief-degree and HFL:
    # LP-based sampling tends to return many vertices; mix feasible solutions to
    # guarantee points inside the polyhedron for PCA clouds.
    if comp_rule_successive in ['probability-distribution', 'hfl-linguistic-interval'] and len(results) >= 2:
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

    # Final densification: if still sparse, keep adding interior convex combinations.
    seen_signatures = {_solution_signature(solution) for solution in results}
    if len(results) < n_samples and len(results) >= 2:
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

    # Convert to DataFrame and export into a JSON file
    srf_samples = pd.DataFrame(results)
    srf_samples.rename(columns=cards_arrangement['name']).to_json(
        str(SRF_SAMPLES_PATH), orient='records'
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

    results = []
    for _ in range(n_samples):
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
    for idx in weights:
        # Solve for a minimum criterion weight
        model.setObjective(weights[idx], GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            # Extract and record weights for this solution
            solution = {idx: weights[idx].X for idx in criteria_cards.index}
            results.append(solution)

        # Solve for a maximum criterion weight
        model.setObjective(weights[idx], GRB.MAXIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            # Extract and record weights for this solution
            solution = {idx: weights[idx].X for idx in criteria_cards.index}
            results.append(solution)

    # Convert to DataFrame
    srf_min_max = pd.DataFrame(results)

    # Calculate the ASI value
    asi_srf_min_max = calc_asi(srf_min_max)

    return srf_min_max, asi_srf_min_max

