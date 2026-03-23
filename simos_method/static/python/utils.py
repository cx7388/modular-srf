from math import floor, sqrt
from pathlib import Path
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
PCA_OUTPUT_PATH = DATA_DIR / 'pca_output.json'


"""
FUNCTION FOR ROUNDING-UP OF CRITERIA WEIGHTS TO ENSURE THEIR SUMMATION INTO 100
"""


def round_up_selected(weights_input, w_value, target_sum=100):
    """
    This function replicates the logic outlined in Figueira and Roy (2002) to round up the selected criteria.
    It is necessary because when we apply selected decimal points, we may end up with a situation where the sum
    of all weights is not equal to exactly target_sum.
    
    Args:
        weights_input (pd.Series): Input weights to be rounded
        w_value (int): Decimal precision for rounding
        target_sum (float, optional): Target sum for the weights. Defaults to 100.
        
    Returns:
        pd.Series: Rounded weights that sum to target_sum
    """
    weights_rounded = pd.DataFrame(columns=['k_r', 'k_i_star', 'k_i_quote', 'k_i'], index=weights_input.index)
    weights_rounded['k_r'] = weights_input

    weights_rounded['k_i_star'] = weights_rounded['k_r'] / weights_rounded['k_r'].sum() * target_sum
    weights_rounded['k_i_quote'] = (weights_rounded['k_i_star'] * 10 ** w_value).apply(floor) / 10 ** w_value
    weights_rounded['k_i'] = weights_rounded['k_i_quote']

    weights_rounded['d_i'] = (10 ** (-w_value) - (weights_rounded['k_i_star'] - weights_rounded['k_i_quote'])) / weights_rounded['k_i_star']
    weights_rounded['d_i_hat'] = (weights_rounded['k_i_star'] - weights_rounded['k_i_quote']) / weights_rounded['k_i_star']

    # define a full set of criteria
    full_set = set(weights_rounded.index)

    # define the subsets M, L, and L_hat
    subset_m = set(weights_rounded[(weights_rounded['d_i'] - weights_rounded['d_i_hat']) > 0].index)
    subset_l = weights_rounded.loc[list(full_set - subset_m), :].sort_values(by=['d_i']).index
    subset_l_hat = weights_rounded.loc[list(full_set - subset_m), :].sort_values(by=['d_i_hat'], ascending=False).index

    # determine the size of different subsets
    size_n = len(full_set)
    size_m = len(subset_m)
    size_v = round((target_sum - weights_rounded.loc[:, 'k_i_quote'].sum()) / 10 ** (-w_value))

    # identify weights of which criteria should be rounded up
    if size_m + size_v <= size_n:
        f_plus = list(subset_l_hat[:size_v])
    else:
        f_minus = set(subset_l[-(size_n - size_v):])
        f_plus = list(full_set - f_minus)

    # to ensure that we do not round up extra weights
    if len(f_plus) > size_v:
        f_plus = f_plus[:size_v]

    # round up the selected criteria
    weights_rounded.loc[f_plus, 'k_i'] = weights_rounded.loc[f_plus, 'k_i_quote'] + 10 ** (-w_value)

    return round(weights_rounded['k_i'], w_value)


"""
SECONDARY FUNCTIONS FOR MISCELLANEOUS CALCULATIONS
"""


def calc_asi(srf_samples):
    """
    Calculates the Average Stability Index (ASI) for a set of weight samples.
    
    The ASI measures how stable the weight assignments are across multiple
    random samples. Values closer to 1 indicate higher stability.
    
    Args:
        srf_samples (pd.DataFrame): Matrix of weight samples
        
    Returns:
        float: ASI value rounded to 3 decimal places
    """
    if srf_samples is None:
        return None
    if not isinstance(srf_samples, pd.DataFrame) or srf_samples.empty:
        return None

    srf_samples = srf_samples.copy().apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')
    if srf_samples.empty:
        return None

    srf_samples = srf_samples / 100

    m, n = srf_samples.shape  # number of instances, number of criteria
    if m <= 0 or n <= 1:
        return None

    denom = (m / n) * sqrt(n - 1)
    if abs(denom) <= 1e-12:
        return None

    asi_value = 1 - (1 / n) * (
            np.sqrt(m * (srf_samples ** 2).sum() - (srf_samples.sum() ** 2))
            / denom
    ).sum()

    return float(round(asi_value, 3))


def calc_pca(srf_samples, selected=None, vertices=None):
    """
    Performs Principal Component Analysis on weight samples.
    
    Reduces the dimensionality of the weight samples to 2D for visualization.
    If a selected weight vector is provided, it is included in the analysis.
    
    Args:
        srf_samples (pd.DataFrame): Matrix of weight samples
        selected (pd.Series, optional): Selected weight vector to include
        
    Returns:
        None: Results are saved to a JSON file for frontend visualization
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if srf_samples is None:
        PCA_OUTPUT_PATH.write_text("{}", encoding="utf-8")
        return

    srf_samples = srf_samples.copy()

    if selected is not None:
        srf_samples = pd.concat([srf_samples, selected.to_frame().T]).rename(index={'k_i': 'selected'})

    if vertices is not None:
        srf_samples = pd.concat([srf_samples, vertices])

    numeric_samples = srf_samples.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')
    if numeric_samples.empty:
        PCA_OUTPUT_PATH.write_text("{}", encoding="utf-8")
        return

    # PCA requires at least 2 rows and 2 columns. If not available, export a degenerate projection.
    if min(numeric_samples.shape[0], numeric_samples.shape[1]) < 2:
        df_pca = pd.DataFrame(
            {'PC1': np.zeros(len(numeric_samples)), 'PC2': np.zeros(len(numeric_samples))},
            index=numeric_samples.index
        )
    else:
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(numeric_samples)
        df_pca = pd.DataFrame(x_pca, columns=['PC1', 'PC2'], index=numeric_samples.index)

    df_pca.to_json(str(PCA_OUTPUT_PATH), orient='index')

    return
