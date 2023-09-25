from itertools import combinations
import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import wasserstein_distance as wasser_dist
from scipy.stats import norm
from data_utils import dropnan, get_lower_upper, lower_upper_scale
from data_utils import fill_missing_comb

from typing import List, Tuple, Dict, Optional, Union 

def value_counts_array(x_array: np.ndarray) -> pd.DataFrame:
    x_series = pd.Series(x_array)
    counts = x_series.value_counts(dropna=False).to_frame('count')
    pct = 100 * x_series.value_counts(normalize=True, dropna=False).to_frame('pct')

    return pd.merge(counts, pct, left_index=True, right_index=True)

def value_counts_dict(data_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    value_counts_dict = {}
    for name, data in data_dict.items():
        value_counts_dict[name] = value_counts_array(data)
    value_counts_df = (
        pd.concat(value_counts_dict)
        .reset_index()
        .rename(columns={'level_0': 'name', 'level_1': 'value'})
    )
    value_counts_df = fill_missing_comb(value_counts_df)
    return value_counts_df

def get_cdf(data_dict: Dict[str, np.ndarray], steps: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    all_values = np.concatenate(list(data_dict.values()))
    all_values = dropnan(all_values)
    start = all_values.min()
    stop = all_values.max()
    x = np.linspace(start, stop, steps)
    cdf_dict = {}
    for name, values in data_dict.items():
        cdf = ECDF(values)
        cdf_dict[name] = (x, 100*cdf(x))
    return cdf_dict

def get_inv_cdf(data_dict: Dict[str, np.ndarray], quantiles: List[float]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if isinstance(quantiles, list):
        quantiles_array = np.array(quantiles)
    else:
        quantiles_array = quantiles

    inv_cdf_dict = {}
    for name, values in data_dict.items():
        values = values[~np.isnan(values)]
        inv_cdf = np.quantile(values, quantiles_array)
        inv_cdf_dict[name] = (100*quantiles_array, inv_cdf)

    return inv_cdf_dict

def calc_wasserstein(
    data_dict: Dict[str, np.ndarray], 
    normalize: bool=True, 
    q_tuple: Optional[Tuple[float, float]]=None, 
    lower_upper_tuple: Optional[Tuple[float, float]]=None
) -> pd.DataFrame:
    if normalize is True:
        if lower_upper_tuple is None:
            lower, upper = get_lower_upper(data_dict, q_tuple)
        else:
            lower, upper = lower_upper_tuple
    names_list = list(data_dict.keys())
    names_combs = combinations(names_list, 2)
    distance_df = pd.DataFrame(names_combs, columns=['data1', 'data2'])
    distance_df['distance'] = 0
    distance_df.set_index(['data1', 'data2'], inplace=True)
    for data1, data2 in distance_df.index:
        x1, x2 = data_dict[data1], data_dict[data2]
        x1, x2 = dropnan(x1), dropnan(x2)
        if normalize is True:
            x1 = lower_upper_scale(x1, lower, upper) 
            x2 = lower_upper_scale(x2, lower, upper)
        dist = wasser_dist(x1, x2)
        distance_df.loc[(data1, data2), 'distance'] = dist
        
    return distance_df 


def calc_total_variation(data_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    names_list = list(data_dict.keys())
    names_combs = combinations(names_list, 2)
    distance_df = pd.DataFrame(names_combs, columns=['data1', 'data2'])
    distance_df['distance'] = 0
    distance_df.set_index(['data1', 'data2'], inplace=True)
    value_counts_df = value_counts_dict(data_dict)


    for data1, data2 in distance_df.index:
        weights_1 = (
            value_counts_df
            .query(f'name == "{data1}"')
            .sort_values(by='value')
            ['pct'].values
        )

        weights_2 = (
            value_counts_df
            .query(f'name == "{data2}"')
            .sort_values(by='value')
            ['pct'].values
        )
        dist = np.abs(weights_1 - weights_2).sum()
        distance_df.loc[(data1, data2), 'distance'] = dist
        
    return distance_df/100

def get_target_proportion(
    data: pd.DataFrame, 
    feature_col: str, 
    target: str, 
    alpha: float, 
    target_class: Union[str, int, float]
) -> pd.DataFrame:
    confidence_factor = -norm.ppf(alpha/2)
    proportions = (data.groupby(feature_col)
        [target]
        .agg(target_class_size=lambda y: (y == target_class).sum(),
             bin_size=lambda y: len(y)
        )
    )
    proportions['proportion'] = proportions.eval('target_class_size / bin_size')
    proportions['variance'] = proportions.eval('proportion*(1-proportion)')
    lower_eq = 'proportion - @confidence_factor * sqrt(variance/bin_size)'
    upper_eq = 'proportion + @confidence_factor * sqrt(variance/bin_size)'
    proportions['lower'] = np.maximum(proportions.eval(lower_eq), 0)
    proportions['upper'] = np.minimum(proportions.eval(upper_eq), 1)
    prop_cols = ['lower', 'upper', 'proportion']
    proportions.loc[:, prop_cols] =  100 * proportions[prop_cols]

    return proportions 