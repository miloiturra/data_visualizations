import pandas as pd
from typing import Union, List, Dict, Optional, Callable
from functools import partial
import numpy as np
from scipy.stats import norm as normal_dist
from .special_types import AggFunction
from .categorical_utils import categorify_feature, trim_categories

def get_cum_pct(data: pd.Series):
    return (
        data
        .sort_values(ascending=False)
        .cumsum()
        .to_frame('cum_pct_rows')
    )

def custom_agg(group: pd.DataFrame, agg_funcs: AggFunction) -> pd.Series:
    """
    Apply custom aggregation functions to a DataFrame group.

    Args:
        group (pd.DataFrame): The group to which aggregation functions are applied.
        agg_funcs (Dict[str, callable]): A dictionary of aggregation functions to apply.

    Returns:
        pd.Series: A Series containing the aggregation results for each function.

    The function iterates through the aggregation functions defined in `agg_funcs` and
    applies them to the 'target' column in the group. It returns a Series with the
    aggregation results.
    """
    result: Dict[str, float] = {}
    for name, func in agg_funcs.items():
        result[name] = func(group)
    return pd.Series(result)

def compute_quantile(df: pd.DataFrame, column: str, quantile: float) -> float:
    return df[column].quantile(quantile)

def get_agg_effects(
    data: pd.DataFrame, 
    features: List[str],
    target: Optional[str]=None,
    categorical_target_class: Optional[str]=None,
    confidence_alpha: float = 0.1,
    categorical_features: Optional[List[str]]=None,
    agg_funcs: Optional[Dict[str, AggFunction]]=None, 
    quantiles: List[float]=[0.2, 0.8],
    sample_frac: float=1,
    quantile_bins: bool=True,
    bins: Union[int, List[float]]=10,
    precision: int=1,
    max_n_categories: Optional[int]=None,
    categories_recall_pct: Optional[float]=None,
    keep_nan: Optional[bool]=None,
) -> pd.DataFrame:
    
    if agg_funcs is None and target is None:
        raise ValueError('Either agg_funcs or target must be specified.')
    elif agg_funcs is None and target is not None:
        agg_funcs = dict()

    if len(data) > 1e6 and sample_frac == 1:
        print(
            f'Warning: data has {round(len(data) / 1e6, 1)}MM rows.' 
            'Consider using a sample of the data.'
        )
        
    sample_data = data.sample(frac=sample_frac)
    #add additional functions to compute over groups
    if categorical_target_class is None:
        agg_funcs['mean'] = lambda df: df[target].mean()
        agg_funcs['std'] = lambda df: df[target].std()
        agg_funcs['median'] = lambda df: df[target].median()
        for q in quantiles: 
            agg_funcs[f'q{int(100*q)}'] = partial(
                compute_quantile, column=target, quantile=q
            )

    elif target and categorical_target_class:
        agg_funcs['proportion'] = lambda df: (df[target] == categorical_target_class).mean()
    agg_funcs['pct_rows'] = lambda df: len(df)/len(sample_data)
    #create numerical feature bins
    if categorical_features is None:
        categorical_features = list()

    if len(categorical_features) < len(features):
        numerical_features = [f for f in features if not f in categorical_features]
        for feat in numerical_features:
            sample_data[feat] = categorify_feature(
                feature_series=sample_data[feat], 
                bins=bins,
                quantile_bins=quantile_bins,
                precision=precision
            )
    #trim categories if necessary
    if categorical_features:
        for feat in categorical_features:
            sample_data[feat] = trim_categories(
            sample_data[feat], 
            max_n_categories=max_n_categories,
            categories_recall_pct=categories_recall_pct,
            keep_nan=keep_nan,
        )
    
    #add nrows from orignal dataset
    agg_effect = (
        sample_data
        .groupby(features)
        .apply(custom_agg, agg_funcs=agg_funcs)
        .merge(
            (
                data
                .astype({feat: sample_data[feat].dtype for feat in features})
                .groupby(features)
                .size().to_frame('n_rows')
            ),
            left_index=True,
            right_index=True,
        )
    )

    #add cumulative count of rows
    agg_effect = pd.merge(
        agg_effect, 
        get_cum_pct(agg_effect['pct_rows']),
        left_index=True,
        right_index=True,
    )
    #add confidence interval for categorical target (bernoulli distribution)
    confidence_factor = normal_dist.ppf(1-confidence_alpha/2)

    if target and categorical_target_class:
        agg_effect = (
            agg_effect
            .assign(
                std=lambda df: df.eval('sqrt(proportion * (1 - proportion))'),
                lower=lambda df: np.maximum(0, df.eval(f'proportion - {confidence_factor} * std/sqrt(n_rows)')),
                upper=lambda df: np.minimum(1, df.eval(f'proportion + {confidence_factor} * std/sqrt(n_rows)')),
            )
        )
    #add confidence numerical for categorical target (bernoulli distribution)
    else:
        agg_effect['lower'] = agg_effect.eval(f'mean - {confidence_factor} * std/sqrt(n_rows)')
        agg_effect['upper'] = agg_effect.eval(f'mean + {confidence_factor} * std/sqrt(n_rows)')
    

    return agg_effect