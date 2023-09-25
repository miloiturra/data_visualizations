import pandas as pd
import numpy as np
from typing import Literal, Union, List, Dict, Optional

def sort_categories(value_counts_df: pd.DataFrame, by: Literal['mean_weight']='mean_weight') -> pd.Index:
    if by=="mean_weight":
        categs = (
            value_counts_df
            .groupby('value')
            .pct.mean()
            .sort_values(ascending=False)
            .index
        )
    elif by=="weight_difference":
        pass
    return categs 

def categorify_feature(
        feature_series: pd.Series, 
        bins: Union[int, List[float]],
        quantile_bins: bool=False,
        precision: int=1
    ) -> pd.Series:
    if quantile_bins:
        bins_values = pd.qcut(
            feature_series, 
            q=bins, precision=precision
        )
    else:
        bins_values = pd.cut(
            feature_series, bins=bins, precision=precision
        )
    return bins_values

def get_category_mapping(
    categories_series: pd.Series, 
    max_n_categories: Optional[int]=None, 
    categories_recall_pct: Optional[float]=None, 
    keep_nan: Optional[bool]=True,
    other_cat_name: str='Others'
) -> Dict[Union[str, float], str]:
    categories_pct = (
        categories_series
        .value_counts(normalize=True, dropna=False)
        .sort_values(ascending=False)
        .cumsum()
        .to_frame('cum_pct')
        .reset_index()
    )
    
    if categories_recall_pct is None:
        n_recalled_categories = len(categories_pct)
    else:
        n_recalled_categories = min(categories_pct.query("cum_pct >= @categories_recall_pct - 1e-6").index)+1
    
    recall_categories = categories_pct[categories_series.name][:n_recalled_categories]

    if max_n_categories is None:
        max_n_categories = len(recall_categories)
    selected_categories = recall_categories[:min(max_n_categories, len(recall_categories))].values
    category_mapping = {
        c: other_cat_name 
        for c in categories_series.unique() if not c in selected_categories
    }

    #TODO fix this
    if keep_nan and np.NaN in categories_pct.index:
        try:
            category_mapping[None] = 'NaN'
        except:
            pass
        try:
            category_mapping[np.NaN] = 'NaN'
        except:
            pass

    return category_mapping

def trim_categories(
    categories_series: pd.Series, 
    max_n_categories: Optional[int]=None,
    categories_recall_pct: Optional[float]=None,
    keep_nan: Optional[bool]=True,
) -> pd.Series:
    category_mapping = get_category_mapping(
        categories_series, max_n_categories, categories_recall_pct, keep_nan
    )
    return categories_series.replace(category_mapping)