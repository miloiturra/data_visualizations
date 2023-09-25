from itertools import product
import numpy as np
import pandas as pd

from typing import Dict, Tuple, Optional

def get_xbins_size(data_dict: Dict[str, np.ndarray], nbins: int) -> float:
    ranges = [data.max() - data.min() for name, data in data_dict.items()]
    max_range = max(ranges)
    return max_range/nbins

def lower_upper_scale(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return (x - lower)/(upper - lower)

def dropnan(x: np.ndarray) -> np.ndarray:
    return x[~np.isnan(x)]

def get_lower_upper(
    data_dict: Dict[str, np.ndarray], 
    q_tuple: Optional[Tuple[float, float]]=None
) -> Tuple[float, float]:
    x = np.concatenate(list(data_dict.values()))
    x = dropnan(x)
    if q_tuple is not None:
        lower, upper = np.quantile(x, q_tuple[0]), np.quantile(x, q_tuple[1])
    else:
        lower, upper = x.min(), x.max()
    return lower, upper 

def fill_missing_comb(value_counts_df: pd.DataFrame) -> pd.DataFrame:
    names = value_counts_df["name"].unique()
    values = value_counts_df["value"].unique()
    all_comb = product(names, values)
    all_comb_df = pd.DataFrame(all_comb, columns=['name', 'value'])
    value_counts_df = pd.merge(value_counts_df, all_comb_df, how='right', on=['name', 'value'])
    value_counts_df.fillna(0, inplace=True)
    value_counts_df['count'] = value_counts_df['count'].astype(int)
    return value_counts_df