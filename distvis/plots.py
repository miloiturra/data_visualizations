import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd 
import numpy as np

from data_utils import get_xbins_size
from distribution_utils import (
    get_cdf, get_inv_cdf, 
    calc_wasserstein, 
    value_counts_dict, 
    calc_total_variation, 
    get_group_statistics, 
    get_target_proportion
)
from categorical_utils import sort_categories, categorify_feature, trim_categories, get_category_mapping
from aggregation_utils import get_agg_effects

from typing import Dict, Tuple, Optional, List, Union, Literal, Any

red = "#E07182"
blue = "#4487D3"
green = "#96D6B4"
purple = "#B140C8"
grey = "#87878A"

def from_hex_to_rgba(hex_color: str, opacity: float) -> str:
    rgb = ', '.join([str(int(hex_color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4)])
    return f"rgba({rgb}, {opacity})"

COLOR_CYCLE = [red, blue, green, purple, grey]

def plot_histograms(
    data_dict: Dict[str, np.ndarray],
    title: Optional[str]=None, 
    xaxis_title: Optional[str]=None, 
    yaxis_title: Optional[str]=None, 
    height: Optional[float]=None, 
    width: Optional[float]=None,
    plot_hist: bool=True,
    plot_cdf:bool=False,
    plot_inv_cdf:bool=False,
    plot_box:bool = False,
    nbins: int=100,
    steps_cdf: int=100,
    quantiles: List[float]=np.linspace(0, 1, 101),
    opacity: float=0.5,
    color_cycle: Optional[List[str]]=COLOR_CYCLE,
    showgrid: bool=True,
) -> go.Figure:

    plot_cols = int(plot_hist) + int(plot_cdf) + int(plot_inv_cdf) + int(plot_box)
    fig = make_subplots(rows=1, cols=plot_cols)
    if plot_cdf:
        cdf_dict =  get_cdf(data_dict, steps=steps_cdf)
    if plot_inv_cdf:
        inv_cdf_dict = get_inv_cdf(data_dict, quantiles=quantiles)
    iter_count = 0
    for name, data in data_dict.items():
        subplot_count = 1
        color = color_cycle[iter_count%len(color_cycle)] if color_cycle else None
        if plot_hist:
            hist = go.Histogram(
                x=data, histnorm='percent', 
                name=name, 
                opacity=opacity, 
                legendgroup=name,
                marker_color=color,
                hovertemplate="%{y:,.2f}",
            )

            fig.add_trace(hist, row=1, col=subplot_count)
            subplot_count += 1
        if plot_cdf:
            x, y = cdf_dict[name]
            lines = go.Scatter(
                x=x, y=y, 
                name=name, 
                opacity=opacity, 
                legendgroup=name, 
                showlegend=False,
                marker_color=color,
                mode='lines',
                hovertemplate="%{y:,.2f}",
            )
            fig.add_trace(lines, row=1, col=subplot_count)
            subplot_count += 1
        if plot_inv_cdf:
            x, y = inv_cdf_dict[name]
            lines = go.Scatter(
                x=x, y=y, 
                name=name, 
                opacity=opacity, 
                legendgroup=name, 
                showlegend=False,
                marker_color=color,
                mode='lines',
                hovertemplate="%{y:,.2f}",
            )
            fig.add_trace(lines, row=1, col=subplot_count)
            subplot_count += 1
        if plot_box:
            boxplot = go.Box(
                y=data, 
                name=name, 
                opacity=opacity, 
                legendgroup=name, 
                showlegend=False,
                marker_color=color,
                hovertemplate="%{y:,.2f}",
            )
            fig.add_trace(boxplot, row=1, col=subplot_count)
            subplot_count += 1
        iter_count += 1

    fig.update_layout(
        yaxis=dict(title=yaxis_title, showgrid=showgrid),
        xaxis=dict(title=xaxis_title, showgrid=showgrid),
        height=height, width=width,
        title=title,
        barmode='overlay',  
        plot_bgcolor='white' if not showgrid else None,
    )
    if plot_hist:
        xbins_size = get_xbins_size(data_dict, nbins)
        fig.update_traces(xbins_size=xbins_size, row=1, col=1)

    return fig

def plot_discrete_histogram(
    data_dict: Dict[str, np.ndarray],
    sort_categories_by: Literal['mean_weight']='mean_weight',
    title: Optional[str]=None, 
    xaxis_title: Optional[str]=None, 
    yaxis_title: Optional[str]=None, 
    height: Optional[float]=None, 
    width: Optional[float]=None,
    color_cycle: Optional[List[str]]=COLOR_CYCLE,
    showgrid: bool=True,
) -> go.Figure:

    value_counts_df = value_counts_dict(data_dict)
    names = value_counts_df["name"].unique()
    categs = sort_categories(value_counts_df, by=sort_categories_by)

    iter_count = 0
    plot_data_list = []
    for name in names:
        name_loc = (value_counts_df.name == name) 
        dist_weights = (
            value_counts_df
            .loc[name_loc]
            .set_index('value')
            .loc[categs, 'pct']
        )
        color = color_cycle[iter_count%len(color_cycle)] if color_cycle else None
        plot_data = go.Bar(name=name, x=categs, y=dist_weights, marker_color=color)
        plot_data_list.append(plot_data)
        iter_count += 1

    fig = go.Figure(data=plot_data_list)

    fig.update_layout(
        barmode='group',
        yaxis=dict(title=yaxis_title, showgrid=showgrid),
        xaxis=dict(title=xaxis_title, showgrid=showgrid),
        height=height, width=width,
        title=title,
        plot_bgcolor='white' if not showgrid else None,
    )
    return fig

def compare_numerical_features(
    data_dict: Dict[str, pd.DataFrame],
    features: List[str], 
    nbins: int, 
    sample_pct:float=1,
    **plot_histograms_kwargs
) -> Tuple[Dict[str, go.Figure], pd.DataFrame]:
    
    sample_data_dict = {name: data.sample(frac=sample_pct) for name, data in data_dict.items()}
    dist_list = list()
    for feature in features:
        feature_data_dict = {name: data[feature].values for name, data in sample_data_dict.items()}
        dist  = calc_wasserstein(feature_data_dict, normalize=True, q_tuple=(0.01, .99))
        dist['feature'] = feature
        dist_list.append(dist)

    dists = (
        pd.concat(dist_list)
        .reset_index()
        .groupby('feature')
        .distance
        .mean()
        .sort_values(ascending=False)
    )

    fig_dict = {}
    for feature, dist in dists.items():
        feature_data_dict = {name: data[feature].values for name, data in sample_data_dict.items()}
        fig_dict[feature] = plot_histograms(
            feature_data_dict,
            title=feature+f", distance = {round(dist, 4)} ", 
            xaxis_title=feature, 
            yaxis_title="% of data", 
            nbins=nbins,
            **plot_histograms_kwargs
        )
        
    return fig_dict, dists.to_frame()

def compare_categorical_features(
    data_dict: Dict[str, pd.DataFrame],
    features: List[str], 
    sample_pct: float=1,
    max_n_categories: int=100,
    categories_recall_pct: float=1,
    keep_nan: bool=True,
    other_cat_name: str='Others',
    **plot_discrete_histogram_kwargs
) -> Tuple[Dict[str, go.Figure], pd.DataFrame]:

    sample_data_dict = {name: data.sample(frac=sample_pct) for name, data in data_dict.items()}

    dist_list = list()
    category_trim_mapping = dict()
    for feature in features:
        cat_map = get_category_mapping(
            pd.concat([data[[feature]] for data in sample_data_dict.values()])[feature], 
            max_n_categories, categories_recall_pct, keep_nan, other_cat_name,
        )
        category_trim_mapping[feature] = cat_map
        feature_data_dict = {name: data[feature].values for name, data in sample_data_dict.items()}
        dist  = calc_total_variation(feature_data_dict)
        dist['feature'] = feature
        dist_list.append(dist)

    dists = (
        pd.concat(dist_list)
        .reset_index()
        .groupby('feature')
        .distance
        .mean()
        .sort_values(ascending=False)
    )
    fig_dict = {}
    for feature, dist in dists.items():
        feature_data_dict = {
            name: data[feature].replace(category_trim_mapping[feature]).values
            for name, data in sample_data_dict.items()
        }
        fig_dict[feature] = plot_discrete_histogram(
            feature_data_dict, 
            title=feature+f", distance = {round(dist, 4)} ", 
            xaxis_title=feature, 
            yaxis_title="% of data", 
            **plot_discrete_histogram_kwargs
        )
    return fig_dict, dists.to_frame()

def marginal_dependency_plot(
    data: pd.DataFrame, 
    feature: str,
    categorical_feature: bool,
    target: str,
    categorical_target_class: Optional[Any] = None,
    confidence_alpha: float = 0.2,
    bins: Union[int, List[float]]=10,
    quantiles: Tuple[float, float] = (0.1, 0.9),
    sample_frac: float = 1,
    quantile_bins: bool = False,
    show_histogram: bool = True,
    precision: int = 1,
    category_sort_by: Literal['mean', 'median', 'n_rows', 'proportion'] = 'mean',
    category_sort_ascending: bool = False,
    confidence_interval_type: Literal['quantile', 'gaussian'] = 'quantile',
    title: Optional[str] = None,
):

    agg_effect = get_agg_effects(
        data=data,
        features=[feature],
        target=target,
        categorical_features=[feature] if categorical_feature else None,
        quantiles=list(quantiles),
        sample_frac=sample_frac,
        quantile_bins=quantile_bins,
        bins=bins,
        precision=precision,
        categorical_target_class=categorical_target_class,
        confidence_alpha=confidence_alpha,
    )

    if categorical_feature:
        category_sort_by = 'proportion' if categorical_target_class else category_sort_by
        agg_effect = agg_effect.sort_values(
            by=category_sort_by, ascending=category_sort_ascending
        )
    
    confidence_interval_type = 'gaussian' if categorical_target_class else confidence_interval_type
    lower_col = f'q{int(100*quantiles[0])}' if confidence_interval_type == 'quantile' else 'lower'
    upper_col = f'q{int(100*quantiles[1])}' if confidence_interval_type == 'quantile' else 'upper'
    lower_name = f'Quantile {int(100*quantiles[0])}%'if confidence_interval_type == 'quantile' else 'Lower effect'
    upper_name = f'Quantile {int(100*quantiles[1])}%'if confidence_interval_type == 'quantile' else 'Upper effect'

    effect_fig = plot_confidence_lines(
        agg_effect,
        central_col='proportion' if categorical_target_class else 'median', 
        lower_col=lower_col,
        upper_col=upper_col,
        dashed_col=None if categorical_target_class else 'mean', 
        central_name='Median',
        lower_name=lower_name,
        upper_name=upper_name,
        dashed_name=None if categorical_target_class else 'Mean',
    )
    
    if show_histogram:
        bar_plot = go.Bar(
            x=agg_effect.index.astype(str), 
            y=agg_effect.pct_rows, 
            text=agg_effect.n_rows,
            marker_color=red,
            name='% of data'
        )
        bar_fig = go.Figure(bar_plot)
        bar_fig.update_layout(
            yaxis=dict(autorange='reversed'),
        )

        fig = make_subplots(
            rows=2, cols=1, row_heights=[0.8, 0.2], 
            shared_xaxes=True,
            vertical_spacing=0.2,
        )
        for trace in effect_fig.data: fig.add_trace(trace, row=1, col=1)
        for trace in bar_fig.data: fig.add_trace(trace, row=2, col=1)
        fig.update_layout(
            xaxis_showticklabels=True, xaxis2_showticklabels=False,
            yaxis2=dict(autorange='reversed', tickformat='.0%'),
            yaxis2_title='% of data',
        )
    else: 
        fig = effect_fig
    fig.update_layout(
        xaxis1_title=feature,
        yaxis_title=target, 
        title=title,
    )
    return fig

def plot_confidence_lines(
    data: pd.DataFrame, 
    central_col: str, 
    lower_col: str, 
    upper_col: str, 
    index_col: Optional[str] = None, 
    dashed_col: Optional[str] = 'mean', 
    show_legend: bool = True,
    xaxis_title: Optional[str] = None, 
    yaxis_title: Optional[str] = None, 
    title: Optional[str] = None, 
    central_name: str = 'Median',
    lower_name: str = 'Lower effect',
    upper_name: str = "Upper effect",
    dashed_name: str = 'Mean',
    showgrid: bool=True,
    opacity: float=0.3,
    color: str=red,
):

    if index_col is None:
        index_col = data.index.name
        x_values = data.index.astype(str)
    else:
        x_values = data[index_col].astype(str)


    color_lower_upper_marker = color
    color_fillbetween = from_hex_to_rgba(color, opacity=opacity)
    color_lower_upper_marker = color
    color_median = color
    
    fig_list = [
        go.Scatter(
            name=central_name,
            x=x_values,
            y=data[central_col],
            mode='lines+markers',
            line=dict(color=color_median),
            marker=dict(symbol="circle"),
            showlegend=show_legend,

        ),
            
        go.Scatter(
            name=upper_name,
            x=x_values,
            y=data[upper_col],
            mode='lines',
            marker=dict(color=color_lower_upper_marker),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name=lower_name,
            x=x_values,
            y=data[lower_col],
            marker=dict(color=color_lower_upper_marker),
            line=dict(width=0),
            mode='lines',
            fillcolor=color_fillbetween,
            fill='tonexty',
            showlegend=False
        )
    ]
    
    if dashed_col is not None:
        dashed_fig = go.Scatter(
            name=dashed_name,
            x=x_values,
            y=data[dashed_col],
            line=dict(color=color, dash='dash'),
            marker=dict(symbol="circle"),
            showlegend=show_legend,    
        )
        fig_list = [dashed_fig] + fig_list
        
    fig = go.Figure(fig_list)

    fig.update_layout(
        yaxis=dict(title=yaxis_title, showgrid=showgrid),
        xaxis=dict(title=xaxis_title, showgrid=False),
        title=title,
    )
    
    return fig