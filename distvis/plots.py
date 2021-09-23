import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd 
from scipy.stats import wasserstein_distance as wasser_dist
from scipy.stats import norm
from itertools import combinations, product
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF   
from pandas.api.types import CategoricalDtype
from copy import copy

red = "#E07182"
blue = "#4487D3"
green = "#96D6B4"
purple = "#B140C8"
grey = "#87878A"

color_cycle = [red, blue, green, purple, grey]

def get_xbins_size(data_dict, nbins):
    ranges = [data.max() - data.min() for name, data in data_dict.items()]
    max_range = max(ranges)
    return max_range/nbins

def lower_upper_scale(x, lower, upper):
    return (x - lower)/(upper - lower)

def dropnan(x):
    return x[~np.isnan(x)]

def get_cdf(data_dict, steps):
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

def get_inv_cdf(data_dict, quantiles):
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

def get_lower_upper(data_dict, q_tuple=None):
    x = np.concatenate(list(data_dict.values()))
    x = dropnan(x)
    if q_tuple is not None:
        lower, upper = np.quantile(x, q_tuple[0]), np.quantile(x, q_tuple[1])
    else:
        lower, upper = x.min(), x.max()
    return lower, upper 

def calc_wasserstein(data_dict, normalize=True, q_tuple=None, lower_upper_tuple=None):
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

def plot_histograms(
    data_dict, 
    title=None, 
    xaxis_title=None, 
    yaxis_title=None, 
    height=None, 
    width=None,
    plot_hist=True,
    plot_cdf=False,
    plot_inv_cdf=False,
    nbins=100,
    steps_cdf=100,
    quantiles=np.linspace(0, 1, 101),
    opacity=0.5):

    cols_plot = int(plot_hist) + int(plot_cdf) + int(plot_inv_cdf)
    fig = make_subplots(rows=1, cols=cols_plot)
    if plot_cdf:
        cdf_dict =  get_cdf(data_dict, steps=steps_cdf)
    if plot_inv_cdf:
        inv_cdf_dict = get_inv_cdf(data_dict, quantiles=quantiles)
    iter_count = 0
    for name, data in data_dict.items():
        subplot_count = 1
        color = color_cycle[iter_count%len(color_cycle)]
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
        iter_count += 1

    fig.update_layout(
        yaxis=dict(title=yaxis_title, showgrid=False),
        xaxis=dict(title=xaxis_title, showgrid=False),
        height=height, width=width,
        title=title,
        hovermode="x",
        paper_bgcolor='white',
        plot_bgcolor='white',
        hoverlabel_align='right',
        barmode='overlay',  
    )
    if plot_hist:
        xbins_size = get_xbins_size(data_dict, nbins)
        fig.update_traces(xbins_size=xbins_size, row=1, col=1)

    return fig

def value_counts_array(x_array):
    x_series = pd.Series(x_array)
    counts = x_series.value_counts(dropna=False).to_frame('count')
    pct = 100 * x_series.value_counts(normalize=True, dropna=False).to_frame('pct')

    return pd.merge(counts, pct, left_index=True, right_index=True)

def fill_missing_comb(value_counts_df):
    names = value_counts_df["name"].unique()
    values = value_counts_df["value"].unique()
    all_comb = product(names, values)
    all_comb_df = pd.DataFrame(all_comb, columns=['name', 'value'])
    value_counts_df = pd.merge(value_counts_df, all_comb_df, how='right', on=['name', 'value'])
    value_counts_df.fillna(0, inplace=True)
    value_counts_df['count'] = value_counts_df['count'].astype(int)
    return value_counts_df

def sort_categories(value_counts_df, by='mean_weight'):
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

def value_counts_dict(data_dict):
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

def calc_total_variation(data_dict, normalize=False):
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


def plot_discrete_histogram(
    data_dict, 
    sort_categories_by='mean_weight',
    title=None, 
    xaxis_title=None, 
    yaxis_title=None, 
    height=None, 
    width=None):

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
        color = color_cycle[iter_count%len(color_cycle)]
        plot_data = go.Bar(name=name, x=categs, y=dist_weights, marker_color=color)
        plot_data_list.append(plot_data)
        iter_count += 1

    fig = go.Figure(data=plot_data_list)

    fig.update_layout(
        barmode='group',
        yaxis=dict(title=yaxis_title, showgrid=False),
        xaxis=dict(title=xaxis_title, showgrid=False),
        height=height, width=width,
        title=title,
        hovermode="x",
        paper_bgcolor='white',
        plot_bgcolor='white',
        hoverlabel_align='right',
    )
    return fig

def compare_numerical_features(
    data, 
    features, 
    nbins, 
    groupby=None, 
    queries_dict=None,
    sample_pct=1,
    **kwargs
):

    if groupby is not None:
        groupby_data = {name: df.sample(frac=sample_pct) for name, df in data.groupby(groupby)}
    elif queries_dict is not None:
        groupby_data = {name: data.query(query).sample(frac=sample_pct) for name, query in queries_dict.items()}

    dist_list = []
    for feature in features:
        data_dict = {name: data[feature].values for name, data in groupby_data.items()}
        dist  = calc_wasserstein(data_dict, normalize=True, q_tuple=(0.01, .99))
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
    for feature, dist in dists.iteritems():
        data_dict = {name: data[feature].values for name, data in groupby_data.items()}
        fig_dict[feature] = plot_histograms(
            data_dict, 
            title=feature+f", distance = {round(dist, 4)} ", 
            xaxis_title=feature, 
            yaxis_title="% of data", 
            height=None, 
            width=None,
            nbins=nbins,
            opacity=0.5,
            **kwargs
        )
        
    return fig_dict, dists.to_frame()

def compare_categorical_features(
    data, 
    features, 
    groupby=None, 
    queries_dict=None, 
    sample_pct=1,
    max_n_categories=100,
    categories_recall_pct=1,
    keep_nan=True,
):
    if groupby is not None:
        groupby_data = {name: df.sample(frac=sample_pct) for name, df in data.groupby(groupby)}
    elif queries_dict is not None:
        groupby_data = {name: data.query(query).sample(frac=sample_pct) for name, query in queries_dict.items()}

    dist_list = []
    category_trim_mapping = {}
    for feature in features:

        cat_map = get_category_mapping(
            pd.concat([data[[feature]] for data in groupby_data.values()])[feature], 
            max_n_categories, categories_recall_pct, keep_nan
        )
        category_trim_mapping[feature] = cat_map
        data_dict = {name: data[feature].values for name, data in groupby_data.items()}
        dist  = calc_total_variation(data_dict, normalize=False)
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
    for feature, dist in dists.iteritems():
        data_dict = {
            name: data[feature].replace(category_trim_mapping[feature])
            for name, data in groupby_data.items()
        }
        fig_dict[feature] = plot_discrete_histogram(
            data_dict, 
            title=feature+f", distance = {round(dist, 4)} ", 
            xaxis_title=feature, 
            yaxis_title="% of data", 
            height=None, 
            width=None,
        )
    return fig_dict, dists.to_frame()

def marginal_dependency_plot(
    data, 
    target, 
    feature_col, 
    categorical_feature=False, 
    categorical_target_class=None,
    categorical_target_alpha=0.1,
    bins=20, 
    sample_pct=1, 
    lower_q=.1,
    upper_q=.9,
    max_n_categories=100,
    categories_recall_pct=1,
    keep_nan=True,
    xaxis_title=None,
    yaxis_title=None,
    show_global_metric=True,
    **kwargs
):
    nan_target_constrain = ~data[target].isna()
    
    sample_data = (
        data[[target, feature_col]].loc[nan_target_constrain]
        .sample(frac=sample_pct)
        .copy()
    )
    if categorical_feature is False:
        sample_data[feature_col] = categorify_feature(sample_data[feature_col], bins)
    else:
        sample_data[feature_col] = trim_categories(
            sample_data[feature_col], 
            max_n_categories,
            categories_recall_pct,
            keep_nan)

    if categorical_target_class is None:
        group_statistics = get_group_statistics(sample_data, feature_col, target, lower_q, upper_q)
        if categorical_feature is True:
            group_statistics = group_statistics.sort_values(by='quantile_0.5')
        lower_col = f'quantile_{lower_q}'
        upper_col = f'quantile_{upper_q}'
        middle_col = 'quantile_0.5'
        secondary_middle_col = 'mean'
        lower_name = f"Quantile {100*lower_q}%"
        upper_name = f"Quantile {100*upper_q}%"
        if show_global_metric:
            mean_target, median_target = data[target].mean(), data[target].median()

    else:
        group_statistics = get_target_proportion(
            sample_data, feature_col, 
            target, categorical_target_alpha, categorical_target_class
        )
        if categorical_feature is True:
            group_statistics = group_statistics.sort_values(by='proportion')
        lower_col = 'lower'
        upper_col = 'upper'
        middle_col = 'proportion'
        yaxis_title = f'Proportion of {target} = {categorical_target_class}'
        lower_name = f"Lower confidence alpha = {categorical_target_alpha}"
        upper_name = f"Upper confidence alpha = {categorical_target_alpha}"
        secondary_middle_col = None
        if show_global_metric:
            global_prop = (data[target] == categorical_target_class).mean()

    
    if categorical_feature is False:
        group_statistics.index = group_statistics.index.astype(str)
        
    if xaxis_title is None:
        xaxis_title = feature_col
    if yaxis_title is None:
        yaxis_title = target
    if categorical_feature:
        fig = plot_confidence_bars(
            group_statistics,
            lower_col=lower_col, 
            upper_col=upper_col, 
            confidence_bar_col=middle_col,
            secondary_bar_col=secondary_middle_col,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            **kwargs
        )
        if show_global_metric:
            if categorical_target_class is not None:
                fig.add_hline(y=100*global_prop, line_width=2, line_dash="dash", line_color="black", opacity=0.4)
            else:
                fig.add_hline(y=mean_target, line_width=2, line_color=blue, opacity=0.4)
                fig.add_hline(y=median_target, line_width=2, line_color=red, opacity=0.4)

    else:
        central_name = 'Proportion' if categorical_target_class is not None else 'Median'
        fig = plot_confidence_lines(
            group_statistics,
            lower_col=lower_col, 
            upper_col=upper_col, 
            mean_col=middle_col,
            dashed_col=secondary_middle_col,
            lower_name=lower_name,
            upper_name=upper_name,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            central_name=central_name,
            **kwargs
        )
        if show_global_metric:
            if categorical_target_class is not None:
                fig.add_hline(y=100*global_prop, line_width=2, line_color=red, opacity=0.4)
            else:
                fig.add_hline(y=mean_target, line_width=2, line_dash="dash", line_color="black", opacity=0.4)
                fig.add_hline(y=median_target, line_width=2, line_color=red, opacity=0.4)

    return fig

def categorify_feature(feature_series, bins):
    feature_values = feature_series.to_frame('feature')
    feature_values['feature_bin'] = pd.cut(feature_values['feature'], bins=bins)
    return feature_values['feature_bin'].values

def get_category_mapping(categories_series, max_n_categories, categories_recall_pct, keep_nan):
    categories_pct = (
        categories_series
        .value_counts(normalize=True, dropna=False)
        .sort_values(ascending=False)
        .cumsum()
    )
  
    eps = 10e-6
    n_recall_cats = np.where(categories_pct >= categories_recall_pct-eps)[0][0] + 1

    recall_categories = categories_pct.index[:n_recall_cats]
    selected_categories = recall_categories[:min(max_n_categories, len(recall_categories))]
    category_mapping = {c: 'Otras' for c in categories_series.unique() if not c in selected_categories}

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

def trim_categories(categories_series, max_n_categories, categories_recall_pct, keep_nan):
    category_mapping = get_category_mapping(categories_series, max_n_categories, categories_recall_pct, keep_nan)
    return categories_series.replace(category_mapping)

def get_group_statistics(data, feature_col, target, lower_q, upper_q):
    quantiles = [lower_q, 0.5, upper_q]
    group_quantiles = (
        data.groupby(feature_col)
        [target]
        .apply(np.quantile, q=quantiles)
        .explode()
        .reset_index()
    )
    group_quantiles[target] = group_quantiles[target].astype(float)
    quantile_names = [f'quantile_{q}' for q in quantiles]
    group_quantiles['statistic'] = np.concatenate([quantile_names for _ in range(group_quantiles[feature_col].nunique())])
    group_quantiles = pd.pivot_table(group_quantiles, index=feature_col, columns='statistic', values=target)
    group_quantiles['mean'] = data.groupby(feature_col)[target].mean()

    return group_quantiles

def get_target_proportion(data, feature_col, target, alpha, target_class):
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
def plot_confidence_bars(
    data, 
    lower_col, 
    upper_col, 
    confidence_bar_col, 
    confidence_bar_name='Median',
    secondary_bar_name='Mean',
    index_col=None,
    secondary_bar_col=None,
    xaxis_title='Feature Bins', 
    yaxis_title='Target',
    title=None
): 
    if index_col is None:
        index_col = data.index.name
        data.reset_index(inplace=True)

    fig = go.Figure()
    lower_error = data[confidence_bar_col] - data[lower_col]
    upper_error = data[upper_col] - data[confidence_bar_col]

    fig.add_trace(
    go.Bar(
        name=confidence_bar_name,
        x=data[index_col], y=data[confidence_bar_col],
        error_y=dict(symmetric=False, type='data', array=upper_error, arrayminus=lower_error),
        marker_color=red,
    ))

    if secondary_bar_col:
        fig.add_trace(
        go.Bar(
            name=secondary_bar_name,
            x=data[index_col], y=data[secondary_bar_col],
            marker_color=blue,
        ))
    fig.update_layout(
        yaxis=dict(title=yaxis_title, showgrid=False),
        xaxis=dict(title=xaxis_title, showgrid=False),
        title=title,
        hovermode="x",
        paper_bgcolor='white',
        plot_bgcolor='white',
        hoverlabel_align = 'right',
    )
    return fig

def plot_confidence_lines(
    data, 
    lower_col, 
    upper_col, 
    mean_col, 
    index_col=None, 
    dashed_col='mean', 
    show_legend=True,
    xaxis_title='Feature Bins', 
    yaxis_title='Target', 
    title=None, 
    central_name='Median',
    dashed_name='Mean',
    lower_name='Lower effect',
    upper_name="Upper effect",
):

    if index_col is None:
        index_col = data.index.name
        data.reset_index(inplace=True)

    color_lower_upper_marker = red
    color_fillbetween = 'rgba(88, 44, 51, 0.3)'
    color_lower_upper_marker = color_fillbetween
    color_median = red
    fig_list = [
    go.Scatter(
        name=central_name,
        x=data[index_col],
        y=data[mean_col],
        mode='lines',
        line=dict(color=color_median),
        showlegend=show_legend,

    ),
        
    go.Scatter(
        name=upper_name,
        x=data[index_col],
        y=data[upper_col],
        mode='lines',
        marker=dict(color=color_lower_upper_marker),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name=lower_name,
        x=data[index_col],
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
            x=data[index_col],
            y=data[dashed_col],
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=show_legend,    
        )
        fig_list = [dashed_fig] + fig_list
        
    fig = go.Figure(fig_list)

    fig.update_layout(
        yaxis=dict(title=yaxis_title, showgrid=False),
        xaxis=dict(title=xaxis_title, showgrid=False),
        title=title,
        hovermode="x",
        paper_bgcolor='white',
        plot_bgcolor='white',
        hoverlabel_align = 'right',
    )
    
    return fig