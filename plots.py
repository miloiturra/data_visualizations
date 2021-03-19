import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd 
from scipy.stats import wasserstein_distance as wasser_dist
from itertools import combinations, product
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF   


red = "#E07182"
blue = "#4487D3"
green = "#96D6B4"
grey = "#87878A"

color_cycle = [red, blue, green, grey]

def get_xbins_size(data_dict, nbins):
    ranges = [data.max() - data.min() for name, data in data_dict.items()]
    max_range = max(ranges)
    return max_range/nbins

def lower_upper_scale(x, lower, upper):
    return (x - lower)/(upper - lower)

def get_cdf(data_dict, steps):
    all_values = np.concatenate(list(data_dict.values()))
    start = all_values.min()
    stop = all_values.max()
    x = np.linspace(start, stop, steps)
    cdf_dict = {}
    for name, values in data_dict.items():
        cdf = ECDF(values)
        cdf_dict[name] = (x, cdf(x))
    return cdf_dict


def get_lower_upper(data_dict, q_tuple=None):
    x = np.concatenate(list(data_dict.values()))
    if q_tuple is not None:
        lower, upper = np.quantile(x, q_tuple[0]), np.quantile(x, q_tuple[1])
    else:
        lower, upper = x.min(), x.max()
    return lower, upper 

def calc_wasserstein(data_dict, normalize=True, q_tuple=None):
    if normalize is True:
        lower, upper = get_lower_upper(data_dict, q_tuple)
    names_list = list(data_dict.keys())
    names_combs = combinations(names_list, 2)
    distance_df = pd.DataFrame(names_combs, columns=['data1', 'data2'])
    distance_df['distance'] = 0
    distance_df.set_index(['data1', 'data2'], inplace=True)
    for data1, data2 in distance_df.index:
        x1, x2 = data_dict[data1], data_dict[data2]
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
    plot_cdf=False,
    nbins=100,
    steps_cdf=50,
    opacity=0.5):

    cols_plot = 2 if plot_cdf else 1
    fig = make_subplots(rows=1, cols=cols_plot)
    if plot_cdf:
        cdf_dict =  get_cdf(data_dict, steps=steps_cdf)

    iter_count = 0
    for name, data in data_dict.items():
        color = color_cycle[iter_count%len(color_cycle)]

        hist = go.Histogram(
            x=data, histnorm='percent', 
            name=name, 
            opacity=opacity, 
            legendgroup=name,
            marker_color=color
        )
        fig.add_trace(hist, row=1, col=1)
        if plot_cdf:
            x, y = cdf_dict[name]
            lines = go.Scatter(
                x=x, y=y, 
                name=name, 
                opacity=opacity, 
                legendgroup=name, 
                showlegend=False,
                marker_color=color,
                mode='lines'
                )
            fig.add_trace(lines, row=1, col=2)
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
    xbins_size = get_xbins_size(data_dict, nbins)
    fig.update_traces(xbins_size=xbins_size, row=1, col=1)

    fig.show()
    return fig

def value_counts_array(x_array):
    x_series = pd.Series(x_array)
    counts = x_series.value_counts(dropna=False).to_frame('count')
    pct = x_series.value_counts(normalize=True, dropna=False).to_frame('pct')

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

def trim_categories(value_counts_df, min_weight_thresh):
    pass

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
        
    return distance_df


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
    fig.show()

def compare_numerical_features(data, features, nbins, groupby=None, plot_cdf=True):
    groupby_data = data.groupby(groupby)
    dist_list = []
    for feature in features:
        data_dict = {name: data.values for name, data in groupby_data[feature]}
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

    for feature, dist in dists.iteritems():
        data_dict = {name: data.values for name, data in groupby_data[feature]}
        fig = plot_histograms(
            data_dict, 
            title=feature+f", distance = {round(dist, 4)} ", 
            xaxis_title=feature, 
            yaxis_title="% of data", 
            height=None, 
            width=None,
            nbins=nbins,
            opacity=0.5,
            plot_cdf=True
        )
    return dists 

def compare_categorical_features(data, features, groupby=None):
    groupby_data = data.groupby(groupby)
    dist_list = []
    for feature in features:
        data_dict = {name: data.values for name, data in groupby_data[feature]}
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

    for feature, dist in dists.iteritems():
        data_dict = {name: data.values for name, data in groupby_data[feature]}
        fig = plot_discrete_histogram(
            data_dict, 
            title=feature+f", distance = {round(dist, 4)} ", 
            xaxis_title=feature, 
            yaxis_title="% of data", 
            height=None, 
            width=None,
        )
    return dists 