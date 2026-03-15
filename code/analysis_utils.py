"""
Analysis utility functions for duration reproduction experiment.

This module provides reusable functions for common analysis patterns,
following tidyverse-like principles for readable, chainable operations.
"""

import pandas as pd
import numpy as np
from scipy import stats
def regress_on(df, y, x, output='slope'):
    """
    Generic regression helper for groupby.apply patterns.

    Performs linear regression y ~ x and returns requested statistic.
    Useful for calculating CTI, SDI, or custom regression metrics.

    Args:
        df: DataFrame with data for regression
        y: Name of dependent variable column
        x: Name of independent variable column
        output: What to return:
            - 'slope': Just the slope coefficient
            - 'cti': Central Tendency Index (1 - slope)
            - 'sdi': Serial Dependence Index (alias for slope)
            - 'full': Series with all statistics

    Returns:
        Requested statistic(s). Returns None or empty Series if < 3 observations.

    Examples:
        >>> # Calculate CTI per participant
        >>> cti = df.groupby('nPar').apply(
        ...     lambda x: regress_on(x, y='bias', x='curDur', output='cti')
        ... )

        >>> # Calculate SDI with full stats
        >>> sdi_stats = df.groupby(['nPar', 'transition']).apply(
        ...     lambda x: regress_on(x, y='bias', x='preDur', output='full')
        ... )
    """
    if len(df) < 3:
        return pd.Series() if output == 'full' else None

    res = stats.linregress(df[x], df[y])

    if output == 'slope':
        return res.slope
    elif output == 'cti':
        return 1 - res.slope
    elif output == 'sdi':
        return res.slope  # Alias for clarity in sequential effects
    elif output == 'full':
        return pd.Series({
            'slope': res.slope,
            'intercept': res.intercept,
            'r_squared': res.rvalue ** 2,
            'p_value': res.pvalue,
            'stderr': res.stderr
        })
    else:
        raise ValueError(f"Unknown output type: {output}")


def calculate_participant_metric(
    df,
    group_cols,
    y,
    x,
    metric='slope',
    min_obs=3
):
    """
    Calculate regression metric per group using pandas chaining.

    Wrapper around regress_on that handles grouping, filtering,
    and result formatting in one clean pipeline.

    Args:
        df: Input DataFrame
        group_cols: Columns to group by (e.g., ['nPar', 'curTask'])
        y: Dependent variable
        x: Independent variable
        metric: Type of metric to calculate ('slope', 'cti', 'sdi', 'full')
        min_obs: Minimum observations required per group

    Returns:
        DataFrame with group columns and calculated metrics

    Example:
        >>> cti_results = calculate_participant_metric(
        ...     df_valid,
        ...     group_cols=['nPar', 'curTask'],
        ...     y='bias',
        ...     x='curDur',
        ...     metric='cti'
        ... )
    """
    # Filter groups with sufficient observations
    df_filtered = (
        df
        .groupby(group_cols, group_keys=False)
        .filter(lambda grp: len(grp) >= min_obs)
    )

    # Apply regression to each group
    # Use different variable name to avoid collision with x parameter
    result = (
        df_filtered
        .groupby(group_cols)
        .apply(lambda grp: regress_on(grp, y=y, x=x, output=metric), include_groups=False)
        .reset_index()
    )

    # If metric='full', the result will have the Series keys as columns
    # Otherwise, we need to rename the value column
    if metric != 'full':
        # The single value is in a column named 0, rename it to metric name
        if 0 in result.columns:
            result = result.rename(columns={0: metric})

    return result.dropna()


def add_response_and_bias(df):
    """
    Add response, bias, and relative error columns.

    Pipeline function for creating core dependent variables.

    Args:
        df: Raw data with 'rpr', 'flw', 'curTask', 'curDur' columns

    Returns:
        DataFrame with added columns: response, bias, relError
    """
    return df.assign(
        response=lambda x: x['rpr'].where(x['curTask'] == 'reproduction', x['flw']),
        bias=lambda x: x['response'] - x['curDur'],
        relError=lambda x: (x['response'] - x['curDur']) / x['curDur']
    )


def add_task_transitions(df):
    """
    Create task transition labels and switch indicator.

    Adds columns:
        - transition: RR, RF, FR, FF, or First
        - taskSwitch: repeat, switch, or First
        - responseType: reproduction or following (for RF/FR)

    Args:
        df: DataFrame with 'preTask' and 'curTask' columns

    Returns:
        DataFrame with added transition columns
    """
    return (
        df
        .assign(
            # Create transition labels
            transition=lambda x: (
                x['preTask'].fillna('') + x['curTask'].str[0].str.upper()
            ).replace({
                'reproductionR': 'RR',
                'reproductionF': 'RF',
                'followingR': 'FR',
                'followingF': 'FF',
                'R': 'First',
                'F': 'First'
            }),
            # Add task switch indicator
            taskSwitch=lambda x: x['transition'].map({
                'RR': 'repeat',
                'FF': 'repeat',
                'RF': 'switch',
                'FR': 'switch',
                'First': 'First'
            }),
            # Add response type for sequential trials
            responseType=lambda x: x['transition'].map({
                'RR': 'reproduction',
                'FR': 'reproduction',
                'RF': 'following',
                'FF': 'following'
            })
        )
    )


def add_nback_durations(df, n_back=3, block_cols=None):
    """
    Add n-back duration columns using vectorized groupby operations.

    Creates columns dur_n1, dur_n2, ..., dur_nN for each participant and block.
    Much faster than nested loops.

    Args:
        df: DataFrame with duration column
        n_back: How many previous trials to include
        block_cols: Columns defining independent sequences
                   (default: ['nPar', 'nB'])

    Returns:
        DataFrame with added n-back columns, filtered to trials with
        complete history (nT > n_back)
    """
    if block_cols is None:
        block_cols = ['nPar', 'nB']

    # Create shift columns in one pass per group
    result = (
        df
        .groupby(block_cols, group_keys=False)
        .apply(lambda x: x.assign(**{
            f'dur_n{n}': x['curDur'].shift(n)
            for n in range(1, n_back + 1)
        }))
        .reset_index(drop=True)
    )

    return result


def calculate_nback_correlations(df, max_n=3, group_col='curTask'):
    """
    Calculate correlations between bias and n-back durations.

    Returns tidy DataFrame with one row per task × n_back combination.

    Args:
        df: DataFrame with bias and dur_n1, dur_n2, ... columns
        max_n: Maximum n-back to calculate (must match available columns)
        group_col: Column to group by (default: curTask)

    Returns:
        DataFrame with columns: group_col, n_back, r, p, n_obs

    Example:
        >>> corr_df = calculate_nback_correlations(df_nback)
        >>> # Use with seaborn
        >>> sns.catplot(data=corr_df, x='n_back', y='r', hue='curTask', kind='bar')
    """
    from scipy.stats import pearsonr

    results = []

    for group in df[group_col].unique():
        group_data = df[df[group_col] == group]

        for n in range(1, max_n + 1):
            col = f'dur_n{n}'
            if col not in df.columns:
                continue

            valid = group_data[['bias', col]].dropna()

            if len(valid) > 0:
                r, p = pearsonr(valid['bias'], valid[col])
                results.append({
                    group_col: group,
                    'n_back': n,
                    'r': r,
                    'p': p,
                    'n_obs': len(valid)
                })

    return pd.DataFrame(results)


def prepare_tidy_for_facet(df, value_col, facet_cols, agg_cols, agg_func='mean'):
    """
    Prepare data in tidy format for seaborn FacetGrid.

    Aggregates data for plotting while preserving participant-level
    structure for error bar calculation.

    Args:
        df: Input DataFrame
        value_col: Column to aggregate (e.g., 'bias')
        facet_cols: Columns for faceting (e.g., ['curTask', 'taskSwitch'])
        agg_cols: Additional grouping columns (e.g., ['preDur', 'nPar'])
        agg_func: Aggregation function (default: 'mean')

    Returns:
        Tidy DataFrame ready for seaborn plotting

    Example:
        >>> plot_data = prepare_tidy_for_facet(
        ...     df_seq,
        ...     value_col='bias',
        ...     facet_cols=['curTask', 'taskSwitch'],
        ...     agg_cols=['preDur', 'nPar']
        ... )
        >>> g = sns.FacetGrid(plot_data, col='curTask', hue='taskSwitch')
        >>> g.map_dataframe(sns.pointplot, x='preDur', y='bias', errorbar='se')
    """
    group_cols = facet_cols + agg_cols

    return (
        df
        .groupby(group_cols, as_index=False)
        .agg({value_col: agg_func})
        .rename(columns={value_col: f'{value_col}_{agg_func}'})
    )
