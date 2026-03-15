"""Utilities for publication-quality figures."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams


# Publication style settings
PUBLICATION_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'patch.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3
}


# Color palettes
TASK_COLORS = {
    'reproduction': '#E74C3C',  # Red
    'following': '#3498DB',      # Blue
}

TRANSITION_COLORS = {
    'RR': '#E74C3C',  # Red (Repro → Repro)
    'FR': '#3498DB',  # Blue (Follow → Repro)
    'RF': '#9B59B6',  # Purple (Repro → Follow)
    'FF': '#2ECC71'   # Green (Follow → Follow)
}

COLORBLIND_PALETTE = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC',
                      '#CA9161', '#949494', '#ECE133']


def setup_publication_style():
    """Apply publication-quality matplotlib settings."""
    for key, value in PUBLICATION_STYLE.items():
        rcParams[key] = value
    sns.set_style('ticks')
    sns.set_context('paper')  # or 'notebook' for larger text


def save_figure(fig, filepath, formats=['png', 'pdf']):
    """Save figure in multiple formats.

    Args:
        fig: matplotlib figure
        filepath: path without extension
        formats: list of formats to save (e.g., ['png', 'pdf', 'svg'])
    """
    for fmt in formats:
        fig.savefig(f"{filepath}.{fmt}", dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}.{fmt}")


def add_significance_bar(ax, x1, x2, y, text, line_height=0.02,
                        text_offset=0.01, **kwargs):
    """Add significance bar between two points.

    Args:
        ax: matplotlib axis
        x1, x2: x-coordinates of bar endpoints
        y: y-coordinate of bar
        text: significance text (e.g., '*', '**', 'ns')
        line_height: height of vertical ticks
        text_offset: offset of text above bar
        **kwargs: additional arguments for plot (color, linewidth, etc.)
    """
    kwargs.setdefault('color', 'black')
    kwargs.setdefault('linewidth', 1.5)

    # Draw bar
    ax.plot([x1, x1, x2, x2], [y, y + line_height, y + line_height, y], **kwargs)

    # Add text
    ax.text((x1 + x2) / 2, y + line_height + text_offset, text,
            ha='center', va='bottom', fontsize=11, fontweight='bold')


def format_pvalue_stars(p):
    """Convert p-value to significance stars.

    Args:
        p: p-value

    Returns:
        String with stars ('***', '**', '*', 'ns')
    """
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def create_figure_grid(nrows, ncols, figsize=None, **subplot_kw):
    """Create figure with subplot grid using publication style.

    Args:
        nrows: number of rows
        ncols: number of columns
        figsize: figure size (if None, calculated based on grid)
        **subplot_kw: additional subplot arguments

    Returns:
        fig, axes
    """
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **subplot_kw)
    return fig, axes


def add_panel_label(ax, label, x=-0.15, y=1.1, fontsize=20, fontweight='bold'):
    """Add panel label (A, B, C, etc.) to subplot.

    Args:
        ax: matplotlib axis
        label: label text (e.g., 'A', 'B')
        x, y: position in axis coordinates
        fontsize: font size
        fontweight: font weight
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight, va='top', ha='right')


def despine_all(fig):
    """Remove top and right spines from all axes in figure.

    Args:
        fig: matplotlib figure
    """
    for ax in fig.get_axes():
        sns.despine(ax=ax)


def compute_errorbar_data(data, x, y, hue=None, errorbar='se'):
    """
    Aggregate data for plotting with error bars on continuous x-axis.

    Computes mean and error (SEM or SD) for y variable, grouped by x and optionally hue.
    Returns data in format ready for scatter + errorbar plotting.

    Args:
        data: DataFrame with data to aggregate
        x: Column name for x-axis (continuous variable)
        y: Column name for y-axis
        hue: Column name for grouping (optional)
        errorbar: Error type - 'se' for SEM, 'sd' for SD

    Returns:
        DataFrame with columns: [hue], x, y_mean, y_err

    Example:
        >>> agg_data = compute_errorbar_data(df, x='curDur', y='bias', hue='curTask')
        >>> # Returns: curTask, curDur, y_mean, y_err
    """
    import pandas as pd

    group_cols = [x] if hue is None else [hue, x]
    agg_func = 'sem' if errorbar == 'se' else 'std'

    result = (
        data
        .groupby(group_cols, as_index=False)
        .agg(y_mean=(y, 'mean'), y_err=(y, agg_func))
    )

    return result


def plot_points_with_errors(data, x, y, hue=None, ax=None, palette=None,
                            errorbar='se', capsize=0.1, markersize=8,
                            **kwargs):
    """
    Plot points with error bars on continuous x-axis (for FacetGrid compatibility).

    This function can be mapped to FacetGrid just like sns.pointplot(),
    but correctly handles continuous x-variables by using scatter + errorbar
    instead of pointplot's categorical approach.

    Args:
        data: DataFrame with data to plot
        x: Column name for x-axis (continuous)
        y: Column name for y-axis
        hue: Column name for grouping (optional)
        ax: Matplotlib axis (if None, uses plt.gca())
        palette: Color palette dict
        errorbar: Error type ('se' or 'sd')
        capsize: Error bar cap size (default 0.1)
        markersize: Marker size for points (default 8)
        **kwargs: Additional arguments (for compatibility)

    Returns:
        matplotlib axis

    Example:
        >>> g = sns.FacetGrid(df, col='task', hue='condition')
        >>> g.map_dataframe(plot_points_with_errors, x='duration', y='bias')
    """
    if ax is None:
        ax = plt.gca()

    # Aggregate data
    agg_data = compute_errorbar_data(data, x, y, hue, errorbar)

    if hue:
        # Plot separately for each hue group
        for group in data[hue].unique():
            group_agg = agg_data[agg_data[hue] == group]
            color = palette.get(group) if palette else None

            # Scatter points
            ax.scatter(
                group_agg[x], group_agg['y_mean'],
                color=color, s=markersize**2, zorder=3,
                label=group
            )

            # Error bars
            ax.errorbar(
                group_agg[x], group_agg['y_mean'],
                yerr=group_agg['y_err'],
                fmt='none', color=color, capsize=capsize,
                linewidth=1.5, capthick=1.5, zorder=2
            )
    else:
        # Single group
        ax.scatter(
            agg_data[x], agg_data['y_mean'],
            s=markersize**2, zorder=3
        )

        ax.errorbar(
            agg_data[x], agg_data['y_mean'],
            yerr=agg_data['y_err'],
            fmt='none', capsize=capsize,
            linewidth=1.5, capthick=1.5, zorder=2
        )

    return ax


def plot_with_regression(data, x, y, hue=None, ax=None, palette=None,
                         errorbar='se', markers=True, capsize=0.1,
                         markersize=8, **kwargs):
    """
    Declarative wrapper: points with error bars + regression line.

    FIXED: Now uses scatter + errorbar instead of pointplot to correctly
    handle continuous x-variables (no categorical position mismatch).

    Args:
        data: DataFrame with data to plot
        x: Column name for x-axis (continuous variable)
        y: Column name for y-axis
        hue: Column name for grouping (optional)
        ax: Matplotlib axis (creates new if None)
        palette: Color palette dict (e.g., TASK_COLORS)
        errorbar: Error bar type ('se' or 'sd')
        markers: If True, plot points with error bars; if False, only regression
        capsize: Error bar cap size (default 0.1, much smaller than pointplot default)
        markersize: Marker size for scatter points (default 8)
        **kwargs: Additional arguments (for compatibility)

    Returns:
        matplotlib axis

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_with_regression(
        ...     df, x='curDur', y='bias', hue='curTask',
        ...     ax=ax, palette=TASK_COLORS, capsize=0.1
        ... )
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Plot points with error bars using scatter + errorbar (continuous x-axis)
    if markers:
        agg_data = compute_errorbar_data(data, x, y, hue, errorbar)

        if hue:
            # Plot separately for each hue group
            for group in data[hue].unique():
                group_agg = agg_data[agg_data[hue] == group]
                color = palette.get(group) if palette else None

                # Scatter points
                ax.scatter(
                    group_agg[x], group_agg['y_mean'],
                    color=color, s=markersize**2, zorder=3,
                    label=group
                )

                # Error bars
                ax.errorbar(
                    group_agg[x], group_agg['y_mean'],
                    yerr=group_agg['y_err'],
                    fmt='none', color=color, capsize=capsize,
                    linewidth=1.5, capthick=1.5, zorder=2
                )
        else:
            # Single group
            ax.scatter(
                agg_data[x], agg_data['y_mean'],
                s=markersize**2, zorder=3
            )

            ax.errorbar(
                agg_data[x], agg_data['y_mean'],
                yerr=agg_data['y_err'],
                fmt='none', capsize=capsize,
                linewidth=1.5, capthick=1.5, zorder=2
            )

    # Add regression line per group (uses continuous x-scale)
    if hue:
        for group in data[hue].unique():
            group_data = data[data[hue] == group]
            color = palette.get(group) if palette else None
            sns.regplot(
                data=group_data, x=x, y=y,
                scatter=False, color=color, ax=ax,
                line_kws={'linestyle': '--', 'alpha': 0.7, 'linewidth': 1.5}
            )
    else:
        sns.regplot(
            data=data, x=x, y=y,
            scatter=False, ax=ax,
            line_kws={'linestyle': '--', 'alpha': 0.7}
        )

    return ax


def create_facet_with_regression(data, x, y, col=None, hue=None,
                                 palette=None, col_wrap=None,
                                 height=5, aspect=1.2, **kwargs):
    """
    Create FacetGrid with points + regression lines.

    Handles the common pattern of faceted plots with regression overlays.

    Args:
        data: DataFrame in tidy format
        x: Column for x-axis
        y: Column for y-axis
        col: Column for faceting
        hue: Column for color grouping
        palette: Color palette
        col_wrap: Number of columns before wrapping
        height: Height of each facet
        aspect: Aspect ratio
        **kwargs: Additional FacetGrid arguments

    Returns:
        seaborn FacetGrid

    Example:
        >>> g = create_facet_with_regression(
        ...     df_seq, x='preDur', y='bias',
        ...     col='curTask', hue='taskSwitch',
        ...     palette={'repeat': 'red', 'switch': 'blue'}
        ... )
    """
    # Create FacetGrid
    g = sns.FacetGrid(
        data=data,
        col=col, hue=hue,
        palette=palette,
        col_wrap=col_wrap,
        height=height, aspect=aspect,
        legend_out=False,
        **kwargs
    )

    # Map pointplot for error bars
    g.map_dataframe(
        sns.pointplot,
        x=x, y=y,
        errorbar='se',
        capsize=0.1
    )

    # Add regression lines
    # This is a bit tricky with FacetGrid - need to map regplot
    g.map_dataframe(
        sns.regplot,
        x=x, y=y,
        scatter=False,
        line_kws={'linestyle': '--', 'alpha': 0.7}
    )

    g.add_legend()

    return g


def add_regression_lines_to_facet(facet_grid, data, x, y,
                                  col_var, hue_var, palette):
    """
    Add regression lines to existing FacetGrid by facet and hue.

    Helper for more complex faceted plots where we need manual control
    over regression lines.

    Args:
        facet_grid: seaborn FacetGrid instance
        data: Source DataFrame
        x: x-axis column
        y: y-axis column
        col_var: Faceting column name
        hue_var: Hue grouping column name
        palette: Color palette dict

    Example:
        >>> g = sns.FacetGrid(df, col='curTask', hue='taskSwitch')
        >>> g.map_dataframe(sns.pointplot, x='preDur', y='bias')
        >>> add_regression_lines_to_facet(
        ...     g, df, 'preDur', 'bias',
        ...     'curTask', 'taskSwitch', {'repeat': 'red', 'switch': 'blue'}
        ... )
    """
    # Get unique values for faceting and hue
    col_vals = data[col_var].unique() if col_var else [None]
    hue_vals = data[hue_var].unique() if hue_var else [None]

    # Iterate through facets
    for col_val in col_vals:
        # Get appropriate axis
        if hasattr(facet_grid, 'axes_dict'):
            ax = facet_grid.axes_dict[col_val]
        elif len(facet_grid.axes.flat) == 1:
            ax = facet_grid.axes.flat[0]
        else:
            # Multiple facets in array
            ax_idx = list(col_vals).index(col_val)
            ax = facet_grid.axes.flat[ax_idx]

        # Add regression per hue group
        for hue_val in hue_vals:
            # Filter data
            if col_var and hue_var:
                subset = data.query(f'{col_var} == @col_val and {hue_var} == @hue_val')
            elif col_var:
                subset = data.query(f'{col_var} == @col_val')
            elif hue_var:
                subset = data.query(f'{hue_var} == @hue_val')
            else:
                subset = data

            if len(subset) >= 3:
                color = palette.get(hue_val) if palette and hue_var else None
                sns.regplot(
                    data=subset, x=x, y=y,
                    scatter=False, color=color, ax=ax,
                    line_kws={'linestyle': '--', 'alpha': 0.7, 'linewidth': 2}
                )
