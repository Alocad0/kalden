from typing import Any

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as g


def plotly_fig_add_filled_rectangle(
    fig: go.Figure,
    x_start: Any,
    x_end: Any,
    y_top: float,
    y_bottom: float,
    *,
    fillcolor: str = "rgba(44, 160, 44, 0.3)",
    name: str = "",
    showlegend: bool = False,
    legendgroup: str | None = None,
) -> go.Figure:
    """
    Add a filled rectangular scatter trace to a Plotly figure.

    The rectangle spans from ``x_start`` to ``x_end`` horizontally and from
    ``y_bottom`` to ``y_top`` vertically. It is implemented as a closed
    Plotly scatter polygon.

    Parameters
    ----------
    fig
        Plotly figure to modify.
    x_start, x_end
        Horizontal bounds of the rectangle. Values may be numeric,
        categorical, or datetime-like.
    y_top, y_bottom
        Vertical bounds of the rectangle.
    fillcolor
        Plotly-compatible fill color, such as an RGB, RGBA, or hexadecimal
        color string.
    name
        Trace name displayed in the legend.
    showlegend
        Whether to display this trace in the legend.
    legendgroup
        Optional legend group shared with other traces.

    Returns
    -------
    plotly.graph_objects.Figure
        The modified input figure.
    """
    fig.add_trace(
        go.Scatter(
            x=[x_start, x_end, x_end, x_start, x_start],
            y=[y_top, y_top, y_bottom, y_bottom, y_top],
            mode="lines",
            fill="toself",
            fillcolor=fillcolor,
            line={"width": 0},
            name=name,
            showlegend=showlegend,
            legendgroup=legendgroup,
            hoverinfo="skip",
        )
    )

    return fig

def save_plotly_fig(fig, filepath, width=1200, height=600, **kwargs):
    """
    Save a Plotly figure to a PNG file.
    
    Args:
        fig: Plotly Figure object or dict.
        filepath (str): Path to save the PNG file (e.g., 'plot.png').
        **kwargs: Additional arguments for fig.write_image(), such as:
            - width (int): Image width in pixels.
            - height (int): Image height in pixels.
            - scale (float): Resolution scale factor (>1 for higher DPI).
            - format (str): Image format ('png', 'jpeg', etc.).
            - engine (str): Export engine ('kaleido' or deprecated 'orca').
    
    Example:
        fig = go.Figure(go.Scatter(x=[1,2], y=[1,2]))
        save_plotly_to_png(fig, 'myplot.png', width=800, height=600, scale=2)
    """
    fig.write_image(filepath, width=width, height=height, **kwargs)
    print(f"Figure saved to {filepath}")


def heatmap_colorscale(values, plotly_colorscale="RdBu_r", zmin=None, zmax=None):
    """
    Take a Plotly named colorscale and shift its midpoint so that the
    center color lands at the position of 0 between zmin and zmax.

    Parameters
    ----------
    values : array-like
        Data values, only used if zmin/zmax are not provided.
    plotly_colorscale : str or list
        Plotly named colorscale, e.g. "RdBu_r", "BrBG", "PuOr".
    zmin, zmax : float or None
        Optional explicit bounds.

    Returns
    -------
    colorscale, zmin, zmax
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0 and (zmin is None or zmax is None):
        raise ValueError("`values` contains no finite values, and zmin/zmax were not provided.")

    zmin = float(arr.min()) if zmin is None else float(zmin)
    zmax = float(arr.max()) if zmax is None else float(zmax)

    if zmin >= zmax:
        raise ValueError(f"zmin must be < zmax, got zmin={zmin}, zmax={zmax}")

    if not (zmin <= 0 <= zmax):
        raise ValueError(f"Zero must lie within [zmin, zmax], got zmin={zmin}, zmax={zmax}")

    zero_pos = (0.0 - zmin) / (zmax - zmin)

    # Get the original Plotly colorscale as [[pos, color], ...]
    base = pc.get_colorscale(plotly_colorscale)

    shifted = []
    for p, c in base:
        p = float(p)

        if p <= 0.5:
            # map [0, 0.5] -> [0, zero_pos]
            new_p = 0.0 if zero_pos == 0 else (p / 0.5) * zero_pos
        else:
            # map [0.5, 1] -> [zero_pos, 1]
            new_p = 1.0 if zero_pos == 1 else zero_pos + ((p - 0.5) / 0.5) * (1.0 - zero_pos)

        shifted.append([new_p, c])

    return shifted, zmin, zmax
