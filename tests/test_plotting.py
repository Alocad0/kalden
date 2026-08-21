import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import pytest

from kalden.misc.plotting import (
    heatmap_colorscale,
    plotly_fig_add_filled_rectangle,
    plotly_palette_repeat,
)


def test_plotly_palette_repeat_cycles_to_requested_length() -> None:
    assert plotly_palette_repeat(["red", "blue"], 5) == [
        "red",
        "blue",
        "red",
        "blue",
        "red",
    ]
    assert plotly_palette_repeat(["red"], 0) == []


@pytest.mark.parametrize(
    ("palette", "count", "message"),
    [([], 1, "at least one color"), (["red"], -1, "non-negative")],
)
def test_plotly_palette_repeat_validates_inputs(palette, count, message) -> None:
    with pytest.raises(ValueError, match=message):
        plotly_palette_repeat(palette, count)


def test_filled_rectangle_adds_closed_polygon_trace() -> None:
    figure = go.Figure()

    result = plotly_fig_add_filled_rectangle(
        figure,
        1,
        3,
        5,
        -2,
        fillcolor="red",
        name="limit",
        showlegend=True,
        legendgroup="limits",
    )

    assert result is figure
    assert len(figure.data) == 1
    trace = figure.data[0]
    assert list(trace.x) == [1, 3, 3, 1, 1]
    assert list(trace.y) == [5, 5, -2, -2, 5]
    assert trace.fill == "toself"
    assert trace.fillcolor == "red"
    assert trace.hoverinfo == "skip"


def test_heatmap_colorscale_moves_midpoint_to_zero_position() -> None:
    shifted, zmin, zmax = heatmap_colorscale(
        [-1, np.nan, 3],
        plotly_colorscale="RdBu",
    )
    base = pc.get_colorscale("RdBu")
    midpoint_index = next(
        index for index, (position, _) in enumerate(base) if position == 0.5
    )

    assert (zmin, zmax) == (-1.0, 3.0)
    assert shifted[0][0] == 0.0
    assert shifted[-1][0] == 1.0
    assert shifted[midpoint_index][0] == pytest.approx(0.25)
    assert [position for position, _ in shifted] == sorted(
        position for position, _ in shifted
    )


@pytest.mark.parametrize(
    ("values", "zmin", "zmax", "message"),
    [
        ([np.nan], None, None, "no finite values"),
        ([1, 2], 2, 2, "zmin must be < zmax"),
        ([1, 2], None, None, "Zero must lie within"),
    ],
)
def test_heatmap_colorscale_rejects_invalid_bounds(values, zmin, zmax, message) -> None:
    with pytest.raises(ValueError, match=message):
        heatmap_colorscale(values, zmin=zmin, zmax=zmax)
