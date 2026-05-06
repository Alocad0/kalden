"""Optional plotting and notebook widgets for :mod:`kalden.core.mike.res1d`.

The reader/cache implementation lives in ``res1d.py``. This module is kept
separate because it may need optional dependencies such as ``plotly``,
``matplotlib``, ``ipywidgets``, and ``IPython``. Those packages are imported
lazily inside the functions that need them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

__all__ = [
    "normalize_timeseries_for_plot",
    "plot_timeseries",
    "res1d_timeseries_widget",
]


def _require_plotly():
    """Import Plotly lazily and return ``plotly.express``."""

    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for this feature. Install it with: "
            "pip install plotly"
        ) from exc
    return px


def _require_ipywidgets():
    """Import notebook widget dependencies lazily."""

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            "ipywidgets and IPython are required for the res1d widget. "
            "Install them with: pip install ipywidgets IPython"
        ) from exc
    return widgets, display


def _as_dataframe(ts: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Return a copy of ``ts`` as a DataFrame with string column names."""

    if isinstance(ts, pd.Series):
        frame = ts.to_frame(name=ts.name or "value")
    else:
        frame = ts.copy()

    frame.index = pd.to_datetime(frame.index)
    frame.index.name = frame.index.name or "time"
    frame.columns = [str(column) for column in frame.columns]
    return frame.sort_index()


def normalize_timeseries_for_plot(
    ts: pd.DataFrame | pd.Series,
    quantity: str,
    object_id: str,
    object_type: str,
    value_columns: Sequence[str] | None = None,
    dropna: bool = False,
) -> pd.DataFrame:
    """Return a long-format DataFrame suitable for plotting.

    Parameters
    ----------
    ts : pandas.DataFrame or pandas.Series
        Time series returned by ``Res1D.read_series``.
    quantity : str
        Result quantity name, for example ``"WaterLevel"`` or ``"Discharge"``.
    object_id : str
        Public object ID.
    object_type : str
        Public object type, for example ``"node"``, ``"reach"``, ``"weir"``,
        or ``"pump"``.
    value_columns : sequence of str, optional
        Subset of time series columns to keep before melting to long format.
    dropna : bool, default False
        Drop rows where the plotted value is missing.

    Returns
    -------
    pandas.DataFrame
        Columns: ``time``, ``series``, ``value``, ``quantity``, ``object_id``,
        and ``object_type``.
    """

    frame = _as_dataframe(ts)

    if value_columns is not None:
        missing = [column for column in value_columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Columns not found in time series: {missing}")
        frame = frame.loc[:, list(value_columns)]

    long = frame.reset_index().melt(
        id_vars=frame.index.name or "time",
        var_name="series",
        value_name="value",
    )

    # Make sure the first column is consistently called "time" even if the
    # source index had a different name.
    first_column = long.columns[0]
    if first_column != "time":
        long = long.rename(columns={first_column: "time"})

    long["quantity"] = str(quantity)
    long["object_id"] = str(object_id)
    long["object_type"] = str(object_type)

    if dropna:
        long = long.dropna(subset=["value"])

    return long


def plot_timeseries(
    ts: pd.DataFrame | pd.Series,
    quantity: str,
    object_id: str,
    object_type: str,
    backend: str = "plotly",
    title: str | None = None,
    value_columns: Sequence[str] | None = None,
    dropna: bool = False,
    width: int | None = None,
    height: int = 450,
    template: str = "plotly_white",
    show: bool = True,
    ax: Any | None = None,
) -> Any:
    """Plot one res1d time series with Plotly or Matplotlib.

    The function returns the Plotly ``Figure`` or Matplotlib ``Axes`` object.
    """

    data = normalize_timeseries_for_plot(
        ts=ts,
        quantity=quantity,
        object_id=object_id,
        object_type=object_type,
        value_columns=value_columns,
        dropna=dropna,
    )

    if data.empty:
        raise ValueError("The selected time series is empty.")

    backend_normalized = backend.strip().lower()
    plot_title = title or f"{object_type} {object_id} - {quantity}"

    if backend_normalized == "plotly":
        px = _require_plotly()
        fig = px.line(
            data,
            x="time",
            y="value",
            color="series",
            title=plot_title,
            template=template,
        )
        fig.update_layout(height=height)
        if width is not None:
            fig.update_layout(width=width)
        if show:
            fig.show()
        return fig

    if backend_normalized == "matplotlib":
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Matplotlib is required for backend='matplotlib'. Install it with: "
                "pip install matplotlib"
            ) from exc

        if ax is None:
            _, ax = plt.subplots(figsize=(10, max(height / 100, 2)))

        for series_name, group in data.groupby("series", sort=False):
            ax.plot(group["time"], group["value"], label=str(series_name))

        ax.set_title(plot_title)
        ax.set_xlabel("time")
        ax.set_ylabel(str(quantity))
        ax.legend()

        if show:
            plt.show()
        return ax

    raise ValueError("backend must be one of: 'plotly', 'matplotlib'")


def _first_or_none(values: Sequence[str]) -> str | None:
    return values[0] if values else None


def _limited_objects(
    explorer: Any,
    *,
    object_type: str,
    quantity: str | None,
    max_object_ids: int,
    contains: str = "",
) -> list[str]:
    """Return object IDs filtered for widget dropdown display."""

    object_ids = explorer.available_objects(
        object_type=object_type,
        quantity=quantity,
    )
    if contains:
        needle = contains.lower()
        object_ids = [item for item in object_ids if needle in item.lower()]
    return object_ids[:max_object_ids]


def res1d_timeseries_widget(
    explorer: Any,
    object_types: Sequence[str] = ("node", "reach", "weir", "pump"),
    default_object_type: str = "node",
    default_quantity: str | None = None,
    default_object_id: str | None = None,
    default_backend: str = "plotly",
    max_object_ids: int = 200,
    preview_rows: int = 5,
    keep_history: bool = False,
) -> Any:
    """Return an interactive notebook widget for exploring res1d series.

    The ``explorer`` argument is expected to be a ``Res1D`` instance or any
    object exposing the same methods: ``available_object_types``,
    ``available_quantities``, ``available_objects``, ``read_series``, and
    ``plot_series``.

    Parameters
    ----------
    keep_history : bool, default False
        If True, each update is appended to the output area instead of replacing
        the previous plot. This is useful for visually comparing several
        objects/quantities in a notebook. It does not keep all loaded time
        series DataFrames in memory; each selected series is still read through
        the explorer's normal lazy cache mechanism.
    """

    widgets, display = _require_ipywidgets()

    available_types = set(explorer.available_object_types())
    selectable_types = [
        object_type for object_type in object_types if object_type in available_types
    ]
    if not selectable_types:
        raise ValueError(
            "No matching object types are available. Available types are: "
            f"{sorted(available_types)}"
        )

    if default_object_type not in selectable_types:
        default_object_type = selectable_types[0]

    quantities = explorer.available_quantities(object_type=default_object_type)
    if not quantities:
        raise ValueError(f"No quantities found for object_type={default_object_type!r}.")

    if default_quantity not in quantities:
        default_quantity = quantities[0]

    object_ids = _limited_objects(
        explorer,
        object_type=default_object_type,
        quantity=default_quantity,
        max_object_ids=max_object_ids,
    )
    if not object_ids:
        raise ValueError(
            f"No objects found for object_type={default_object_type!r} "
            f"and quantity={default_quantity!r}."
        )

    if default_object_id not in object_ids:
        default_object_id = object_ids[0]

    backend_options = ("plotly", "matplotlib")
    if default_backend not in backend_options:
        default_backend = "plotly"

    object_type_dropdown = widgets.Dropdown(
        options=selectable_types,
        value=default_object_type,
        description="Type:",
        layout=widgets.Layout(width="220px"),
    )
    quantity_dropdown = widgets.Dropdown(
        options=quantities,
        value=default_quantity,
        description="Quantity:",
        layout=widgets.Layout(width="300px"),
    )
    object_filter = widgets.Text(
        value="",
        placeholder="filter object id",
        description="Filter:",
        layout=widgets.Layout(width="300px"),
    )
    object_dropdown = widgets.Dropdown(
        options=object_ids,
        value=default_object_id,
        description="Object:",
        layout=widgets.Layout(width="520px"),
    )
    backend_dropdown = widgets.Dropdown(
        options=backend_options,
        value=default_backend,
        description="Backend:",
        layout=widgets.Layout(width="220px"),
    )
    force_refresh_checkbox = widgets.Checkbox(
        value=False,
        description="Force refresh",
        indent=False,
        layout=widgets.Layout(width="160px"),
    )
    keep_history_checkbox = widgets.Checkbox(
        value=keep_history,
        description="Keep history",
        indent=False,
        layout=widgets.Layout(width="150px"),
    )
    update_button = widgets.Button(
        description="Update",
        button_style="",
        tooltip="Read/cache the selected series and update the plot.",
    )
    clear_history_button = widgets.Button(
        description="Clear history",
        button_style="",
        tooltip="Clear the widget output area.",
    )
    output = widgets.Output()

    updating_controls = False
    history_count = 0

    def set_quantities_for_type() -> None:
        object_type = object_type_dropdown.value
        new_quantities = explorer.available_quantities(object_type=object_type)
        quantity_dropdown.options = new_quantities
        if not new_quantities:
            quantity_dropdown.value = None
            object_dropdown.options = []
            return
        current_quantity = quantity_dropdown.value
        quantity_dropdown.value = (
            current_quantity if current_quantity in new_quantities else new_quantities[0]
        )

    def set_objects_for_selection() -> None:
        object_type = object_type_dropdown.value
        quantity = quantity_dropdown.value
        if not object_type or not quantity:
            object_dropdown.options = []
            return

        new_object_ids = _limited_objects(
            explorer,
            object_type=object_type,
            quantity=quantity,
            max_object_ids=max_object_ids,
            contains=object_filter.value.strip(),
        )
        object_dropdown.options = new_object_ids
        if not new_object_ids:
            object_dropdown.value = None
            return
        current_object_id = object_dropdown.value
        object_dropdown.value = (
            current_object_id if current_object_id in new_object_ids else new_object_ids[0]
        )

    def clear_history(*_: Any) -> None:
        nonlocal history_count
        history_count = 0
        output.clear_output(wait=True)

    def update_plot(*_: Any) -> None:
        nonlocal history_count

        if updating_controls:
            return

        object_type = object_type_dropdown.value
        quantity = quantity_dropdown.value
        object_id = object_dropdown.value

        if not keep_history_checkbox.value:
            clear_history()

        with output:
            if not object_type or not quantity or not object_id:
                print("No object, quantity, or object type selected.")
                return

            history_count += 1
            if keep_history_checkbox.value:
                print("\n" + "=" * 80)
                print(f"[{history_count}] {object_type} {object_id} - {quantity}")
            else:
                print(f"{object_type} {object_id} - {quantity}")

            try:
                ts = explorer.read_series(
                    object_type=object_type,
                    object_id=object_id,
                    quantity=quantity,
                    force_refresh=force_refresh_checkbox.value,
                )
                if preview_rows > 0:
                    display(ts.head(preview_rows))
                plot_timeseries(
                    ts=ts,
                    object_type=object_type,
                    object_id=object_id,
                    quantity=quantity,
                    backend=backend_dropdown.value,
                    show=True,
                )
            except Exception as exc:  # pragma: no cover - UI feedback path
                print(f"Could not plot selected series: {exc}")

    def on_type_change(*_: Any) -> None:
        nonlocal updating_controls
        updating_controls = True
        try:
            set_quantities_for_type()
            set_objects_for_selection()
        finally:
            updating_controls = False
        update_plot()

    def on_quantity_or_filter_change(*_: Any) -> None:
        nonlocal updating_controls
        updating_controls = True
        try:
            set_objects_for_selection()
        finally:
            updating_controls = False
        update_plot()

    object_type_dropdown.observe(on_type_change, names="value")
    quantity_dropdown.observe(on_quantity_or_filter_change, names="value")
    object_filter.observe(on_quantity_or_filter_change, names="value")
    object_dropdown.observe(update_plot, names="value")
    backend_dropdown.observe(update_plot, names="value")
    force_refresh_checkbox.observe(update_plot, names="value")
    update_button.on_click(update_plot)
    clear_history_button.on_click(clear_history)

    controls = widgets.VBox(
        [
            widgets.HBox([object_type_dropdown, quantity_dropdown]),
            widgets.HBox([object_filter, object_dropdown]),
            widgets.HBox(
                [
                    backend_dropdown,
                    force_refresh_checkbox,
                    keep_history_checkbox,
                    update_button,
                    clear_history_button,
                ]
            ),
        ]
    )
    widget = widgets.VBox([controls, output])

    update_plot()
    return widget
