# -*- coding: utf-8 -*-
"""
Created on Friday March 6th 2025

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import ipywidgets as widgets  # type: ignore
import plotly.graph_objs as go  # type: ignore

from plotly.subplots import make_subplots  # type: ignore
from IPython.display import display  # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)


# Get an interactive plot for reviewing the tracked videos ----
def interactive_plot(
    df: pd.DataFrame,
    interval_size: int = 100
):
    """Construct an interactive plot for reviewing the tracked videos. The data
    has the following columns:
        - permuted_id: Particle ID after smoothing process
        - time: Times (Frames)
        - position_x: Centroid position in X-axis
        - position_y: Centroid position in Y-axis
        - corrected_orientation: Orientation after smoothing process
        - n_x: Normalized position in X-axis using video resolution and the
        distance respect to the center in such a way that n_x is between -0.5
        and 0.5.
        - n_y: Normalized position in Y-axis using video resolution and the
        distance respect to the center in such a way that n_y is between -0.5
        and 0.5.
        - n_orientation: Sine of the orientation respect to the vertical
        measured from 0 (positive Y-axis) to 2*pi (counter clockwise)
        - norm: Square of the euclidean norm of the vector built from n_x, n_y,
        and n_orientation.

    Args:
    ---------------------------------------------------------------------------
    df : pd.DataFrame
        Dataframe with the information of tracked videos
    interval_size : int
        Interval of frames showing in the plots

    Returns:
    ---------------------------------------------------------------------------
    Interactive plot with the following figures:
        - Position in X axis as a function of time (frames)
        - Position in Y axis as a function of time (frames)
        - Orientation as a function of time (frames)
        - Distance from the center as a function of time (frames)
        - n_orientation as a function of time (frames)
        - Norm as a function of time (frames)
    """

    # Compute derived quantities
    df = df.copy()
    df["distance"] = np.power(df["n_x"]**2 + df["n_y"]**2, 1 / 2)
    df["time"] = pd.to_numeric(df["time"])

    # Plot options (Video and time interval)
    video_options = sorted(df["video"].unique())
    time_min = int(df["time"].min())
    time_max = int(df["time"].max())

    # Widgets
    video_dropdown = widgets.Dropdown(
        options=video_options,
        description="Video:",
        layout=widgets.Layout(width="50%")
    )
    time_slider = widgets.IntRangeSlider(
        value=[time_min, time_min + interval_size],
        min=time_min,
        max=time_max,
        step=1,
        description="Time Range:",
        continuous_update=False,
        layout=widgets.Layout(width="95%")
    )
    output = widgets.Output()

    # Plots
    def update_plot(video, time_range):
        with output:
            output.clear_output()
            mask = (
                (df["video"] == video) &
                (df["time"] >= time_range[0]) &
                (df["time"] <= time_range[1])
            )
            df_video = df[mask]
            if df_video.empty:
                print("No data in the selected time range.")
                return

            fig = make_subplots(
                rows=2,
                cols=3,
                shared_xaxes=False,
                shared_yaxes=False,
                horizontal_spacing=0.07,
                vertical_spacing=0.12,
                subplot_titles=[
                    "X(t)", "Y(t)", "Orientation",
                    "Distance from center", "Sine - Orientation", "Norm $n(t)$"
                ]
            )

            colors = {}  # map permuted_id to color index
            color_palette = px.colors.qualitative.Plotly

            ids = df_video["permuted_id"].unique()
            for idx, id_ in enumerate(sorted(ids)):
                df_id = df_video[df_video["permuted_id"] == id_]
                label = video[3:5] + video[6:8] + " - " + str(id_)

                # Assign a color for each ID
                if id_ not in colors:
                    colors[id_] = color_palette[idx % len(color_palette)]

                color = colors[id_]

                fig.add_trace(go.Scatter(
                    x=df_id["time"],
                    y=df_id["position_x"],
                    mode="lines+markers",
                    name=label,
                    legendgroup=str(id_),
                    showlegend=True,
                    line=dict(color=color, width=0.7),
                    marker=dict(color=color)
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=df_id["time"],
                    y=df_id["position_y"],
                    mode="lines+markers",
                    name=label,
                    legendgroup=str(id_),
                    showlegend=False,
                    line=dict(color=color, width=0.7),
                    marker=dict(color=color)
                ), row=1, col=2)

                fig.add_trace(go.Scatter(
                    x=df_id["time"],
                    y=(df_id["corrected_orientation"] + np.pi) % (2 * np.pi),
                    mode="lines+markers",
                    name=label,
                    legendgroup=str(id_),
                    showlegend=False,
                    line=dict(color=color, width=0.7),
                    marker=dict(color=color)
                ), row=1, col=3)

                fig.add_trace(go.Scatter(
                    x=df_id["time"],
                    y=df_id["distance"],
                    mode="lines+markers",
                    name=label,
                    legendgroup=str(id_),
                    showlegend=False,
                    line=dict(color=color, width=0.7),
                    marker=dict(color=color)
                ), row=2, col=1)

                fig.add_trace(go.Scatter(
                    x=df_id["time"],
                    y=df_id["n_orientation"],
                    mode="lines+markers",
                    name=label,
                    legendgroup=str(id_),
                    showlegend=False,
                    line=dict(color=color, width=0.7),
                    marker=dict(color=color)
                ), row=2, col=2)

                fig.add_trace(go.Scatter(
                    x=df_id["time"],
                    y=df_id["norm"],
                    mode="lines+markers",
                    name=label,
                    legendgroup=str(id_),
                    showlegend=False,
                    line=dict(color=color, width=0.7),
                    marker=dict(color=color)
                ), row=2, col=3)

            # Axis styling: 20 major ticks, 5 minor ticks between
            for r in range(1, 3):
                for c in range(1, 4):
                    fig.update_xaxes(
                        row=r,
                        col=c,
                        title_text="Time (t)",
                        ticks="inside",
                        showgrid=True,
                        tickfont=dict(size=11),
                        tickmode="auto",
                        tickangle=90,
                        nticks=20,
                        mirror=True,  # box around
                        showline=True,
                        ticklen=8
                    )
                    fig.update_yaxes(
                        row=r,
                        col=c,
                        ticks="inside",
                        showgrid=True,
                        tickfont=dict(size=11),
                        tickmode="auto",
                        nticks=20,
                        mirror=True,
                        showline=True,
                        ticklen=8
                    )

            # Layout
            fig.update_layout(
                height=800,
                width=1100,
                title=f"Time Range: {time_range[0]}-{time_range[1]}",
                template="plotly_white",
                margin=dict(l=30, r=10, t=60, b=40),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=11)
                )
            )
            fig.show()

    interactive = widgets.interactive_output(  # noqa: 501
        update_plot,
        {"video": video_dropdown, "time_range": time_slider}
    )

    display(widgets.VBox([video_dropdown, time_slider, output]))
