from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Iterable, Sequence

import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # Plotly is optional at import time; the UI falls back gracefully.
    go = None

STAGES = ["Dataset", "YOLO", "OpenCLIP", "Model", "Training", "Evaluation", "Results"]
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _asset_data_uri(filename: str) -> str:
    path = ASSETS_DIR / filename
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def brand(active_stage: str) -> None:
    if active_stage in {"Dataset", "YOLO"}:
        active_tech = "YOLO"
    elif active_stage == "OpenCLIP":
        active_tech = "OpenCLIP"
    elif active_stage in {"Model", "Training"}:
        active_tech = "Mamba"
    else:
        active_tech = "Behavioral Cloning"

    pills = "".join(
        f'<span class="tech-pill{" active" if label == active_tech else ""}">{html.escape(label)}</span>'
        for label in ["YOLO", "OpenCLIP", "Mamba", "Behavioral Cloning"]
    )
    st.markdown(
        f"""
        <div class="brand-row">
          <div>
            <div class="brand-title">RoboMamba</div>
            <div class="brand-sub">OpenCLIP embeddings + Mamba sequence modeling</div>
          </div>
          <div class="pill-row">{pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stage_stepper(active_stage: str, states: dict[str, str]) -> None:
    parts: list[str] = []
    active_index = STAGES.index(active_stage)
    for index, label in enumerate(STAGES):
        state = states.get(label, "not_started")
        if state in {"completed", "ready"}:
            number, css = "✓", "done"
        elif state == "skipped":
            number, css = "–", "skipped"
        elif state in {"existing", "partial", "stopped"}:
            number, css = str(index + 1), "available"
        elif state == "failed":
            number, css = "!", "failed"
        elif state == "running" or index == active_index:
            number, css = str(index + 1), "active"
        else:
            number, css = str(index + 1), "idle"
        connector = '<span class="step-line"></span>' if index < len(STAGES) - 1 else ""
        parts.append(
            f'<div class="step-wrap"><div class="step {css}">{number}</div>'
            f'<div class="step-label">{html.escape(label)}</div></div>{connector}'
        )
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def robot_panel(title: str, subtitle: str, framed: bool = True) -> None:
    filename = "robot_hero_framed.png" if framed else "robot_hero.png"
    uri = _asset_data_uri(filename)
    image = f'<img src="{uri}" alt="Robotic arm" />' if uri else '<div class="missing-asset">Add ui/assets/robot_hero_framed.png</div>'
    st.markdown(
        f"""
        <div class="robot-panel">
          <div class="robot-image">{image}</div>
          <div class="robot-context">
            <div class="robot-context-title">{html.escape(title)}</div>
            <div class="robot-context-sub">{html.escape(subtitle)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stage_header(title: str, subtitle: str, status: str, tone: str = "cyan") -> None:
    badge = f'<span class="status {tone}">{html.escape(status)}</span>' if status else ""
    st.markdown(
        f"""
        <div class="stage-intro">
          <div class="stage-title-row">
            <h1>{html.escape(title)}</h1>
            {badge}
          </div>
          <div class="stage-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_card(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div><div class="section-sub">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )




def source_config_block(title: str, subtitle: str, source_items: Iterable[tuple[str, object]], config_items: Iterable[tuple[str, object]] | None = None) -> None:
    source_values = list(source_items)
    config_values = list(config_items or [])
    source_html = "".join(
        f'<div class="source-row"><span>{html.escape(str(label))}</span><strong title="{html.escape(str(value))}">{html.escape(str(value))}</strong></div>'
        for label, value in source_values
    )
    config_html = ""
    if config_values:
        chips = "".join(
            f'<div class="config-chip"><span>{html.escape(str(label))}</span><strong title="{html.escape(str(value))}">{html.escape(str(value))}</strong></div>'
            for label, value in config_values
        )
        config_html = f'<div class="source-divider"></div><div class="source-grid-title">Hyperparameters</div><div class="config-grid source-config-grid">{chips}</div>'
    st.markdown(
        f"""<div class="source-config-shell">
              <div class="section-title">{html.escape(title)}</div>
              <div class="section-sub">{html.escape(subtitle)}</div>
              <div class="source-list">{source_html}</div>
              {config_html}
            </div>""",
        unsafe_allow_html=True,
    )

def metrics(items: Iterable[tuple[str, object, str | None]]) -> None:
    values = list(items)
    cols = st.columns(len(values))
    for col, item in zip(cols, values):
        label, value, note = item
        with col:
            st.markdown(
                f"""
                <div class="metric">
                  <div class="metric-label">{html.escape(str(label))}</div>
                  <div class="metric-value" title="{html.escape(str(value))}">{html.escape(str(value))}</div>
                  <div class="metric-note">{html.escape(str(note or ''))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def config_grid(items: Iterable[tuple[str, object]]) -> None:
    values = list(items)
    if not values:
        return
    cards = "".join(
        f'<div class="config-chip"><span>{html.escape(str(label))}</span><strong title="{html.escape(str(value))}">{html.escape(str(value))}</strong></div>'
        for label, value in values
    )
    st.markdown(f'<div class="config-grid">{cards}</div>', unsafe_allow_html=True)


def progress_card(value: float, title: str, detail: str) -> None:
    value = max(0.0, min(1.0, float(value)))
    pct = round(value * 100)
    st.markdown(
        f"""
        <div class="progress-shell">
          <div class="section-title">{html.escape(title)}</div>
          <div class="progress-track"><div class="progress-fill" style="width:{pct}%"></div></div>
          <div class="progress-row"><span>{pct}%</span><span>{html.escape(detail)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def path_box(value: str, *, spacious: bool = False) -> None:
    css = "path-box spacious" if spacious else "path-box"
    st.markdown(f'<div class="{css}">{html.escape(value)}</div>', unsafe_allow_html=True)


def log_box(lines: list[str], empty_message: str = "No log output yet.", *, title: str = "Log", max_lines: int | None = 70) -> None:
    if lines:
        shown = lines if max_lines is None else lines[-max_lines:]
        content = "\n".join(shown)
    else:
        content = empty_message
    st.markdown(
        f'''<div class="log-shell">
              <div class="section-title">{html.escape(title)}</div>
              <div class="log-box">{html.escape(content)}</div>
            </div>''',
        unsafe_allow_html=True,
    )


def empty_chart(title: str, message: str = "Waiting for process data") -> None:
    st.markdown(
        f'<div class="chart-box"><div class="section-title">{html.escape(title)}</div><div class="chart-empty">{html.escape(message)}</div></div>',
        unsafe_allow_html=True,
    )


def _format_chart_value(value: float) -> str:
    return f"{value:.6f}" if abs(value) < 0.1 else f"{value:.4f}"


def simple_bars(title: str, values: list[float]) -> None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        empty_chart(title)
        return
    recent = cleaned[-10:]
    maximum = max(recent) or 1.0
    bars = "".join(
        f'<span title="{html.escape(_format_chart_value(value))}" style="height:{max(8, int(78 * value / maximum))}px"></span>'
        for value in recent
    )
    summary = (
        f'<div class="chart-summary">Latest: {_format_chart_value(recent[-1])}'
        f' - Min: {_format_chart_value(min(recent))}'
        f' - Max: {_format_chart_value(max(recent))}</div>'
    )
    st.markdown(
        f'<div class="chart-box"><div class="section-title">{html.escape(title)}</div><div class="bars">{bars}</div>{summary}</div>',
        unsafe_allow_html=True,
    )


def _plotly_layout(title: str, *, y_title: str = "Value", x_title: str = "") -> dict:
    return {
        "title": {"text": title, "font": {"size": 15, "color": "#dff3ff"}, "x": 0.02, "xanchor": "left"},
        "paper_bgcolor": "rgba(14,29,49,.86)",
        "plot_bgcolor": "rgba(4,14,27,.68)",
        "font": {"color": "#b7cce0", "size": 11},
        "margin": {"l": 55, "r": 18, "t": 48, "b": 48},
        "height": 350,
        "xaxis": {"title": x_title, "gridcolor": "rgba(145,168,190,.12)", "zerolinecolor": "rgba(145,168,190,.18)"},
        "yaxis": {"title": y_title, "gridcolor": "rgba(145,168,190,.12)", "zerolinecolor": "rgba(145,168,190,.18)"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        "hovermode": "x unified",
    }


def _plotly_config() -> dict:
    return {
        "displaylogo": False,
        "displayModeBar": False,
        "responsive": True,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "doubleClick": "reset",
        
    }

def training_loss_chart(history: Sequence[dict]) -> None:
    if not history or go is None:
        simple_bars("Train / validation loss", [item.get("val_loss") for item in history]) if history else empty_chart("Train / validation loss")
        return
    epochs = [item.get("epoch") for item in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=[item.get("train_loss") for item in history], mode="lines+markers", name="Train loss", line={"color": "#39d9ff", "width": 2}, marker={"size": 6}))
    fig.add_trace(go.Scatter(x=epochs, y=[item.get("val_loss") for item in history], mode="lines+markers", name="Validation loss", line={"color": "#9b7bff", "width": 2}, marker={"size": 6}))
    fig.update_layout(**_plotly_layout("Train / validation loss by epoch", y_title="Loss", x_title="Epoch"))
    st.plotly_chart(fig, use_container_width=True, config=_plotly_config())


def training_status_chart(history: Sequence[dict]) -> None:
    """Show the early-stopping counter as a line with a visible threshold."""
    if not history or go is None:
        empty_chart("Early-stopping patience by epoch")
        return
    epochs = [item.get("epoch") for item in history]
    counters = [int(item.get("patience_counter") or 0) for item in history]
    limits = [int(item.get("patience") or 0) for item in history]
    max_limit = max(limits or [1])
    labels = []
    new_best_x, new_best_y = [], []
    for item, epoch, counter, limit in zip(history, epochs, counters, limits):
        if item.get("status") == "new_best":
            labels.append(f"Epoch {epoch}<br>New best validation loss - patience reset to 0")
            new_best_x.append(epoch)
            new_best_y.append(counter)
        elif limit and counter >= max(1, limit - 1):
            labels.append(f"Epoch {epoch}<br>Patience: {counter} / {limit} - near threshold")
        else:
            labels.append(f"Epoch {epoch}<br>Patience: {counter} / {limit}" if limit else f"Epoch {epoch}<br>Patience: {counter}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=counters, mode="lines+markers", name="Patience counter",
        line={"color": "#9b7bff", "width": 3}, marker={"size": 7, "color": "#9b7bff"},
        customdata=labels, hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=[max_limit] * len(epochs), mode="lines", name=f"Stop threshold ({max_limit})",
        line={"color": "#f59a2d", "width": 2, "dash": "dash"},
        hovertemplate=f"Early-stopping threshold: {max_limit}<extra></extra>",
    ))
    if new_best_x:
        fig.add_trace(go.Scatter(
            x=new_best_x, y=new_best_y, mode="markers", name="New best",
            marker={"size": 10, "color": "#39d9ff", "symbol": "diamond"},
            hovertemplate="New best validation loss - counter reset<extra></extra>",
        ))
    fig.update_layout(**_plotly_layout("Early-stopping patience by epoch", y_title="Epochs without improvement", x_title="Epoch"))
    fig.update_yaxes(range=[0, max(1, max_limit) + 0.5], dtick=1)
    st.plotly_chart(fig, use_container_width=True, config=_plotly_config())

def per_dimension_mae_chart(values: Sequence[float]) -> None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned or go is None:
        simple_bars("Per-dimension MAE", cleaned) if cleaned else empty_chart("Per-dimension MAE")
        return
    dimensions = [f"Dim {index + 1}" for index in range(len(cleaned))]
    fig = go.Figure(go.Bar(x=dimensions, y=cleaned, marker_color="#9b7bff", hovertemplate="%{x}<br>MAE: %{y:.6f}<extra></extra>"))
    fig.update_layout(**_plotly_layout("Per-dimension MAE", y_title="MAE", x_title="Action dimension"))
    st.plotly_chart(fig, use_container_width=True, config=_plotly_config())


def prediction_comparison_chart(ground_truth: Sequence[float], prediction: Sequence[float], *, title: str = "Ground truth vs prediction") -> None:
    gt = [float(value) for value in ground_truth]
    pred = [float(value) for value in prediction]
    if not gt or not pred or len(gt) != len(pred) or go is None:
        empty_chart(title, "Prediction example will appear after evaluation prints an example")
        return
    dimensions = [f"Dim {index + 1}" for index in range(len(gt))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dimensions, y=gt, name="Ground truth", marker_color="#39d9ff", hovertemplate="%{x}<br>Ground truth: %{y:.6f}<extra></extra>"))
    fig.add_trace(go.Bar(x=dimensions, y=pred, name="Prediction", marker_color="#f59a2d", hovertemplate="%{x}<br>Prediction: %{y:.6f}<extra></extra>"))
    fig.update_layout(**_plotly_layout(title, y_title="Action value", x_title="Action dimension"), barmode="group")
    st.plotly_chart(fig, use_container_width=True, config=_plotly_config())
