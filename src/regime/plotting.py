import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.utils.paths import FIGURES_DIR

os.makedirs(FIGURES_DIR / "regime/", exist_ok=True)

REGIME_COLOURS = {
    0: "tab:blue",
    1: "tab:green",
    2: "tab:red",
}

REGIME_NAME_MAP = {
    0: "Neutral",
    1: "Risk-On",
    2: "Risk-Off"
}


def _ensure_datetime(
    df: pd.DataFrame,
    date_col: str = "date"
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out


def _contiguous_regime_segments(
    df: pd.DataFrame,
    regime_col: str = "regime"
) -> list[tuple]:
    """
    Convert a regime time series into contiguous segments.

    returns
    -------
    segments: list of tuples
        Each tuple is (start_date, end_date, regime_label).
    """
    out = df.copy().reset_index(drop=True)
    segments = []

    if out.empty:
        return segments

    start_idx = 0
    current_regime = out.loc[0, regime_col]

    for i in range(1, len(out)):
        if out.loc[i, regime_col] != current_regime:
            start_date = out.loc[start_idx, "date"]
            end_date = out.loc[i-1, "date"]
            segments.append((start_date, end_date, current_regime))

    # final segment
    segments.append((
        out.loc[start_idx, "date"],
        out.loc[len(out) - 1, "date"],
        current_regime
    ))

    return segments


def plot_regime_timeline(
    regime_df: pd.DataFrame,
    model_name: str = "HMM",
    save_path=None,
    figsize=(14, 3),
):
    """
    Plot regime labels through time as a coloured timeline.
    """
    df = _ensure_datetime(regime_df)

    fig, ax = plt.subplots(figsize=figsize)

    segments = _contiguous_regime_segments(df, regime_col="regime")

    for start_date, end_date, regime in segments:
        ax.axvspan(
            start_date,
            end_date,
            alpha=0.7,
            color=REGIME_COLOURS.get(regime, "gray"),
            label=REGIME_NAME_MAP.get(regime, f"Regime {regime}")
        )

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", ncol=min(3, len(unique)))

    ax.set_title(f"{model_name} Regime Timeline")
    ax.set_xlabel("Date")
    ax.set_yticks([])
    ax.set_ylabel("")

    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"regime/{model_name}_timeline.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_regime_probabilities(
        regime_df: pd.DataFrame,
        model_name: str="HMM",
        save_path: str | None=None,
        figsize: tuple=(14, 6)
) -> None:
    """
    Plot posterior regime probabilities over time
    Expects columns like prob_regime_0, prob_regime_1, ...
    """
    df = _ensure_datetime(regime_df)
    prob_cols = [c for c in df.columns if c.startswith("prob_regime_")]
    if len(prob_cols) == 0:
        raise ValueError("No probability columns found. Expected columns like prob_regime_0, prob_regime_1, ...")

    fig, ax = plt.subplots(figsize=figsize)
    for col in prob_cols:
        regime = int(col.split("_")[-1])
        ax.plot(
            df["date"],
            df[col].rolling(3, min_periods=1).mean(),
            label=REGIME_NAME_MAP.get(regime, f"Regime {regime}"),
            color=REGIME_COLOURS.get(regime, None),
            linewidth=1.8,
        )

    ax.set_title(f"{model_name} Posterior Regime Probabilities")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"regime/{model_name.lower()}_regime_probabilities.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_growth_with_regime_shading(
    regime_df: pd.DataFrame,
    value_col: str = "growth_proxy_ret",
    model_name: str = "HMM",
    save_path: str | None=None,
    figsize: tuple=(14, 6)
) -> None:
    """
    Plot cumulative growth of $100 for a return series, with background shading by regime.
    """
    df = _ensure_datetime(regime_df)
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in regime_df")
    temp = df[["date", "regime", value_col]].copy()

    # For cumulative wealth plotting, fill missing initial values with 0 return
    temp[value_col] = temp[value_col].fillna(0.0)
    temp["growth_of_100"] = (1 + temp[value_col]).cumprod() * 100

    fig, ax = plt.subplots(figsize=figsize)

    # Add background regime shading
    segments = _contiguous_regime_segments(temp, regime_col="regime")
    for start_date, end_date, regime in segments:
        ax.axvspan(
            start_date,
            end_date,
            alpha=0.15,
            color=REGIME_COLOURS.get(regime, "gray")
        )

    # Plot cumulative path
    ax.plot(
        temp["date"],
        temp["growth_of_100"],
        color="black",
        linewidth=2.0,
        label="Growth Proxy ($100)"
    )

    # Build custom legend
    legend_items = [Patch(
        facecolor=REGIME_COLOURS[k],
        alpha=0.25,
        label=REGIME_NAME_MAP[k]
    ) for k in sorted(REGIME_COLOURS.keys())]
    line_handle = plt.Line2D([0], [0], color="black", linewidth=2.0, label="Growth Proxy ($100)")
    ax.legend(handles=[line_handle] + legend_items, loc="upper right")

    ax.set_title(f"{model_name} Regimes over Growth Proxy Cumulative Path")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Value")

    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"regime/{model_name.lower()}_growth_with_regime_shading.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_regime_strip(
    regime_df, model_name="HMM",
    save_path=None,
    figsize=(14, 2.5)
) -> None:
    df = regime_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        df["date"],
        np.zeros(len(df)),
        c=df["regime"].map(REGIME_COLOURS),
        marker="s",
        s=60
    )

    ax.set_title(f"{model_name} Regime Strip")
    ax.set_xlabel("Date")
    ax.set_yticks([])

    from matplotlib.patches import Patch
    handles = [
        Patch(color=REGIME_COLOURS[k], label=REGIME_NAME_MAP[k])
        for k in sorted(REGIME_COLOURS.keys())
    ]
    ax.legend(handles=handles, loc="upper right", ncol=3)

    plt.tight_layout()
    if save_path is None:
        save_path = FIGURES_DIR / f"regime/{model_name.lower()}_regime_strip.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()