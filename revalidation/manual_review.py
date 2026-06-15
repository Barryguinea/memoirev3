"""Generate visual panels used for the human review of synthetic events."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

from core.io import COW, TIME, load_csv
from core.early_warning import EarlyWarningConfig
from core.pipeline import run_pipeline_one_cow
from revalidation.campaign import final_params, inject_events_for_cow


ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / "data/revalidation/events_primary.csv"
SAMPLE_PATH = ROOT / "data/revalidation/manual_review_sample.csv"
OUT = ROOT / "data/revalidation/manual_review_figures"


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _minutes(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds() / 60


def _rolling_12h(series: pd.Series) -> pd.Series:
    return series.rolling(48, min_periods=12).sum()


def _plot_event(
    event: pd.Series,
    clean_raw: pd.DataFrame,
    injected_raw: pd.DataFrame,
    prediction: pd.DataFrame,
    clean_prediction: pd.DataFrame,
    target: Path,
) -> None:
    start = pd.Timestamp(event["start"])
    end = pd.Timestamp(event["end"])
    left = start - pd.Timedelta(hours=24)
    right = end + pd.Timedelta(hours=24)
    raw = clean_raw[(clean_raw[TIME] >= left) & (clean_raw[TIME] <= right)].copy()
    injected = injected_raw[
        (injected_raw[TIME] >= left) & (injected_raw[TIME] <= right)
    ].copy()
    pred = prediction[(prediction[TIME] >= left) & (prediction[TIME] <= right)].copy()
    clean_pred = clean_prediction[
        (clean_prediction[TIME] >= left) & (clean_prediction[TIME] <= right)
    ].copy()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for axis in axes:
        axis.axvspan(start, end, color="#d95f5f", alpha=0.12, label="injection")
        axis.grid(alpha=0.18)

    axes[0].plot(
        raw[TIME],
        _rolling_12h(_series(raw, "Steps")),
        color="#777777",
        alpha=0.75,
        label="pas origine, total 12 h",
    )
    axes[0].plot(
        injected[TIME],
        _rolling_12h(_series(injected, "Steps")),
        color="#235789",
        linewidth=1.2,
        label="pas injectés, total 12 h",
    )
    axes[0].set_ylabel("Pas / 12 h")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)

    axes[1].plot(
        raw[TIME],
        _rolling_12h(_series(raw, "Motion Index")),
        color="#777777",
        alpha=0.65,
        label="MI origine, total 12 h",
    )
    axes[1].plot(
        injected[TIME],
        _rolling_12h(_series(injected, "Motion Index")),
        color="#238b45",
        linewidth=1.2,
        label="MI injecté, total 12 h",
    )
    axes[1].plot(
        injected[TIME],
        _rolling_12h(_series(injected, "Transitions")) * 20,
        color="#cb7c22",
        linewidth=1.0,
        label="transitions, total 12 h x20",
    )
    axes[1].set_ylabel("Mouvement")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)

    lying_minutes = _minutes(injected["Lying Time"])
    raw_lying_minutes = _minutes(raw["Lying Time"])
    axes[2].plot(
        raw[TIME],
        _rolling_12h(raw_lying_minutes),
        color="#777777",
        alpha=0.75,
        label="couché origine, total 12 h",
    )
    axes[2].plot(
        injected[TIME],
        _rolling_12h(lying_minutes),
        color="#6a51a3",
        label="couché injecté, total 12 h",
    )
    axes[2].set_ylabel("Couché min / 12 h")
    axes[2].legend(loc="upper right", ncol=3, fontsize=8)

    axes[3].plot(
        pred[TIME],
        _series(pred, "behavioral_warning_score"),
        color="#1b9e77",
        label="score",
    )
    cusum_axis = axes[3].twinx()
    cusum_axis.plot(
        pred[TIME],
        _series(pred, "behavioral_warning_cusum"),
        color="#7570b3",
        alpha=0.75,
        label="CUSUM",
    )
    cusum_axis.axhline(1.2, color="#7570b3", linestyle="--", linewidth=0.8)
    cusum_axis.set_ylabel("CUSUM")
    episode = _series(pred, "behavioral_warning_episode").fillna(0).astype(int)
    clean_flags = (
        clean_pred.drop_duplicates(subset=[TIME], keep="last")
        .set_index(TIME)["behavioral_warning_episode"]
    )
    reference_episode = (
        pred[TIME].map(pd.to_numeric(clean_flags, errors="coerce")).fillna(0).astype(int)
    )
    attributable_episode = ((episode == 1) & (reference_episode != 1)).astype(int)
    axes[3].fill_between(
        pred[TIME],
        0,
        reference_episode,
        step="mid",
        color="#777777",
        alpha=0.14,
        label="épisode de référence",
    )
    axes[3].fill_between(
        pred[TIME],
        0,
        attributable_episode,
        step="mid",
        color="#e7298a",
        alpha=0.24,
        label="intervalle attribuable",
    )
    starts = pred[_series(pred, "behavioral_warning_start").fillna(0).astype(int) == 1]
    for timestamp in starts[TIME]:
        axes[3].axvline(timestamp, color="#e7298a", linewidth=1.0, alpha=0.85)
    axes[3].set_ylabel("Alerte")
    handles, labels = axes[3].get_legend_handles_labels()
    handles2, labels2 = cusum_axis.get_legend_handles_labels()
    axes[3].legend(handles + handles2, labels + labels2, loc="upper right", ncol=4, fontsize=8)

    fig.suptitle(
        f"{event['event_id']} | attendu={int(event['expected_detected'])} | "
        f"nouveau début={int(event['detected_any_overlap'])} | "
        f"IoU={float(event['best_iou']):.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(target, dpi=140)
    plt.close(fig)


def _contact_sheet(paths: list[Path], target: Path) -> None:
    thumbs: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((700, 500))
        canvas = Image.new("RGB", (720, 540), "white")
        canvas.paste(image, ((720 - image.width) // 2, 28))
        ImageDraw.Draw(canvas).text((12, 8), path.stem, fill="black")
        thumbs.append(canvas)
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 720, rows * 540), "white")
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % cols) * 720, (index // cols) * 540))
    sheet.save(target)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old_figure in OUT.glob("*.png"):
        old_figure.unlink()
    params = final_params()
    protocol = json.loads(
        (ROOT / "data/revalidation/protocol_configuration.json").read_text(encoding="utf-8")
    )
    warning_config = EarlyWarningConfig(**protocol["early_warning"])
    raw = load_csv(ROOT / "data/brut.csv")
    raw[COW] = raw[COW].astype(str)
    events = pd.read_csv(EVENTS_PATH)
    sample = pd.read_csv(SAMPLE_PATH)[["event_id"]].merge(events, on="event_id", how="left")
    paths: list[Path] = []
    for event in sample.to_dict(orient="records"):
        event = pd.Series(event)
        cow = str(event["cow"])
        raw_cow = raw[raw[COW] == cow].copy()
        injected, regenerated = inject_events_for_cow(
            raw_cow,
            cow=cow,
            scenario=str(event["scenario"]),
            seed=int(event["seed"]),
            interval=str(params["interval"]),
            persist_hours=int(params["persist_hours"]),
            baseline_ratio=float(params["baseline_ratio"]),
            window_baseline=int(params["window_baseline"]),
            coverage_min_pct=float(params["coverage_min_pct"]),
            heldout_start=pd.Timestamp(event["heldout_start"]),
            schedule_rotation=int(event["schedule_rotation"]),
        )
        if regenerated.empty or regenerated.iloc[0]["event_id"] != event["event_id"]:
            raise RuntimeError(f"Impossible de régénérer {event['event_id']}")
        clean_prediction = run_pipeline_one_cow(
            raw_cow,
            cow,
            **params,
            warning_config=warning_config,
        )
        prediction = run_pipeline_one_cow(
            injected,
            cow,
            **params,
            warning_config=warning_config,
        )
        path = OUT / f"{event['event_id']}.png"
        _plot_event(
            event,
            raw_cow,
            injected,
            prediction,
            clean_prediction,
            path,
        )
        paths.append(path)
    _contact_sheet(paths, OUT / "manual_review_contact_sheet.png")
    print(f"Generated {len(paths)} review panels in {OUT}")


if __name__ == "__main__":
    main()
