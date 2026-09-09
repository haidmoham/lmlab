"""Render context experiment measurements without interpreting the results."""

import json
import textwrap
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONTEXTS = (8, 32, 128, 256)
COLORS = ("#92b9e0", "#a5cea7", "#b9a3df", "#82cbd1")


from experiments.notebook_theme import notebook_figure


def read_report(directory):
    report = json.loads((Path(directory) / "results.json").read_text())
    expected_runs = len(CONTEXTS) * len(report["config"]["seeds"])
    if len(report["runs"]) != expected_runs:
        raise ValueError("training runs are not complete")
    for run in report["runs"]:
        if run["history"][-1]["step"] != report["config"]["steps"]:
            raise ValueError("a training run has not reached its planned endpoint")
    return report


def summary_table(report):
    lines = [
        "| training context | sequences/update | parameters | train loss | validation loss | seed range | update seconds |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for context in CONTEXTS:
        runs = [run for run in report["runs"] if run["context"] == context]
        train_losses = [run["history"][-1]["train"] for run in runs]
        validation_losses = [run["history"][-1]["validation"] for run in runs]
        times = [run["update_seconds"] for run in runs]
        lines.append(
            f"| {context} | {256 // context} | {runs[0]['parameters']:,} | "
            f"{mean(train_losses):.4f} | {mean(validation_losses):.4f} | "
            f"{min(validation_losses):.4f}–{max(validation_losses):.4f} | {mean(times):.1f} |"
        )
    return "\n".join(lines)


def cross_evaluation_table(report, seed=None):
    lines = [
        "| trained context / evaluated context | 8 | 32 | 128 | 256 |",
        "|---|---:|---:|---:|---:|",
    ]
    for training_context in CONTEXTS:
        cells = [str(training_context)]
        for evaluation_context in CONTEXTS:
            if evaluation_context > training_context:
                cells.append("—")
                continue
            values = []
            for row in report["cross_evaluation"]:
                matches_contexts = (
                    row["training_context"] == training_context
                    and row["evaluation_context"] == evaluation_context
                )
                matches_seed = seed is None or row["seed"] == seed
                if matches_contexts and matches_seed:
                    values.append(row["validation"])
            expected_count = len(report["config"]["seeds"]) if seed is None else 1
            if len(values) != expected_count:
                raise ValueError("cross-evaluation is incomplete or duplicated")
            cells.append(f"{mean(values):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


@notebook_figure
def plot_measurements(report, directory):
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for context, color in zip(CONTEXTS, COLORS, strict=True):
        runs = [run for run in report["runs"] if run["context"] == context]
        steps = [row["step"] for row in runs[0]["history"]]
        for axis, split in [(axes[0], "train"), (axes[1], "validation")]:
            losses = np.array([[row[split] for row in run["history"]] for run in runs])
            axis.plot(steps, losses.mean(axis=0), color=color, label=f"context {context}")
            axis.fill_between(
                steps, losses.min(axis=0), losses.max(axis=0), color=color, alpha=0.15
            )
    for axis, title in [
        (axes[0], "matched training targets"),
        (axes[1], "matched held-out targets"),
    ]:
        axis.set(title=title, xlabel="updates", ylabel="cross-entropy (nats/BPE token)")
        axis.grid(alpha=0.15)
    axes[0].legend(fontsize=9)
    matrix = np.full((4, 4), np.nan)
    for row_index, training_context in enumerate(CONTEXTS):
        for column_index, evaluation_context in enumerate(CONTEXTS):
            if evaluation_context > training_context:
                continue
            values = [
                row["validation"]
                for row in report["cross_evaluation"]
                if row["training_context"] == training_context
                and row["evaluation_context"] == evaluation_context
            ]
            matrix[row_index, column_index] = mean(values)
    axes[2].imshow(np.zeros((4, 4)), cmap=matplotlib.colors.ListedColormap(["#222735"]))
    for row_index in range(4):
        for column_index in range(4):
            value = matrix[row_index, column_index]
            label = "—" if np.isnan(value) else f"{value:.4f}"
            axes[2].text(column_index, row_index, label, ha="center", va="center", fontsize=11)
    axes[2].set_xticks(range(4), CONTEXTS)
    axes[2].set_yticks(range(4), CONTEXTS)
    axes[2].set(
        title="final cross-evaluation: validation means",
        xlabel="evaluation context",
        ylabel="training context",
    )
    figure.suptitle(
        "context comparison · four blocks · 10,000 updates · three seeds\ncurves: mean and seed range; same target positions"
    )
    figure.tight_layout()
    output_path = Path(directory) / "context-measurements.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


@notebook_figure
def plot_text_panels(report, directory, frozen_long_model=False):
    """Use preselected seed 42 and fixed sampling; no sample selection by quality."""
    samples = []
    for context in CONTEXTS:
        if frozen_long_model:
            sample = next(
                sample
                for sample in report["cross_samples"]
                if sample["training_context"] == 256
                and sample["evaluation_context"] == context
                and sample["seed"] == 42
            )
        else:
            sample = next(
                sample
                for sample in report["checkpoint_samples"]
                if sample["context"] == context
                and sample["seed"] == 42
                and sample["step"] == report["config"]["steps"]
            )
        samples.append(sample)
    reference = samples[0]
    wrapped_texts = []
    for sample in samples:
        for setting in ["prompt_ids", "sampling_seed", "temperature"]:
            if sample[setting] != reference[setting]:
                raise ValueError(f"inconsistent text-panel setting: {setting}")
        if len(sample["continuation_ids"]) != 100:
            raise ValueError("expected exactly 100 generated tokens")
        printable_text = "".join(
            character if character.isprintable() or character == "\n" else repr(character)[1:-1]
            for character in sample["continuation"]
        )
        lines = []
        for line in printable_text.split("\n"):
            lines.append(textwrap.fill(line, width=57, replace_whitespace=False))
        wrapped_texts.append("\n".join(lines))
    longest_panel = max(text.count("\n") + 1 for text in wrapped_texts)
    panel_height = 1.2 + longest_panel * 0.20
    figure, axes = plt.subplots(2, 2, figsize=(14, 2 * panel_height + 2))
    figure.patch.set_facecolor("#181b25")
    for index, axis in enumerate(axes.flat):
        context = CONTEXTS[index]
        training_context = 256 if frozen_long_model else context
        axis.set_axis_off()
        axis.text(
            0,
            1,
            f"trained {training_context} · evaluated {context}",
            transform=axis.transAxes,
            fontsize=14,
            color=COLORS[index],
            fontweight="bold",
            va="top",
        )
        loss_key = "validation" if frozen_long_model else "validation_loss"
        axis.text(
            0,
            0.88,
            f"seed 42 validation loss: {samples[index][loss_key]:.4f} nats/token",
            transform=axis.transAxes,
            fontsize=10,
            color="#b9bfce",
            va="top",
        )
        axis.text(
            0,
            0.73,
            wrapped_texts[index],
            transform=axis.transAxes,
            fontsize=10.5,
            fontfamily="monospace",
            color="#e5e7ef",
            va="top",
            linespacing=1.25,
        )
    title = (
        "one frozen model, different available history"
        if frozen_long_model
        else "context treatments at 10,000 updates"
    )
    figure.suptitle(title, x=0.055, ha="left", y=0.98, fontsize=22, fontweight="bold")
    prompt_suffix = reference["prompt"][-70:].replace("\n", " / ")
    figure.text(
        0.055,
        0.935,
        f"same 256-token prompt, truncated to evaluation context · suffix: {prompt_suffix!r}\n"
        "training seed 42 · sampling seed 123 · temperature 1 · 100 new BPE tokens per panel",
        fontsize=10.5,
        color="#b9bfce",
        va="top",
    )
    figure.text(
        0.055,
        0.025,
        "verbatim continuations with visual wrapping; fixed samples, not a quality ranking.",
        fontsize=10,
        color="#b9bfce",
    )
    figure.subplots_adjust(left=0.055, right=0.97, top=0.82, bottom=0.09, hspace=0.35, wspace=0.12)
    filename = "frozen-model-text" if frozen_long_model else "context-treatment-text"
    for suffix in ["png", "pdf"]:
        figure.savefig(
            Path(directory) / f"{filename}.{suffix}", dpi=180, facecolor=figure.get_facecolor()
        )
    plt.close(figure)
    return Path(directory) / f"{filename}.png"
