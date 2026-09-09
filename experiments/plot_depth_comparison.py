"""Combine compatible depth runs into inspectable notebook tables and figures."""

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

TREATMENTS = ["bigram", "transformer", "two_blocks", "4_blocks", "6_blocks", "8_blocks"]
LABELS = ["bigram", "1 block", "2 blocks", "4 blocks", "6 blocks", "8 blocks"]
COLORS = ["#64748b", "#b45309", "#15803d", "#7c3aed", "#db2777", "#0284c7"]


def load_depth_reports(root):
    """Reject mismatched controls before combining historical and new treatments."""
    directories = [root / "all-treatments-10k", root / "depth-4-6-8-10k"]
    reports = []
    for directory in directories:
        reports.append(json.loads((directory / "results.json").read_text()))
    reference_config = reports[0]["config"]
    control_names = [
        "steps",
        "seeds",
        "context",
        "batch_size",
        "width",
        "heads",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "eval_batches",
        "train_tokens",
        "validation_tokens",
        "vocab_size",
        "torch_version",
    ]
    for control_name in control_names:
        if reports[1]["config"][control_name] != reference_config[control_name]:
            raise ValueError(f"incompatible comparison control: {control_name}")
    for source_path in [
        "src/language_models.py",
        "src/dataset.py",
        "src/tokenizer.py",
        "data/tiny_shakespeare.txt",
    ]:
        original_hash = reference_config["source_sha256"][source_path]
        new_hash = reports[1]["config"]["source_sha256"][source_path]
        if original_hash != new_hash:
            raise ValueError(f"changed model or data source: {source_path}")
    combined_runs = reports[0]["runs"] + reports[1]["runs"]
    for treatment in TREATMENTS:
        treatment_runs = [run for run in combined_runs if run["model"] == treatment]
        observed_seeds = sorted(run["seed"] for run in treatment_runs)
        if observed_seeds != sorted(reference_config["seeds"]):
            raise ValueError(f"incomplete or duplicate seeds for {treatment}")
        for run in treatment_runs:
            if run["history"][-1]["step"] != reference_config["steps"]:
                raise ValueError(f"unfinished run for {treatment}")
    return combined_runs


def summarize_depths(combined_runs):
    rows = []
    for treatment, label in zip(TREATMENTS, LABELS, strict=True):
        runs = [run for run in combined_runs if run["model"] == treatment]
        validation_losses = [run["history"][-1]["validation"] for run in runs]
        rows.append(
            {
                "treatment": label,
                "parameters": runs[0]["parameters"],
                "train_loss": mean(run["history"][-1]["train"] for run in runs),
                "validation_loss": mean(validation_losses),
                "validation_min": min(validation_losses),
                "validation_max": max(validation_losses),
                "update_seconds": mean(run["update_seconds"] for run in runs),
                "seed_losses": {run["seed"]: run["history"][-1]["validation"] for run in runs},
            }
        )
    return rows


def plot_depth_comparison(combined_runs, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for treatment_index, treatment in enumerate(TREATMENTS):
        runs = [run for run in combined_runs if run["model"] == treatment]
        update_steps = np.array([row["step"] for row in runs[0]["history"]])
        color = COLORS[treatment_index]
        label = LABELS[treatment_index]
        for axis, split in [(axes[0, 0], "train"), (axes[0, 1], "validation")]:
            losses = np.array([[row[split] for row in run["history"]] for run in runs])
            axis.plot(update_steps, losses.mean(axis=0), color=color, label=label)
            axis.fill_between(
                update_steps, losses.min(axis=0), losses.max(axis=0), color=color, alpha=0.1
            )
        validation_losses = np.array(
            [[row["validation"] for row in run["history"]] for run in runs]
        )
        later_updates = update_steps >= 1000
        axes[1, 0].plot(
            update_steps[later_updates],
            validation_losses.mean(axis=0)[later_updates],
            color=color,
            label=label,
        )
        axes[1, 0].fill_between(
            update_steps[later_updates],
            validation_losses.min(axis=0)[later_updates],
            validation_losses.max(axis=0)[later_updates],
            color=color,
            alpha=0.1,
        )
        for seed_index, run in enumerate(runs):
            horizontal_offset = (seed_index - 1) * 0.12
            axes[1, 1].scatter(
                treatment_index + horizontal_offset,
                run["history"][-1]["validation"],
                color=color,
                s=30,
            )
        axes[1, 1].scatter(
            treatment_index,
            validation_losses[:, -1].mean(),
            color=color,
            marker="_",
            s=250,
            linewidths=2,
        )
    for axis in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        axis.set(xlabel="updates", ylabel="cross-entropy (nats/BPE token)")
        axis.grid(alpha=0.15)
    axes[0, 0].set_title("training windows")
    axes[0, 1].set_title("validation windows")
    axes[1, 0].set_title("validation after 1,000 updates")
    axes[1, 1].set(
        title="final validation: each seed and mean", ylabel="cross-entropy (nats/BPE token)"
    )
    axes[1, 1].set_xticks(range(len(LABELS)), LABELS, rotation=20)
    axes[0, 0].legend(ncol=2, fontsize=9)
    figure.suptitle(
        "depth comparison · 10,000 updates · three paired seeds\ncurves: mean and seed range; equal updates are not equal compute"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return Path(output_path)


def plot_treatment_text(artifact_root, seed=42, step=10000):
    """Show the same fixed-prompt checkpoint sample for each treatment in a grid."""
    import textwrap

    checkpoint_samples = {}
    for directory_name in ["all-treatments-10k", "depth-4-6-8-10k"]:
        report_path = artifact_root / directory_name / "results.json"
        report = json.loads(report_path.read_text())
        for sample in report["checkpoint_samples"]:
            if sample["seed"] == seed and sample["step"] == step:
                checkpoint_samples[sample["model"]] = sample

    reference_sample = checkpoint_samples["bigram"]
    wrapped_continuations = []
    for treatment in TREATMENTS:
        sample = checkpoint_samples[treatment]
        for setting in ["prompt", "sampling_seed", "temperature"]:
            if sample[setting] != reference_sample[setting]:
                raise ValueError(f"sample setting differs for {treatment}: {setting}")
        # Preserve generated spelling and newlines; expose any control characters.
        printable_text = "".join(
            character if character.isprintable() or character == "\n" else repr(character)[1:-1]
            for character in sample["continuation"]
        )
        wrapped_lines = []
        for line in printable_text.split("\n"):
            wrapped_lines.append(textwrap.fill(line, width=57, replace_whitespace=False))
        wrapped_continuations.append("\n".join(wrapped_lines))

    longest_panel_lines = max(text.count("\n") + 1 for text in wrapped_continuations)
    panel_height = 1.0 + longest_panel_lines * 0.19
    figure, axes = plt.subplots(3, 2, figsize=(14, 3 * panel_height + 2))
    figure.patch.set_facecolor("#faf8f3")
    for index, axis in enumerate(axes.flat):
        treatment = TREATMENTS[index]
        sample = checkpoint_samples[treatment]
        axis.set_axis_off()
        axis.text(
            0,
            1,
            f"{LABELS[index]} · {step:,} updates",
            transform=axis.transAxes,
            fontsize=14,
            fontweight="bold",
            color=COLORS[index],
            va="top",
        )
        axis.text(
            0,
            0.90,
            f"seed {seed} validation loss: {sample['validation_loss']:.4f} nats/token",
            transform=axis.transAxes,
            fontsize=10,
            color="#626262",
            va="top",
        )
        axis.text(
            0,
            0.77,
            wrapped_continuations[index],
            transform=axis.transAxes,
            fontsize=10.5,
            fontfamily="monospace",
            color="#222222",
            va="top",
            linespacing=1.25,
        )
    prompt_label = reference_sample["prompt"].replace("\n", " / ")
    figure.suptitle(
        "generated text across treatments",
        x=0.055,
        ha="left",
        y=0.98,
        fontsize=22,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.94,
        f"shared prompt: {prompt_label!r}\ntraining seed {seed} · sampling seed 123 · temperature 1 · 100 new BPE tokens per panel",
        va="top",
        fontsize=11,
        color="#444444",
    )
    figure.text(
        0.055,
        0.018,
        "verbatim continuations with visual wrapping; one preselected sample per treatment, not a quality ranking.",
        fontsize=10,
        color="#555555",
    )
    figure.subplots_adjust(left=0.055, right=0.97, top=0.86, bottom=0.06, hspace=0.30, wspace=0.12)
    output_directory = artifact_root / "depth-4-6-8-10k"
    for suffix in ["png", "pdf"]:
        figure.savefig(
            output_directory / f"treatment-text-{step}.{suffix}",
            dpi=180,
            facecolor=figure.get_facecolor(),
        )
    plt.close(figure)
    return output_directory / f"treatment-text-{step}.png"
