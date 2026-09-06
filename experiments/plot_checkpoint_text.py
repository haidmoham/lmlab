"""Render fixed-prompt continuations across training milestones."""

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_checkpoint_text(directory="artifacts/bigram-vs-transformer-milestones", seed=42):
    directory = Path(directory)
    report = json.loads((directory / "results.json").read_text())
    samples = {
        (s["step"], s["model"]): s for s in report["checkpoint_samples"] if s["seed"] == seed
    }
    milestones = (1000, 2500, 5000, 10000)
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.patch.set_facecolor("#faf8f3")
    for row, step in enumerate(milestones):
        for col, (name, color) in enumerate((("bigram", "#526477"), ("transformer", "#ac4827"))):
            ax = axes[row, col]
            ax.set_facecolor("#faf8f3")
            sample = samples[step, name]
            # show control bytes as escapes; retain generated spelling and line breaks.
            safe_text = "".join(
                c if c.isprintable() or c == "\n" else repr(c)[1:-1] for c in sample["continuation"]
            )
            wrapped = "\n".join(
                textwrap.fill(line, width=58, replace_whitespace=False)
                for line in safe_text.split("\n")
            )
            ax.text(
                0,
                0.98,
                f"{step:,} updates · {name}",
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                color=color,
                va="top",
            )
            ax.text(
                0,
                0.86,
                f"validation loss {sample['validation_loss']:.3f} nats/token",
                transform=ax.transAxes,
                fontsize=10,
                color="#626262",
                va="top",
            )
            ax.text(
                0,
                0.72,
                wrapped,
                transform=ax.transAxes,
                fontsize=10.5,
                fontfamily="monospace",
                color="#222222",
                va="top",
                linespacing=1.25,
            )
            ax.set_axis_off()
    prompt = samples[1000, "bigram"]["prompt"].replace("\n", " / ")
    fig.suptitle(
        "how generated text changes with training",
        x=0.055,
        ha="left",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.055,
        0.94,
        f"shared prompt: {prompt!r}\n"
        f"training seed {seed} · sampling seed 123 · temperature 1 · 100 new BPE tokens per panel",
        fontsize=11,
        color="#444444",
        va="top",
    )
    fig.text(
        0.055,
        0.018,
        "verbatim continuations with visual wrapping; one sample per checkpoint, "
        "not a quality ranking. lower validation loss is better.",
        fontsize=10,
        color="#555555",
    )
    fig.subplots_adjust(left=0.055, right=0.97, top=0.875, bottom=0.045, hspace=0.22, wspace=0.10)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"text-milestones.{suffix}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return directory / "text-milestones.png"


if __name__ == "__main__":
    plot_checkpoint_text()
