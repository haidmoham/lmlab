"""Generate matched checkpoint continuations and render a dark three-panel text figure."""

import json
import textwrap
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.compare_context_lengths import generate_sample
from experiments.compare_position_encodings import TREATMENTS, initialize_models, result_tables
from src.dataset import load_tiny_shakespeare_tokens


def plot_position_samples(directory, seed=42):
    directory = Path(directory)
    report = json.loads((directory / "results.json").read_text())
    result_tables(report)  # Require every planned training run to be complete.
    config = report["config"]
    torch.set_num_threads(2)
    data_directory = Path(__file__).resolve().parents[1] / "data"
    tokens, vocabulary, _ = load_tiny_shakespeare_tokens(data_directory)
    prompt_ids = tokens[: config["context"]]
    samples = []
    with torch.random.fork_rng(devices=[]):
        models = initialize_models(len(vocabulary), seed)
        for treatment in TREATMENTS:
            checkpoint_path = directory / f"{treatment}-seed-{seed}-step-{config['steps']}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = models[treatment]
            model.load_state_dict(checkpoint["model"])
            sample = generate_sample(model, prompt_ids, config["context"], vocabulary)
            sample.update(
                {
                    "treatment": treatment,
                    "training_seed": seed,
                    "step": config["steps"],
                    "context": config["context"],
                    "checkpoint": checkpoint_path.name,
                    "validation_loss": checkpoint["history"][-1]["validation"],
                }
            )
            assert len(sample["continuation_ids"]) == 100
            samples.append(sample)
    (directory / "text-samples.json").write_text(json.dumps(samples, indent=2))

    wrapped_texts = []
    for sample in samples:
        printable = "".join(
            character if character.isprintable() or character == "\n" else repr(character)[1:-1]
            for character in sample["continuation"]
        )
        wrapped_texts.append(
            "\n".join(
                textwrap.fill(line, width=43, replace_whitespace=False)
                for line in printable.split("\n")
            )
        )
    longest_lines = max(text.count("\n") + 1 for text in wrapped_texts)
    figure_height = 2.8 + longest_lines * 0.22
    figure, axes = plt.subplots(1, 3, figsize=(17, figure_height))
    figure.patch.set_facecolor("#181b25")
    labels = ("bigram", "six blocks · no encoding", "six blocks · sinusoidal")
    colors = ("#d7b887", "#92b9e0", "#a5cea7")
    for index, axis in enumerate(axes):
        axis.set_axis_off()
        axis.text(0, 1, labels[index], fontsize=15, weight="bold", color=colors[index], va="top")
        axis.text(
            0,
            0.88,
            f"seed {seed} validation: {samples[index]['validation_loss']:.4f}",
            fontsize=10,
            color="#b9bfce",
            va="top",
        )
        axis.text(
            0,
            0.74,
            wrapped_texts[index],
            fontsize=11,
            fontfamily="monospace",
            color="#e5e7ef",
            va="top",
            linespacing=1.3,
        )
    figure.text(
        0.04,
        0.96,
        "same prompt, three treatments",
        fontsize=22,
        weight="bold",
        color="#e5e7ef",
        va="top",
    )
    prompt = samples[0]["prompt"].replace("\n", " / ")
    figure.text(
        0.04,
        0.885,
        f"prompt: {prompt!r}\n{config['steps']:,} updates · training seed {seed} · sampling seed 123 · temperature 1 · 100 new BPE tokens",
        fontsize=10,
        color="#b9bfce",
        va="top",
    )
    figure.text(
        0.04,
        0.035,
        "verbatim continuations, visually wrapped · transformer context: 32 tokens; bigram uses the last token · losses: nats/BPE token",
        fontsize=9,
        color="#b9bfce",
    )
    figure.subplots_adjust(left=0.04, right=0.98, top=0.73, bottom=0.11, wspace=0.14)
    path = directory / "position-treatment-text.png"
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    return path
