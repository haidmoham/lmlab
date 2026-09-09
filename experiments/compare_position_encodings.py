"""Paired bigram / six-block / six-block-with-position comparison; table output only."""

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from experiments.compare_context_lengths import evaluate, evaluation_batches, training_batch
from src.dataset import load_tiny_shakespeare_tokens, split_token_stream
from src.language_models import BigramLanguageModel, StackedTransformerLanguageModel

TREATMENTS = ("bigram", "transformer", "transformer_sinusoidal")
CONTEXT = 32


class BigramLogits(BigramLanguageModel):
    """Expose the existing bigram logits to the shared experiment loop."""

    def forward(self, token_ids):
        logits, _ = super().forward(token_ids)
        return logits


def initialize_models(vocab_size, seed):
    """Match embeddings/head across all arms and block weights across transformers."""
    torch.manual_seed(seed)
    reference = StackedTransformerLanguageModel(
        vocab_size, n_layers=6, block_size=CONTEXT, position_encoding="none"
    )
    models = {
        "bigram": BigramLogits(vocab_size, n_embd=32),
        "transformer": reference,
        "transformer_sinusoidal": StackedTransformerLanguageModel(
            vocab_size, n_layers=6, block_size=CONTEXT, position_encoding="sinusoidal"
        ),
    }
    reference_state = reference.state_dict()
    for model in models.values():
        state = model.state_dict()
        for name in state:
            if not name.startswith("position_embedding_table."):
                state[name] = reference_state[name]
        model.load_state_dict(state)
    return models


def result_tables(report):
    """Return aggregate losses and paired seed differences without selecting a winner."""
    expected = {(seed, treatment) for seed in report["config"]["seeds"] for treatment in TREATMENTS}
    actual = [(run["seed"], run["treatment"]) for run in report["runs"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("expected one complete run per seed and treatment")
    if any(run["history"][-1]["step"] != report["config"]["steps"] for run in report["runs"]):
        raise ValueError("runs have not reached their planned endpoint")
    summary = [
        "| treatment | trainable parameters | train loss | validation loss | validation seed range | update seconds |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for treatment in TREATMENTS:
        runs = [run for run in report["runs"] if run["treatment"] == treatment]
        train = [run["history"][-1]["train"] for run in runs]
        validation = [run["history"][-1]["validation"] for run in runs]
        summary.append(
            f"| {treatment} | {runs[0]['trainable_parameters']:,} | {mean(train):.4f} | "
            f"{mean(validation):.4f} | {min(validation):.4f}–{max(validation):.4f} | "
            f"{mean(run['update_seconds'] for run in runs):.1f} |"
        )
    paired = [
        "| seed | bigram | transformer | transformer + sinusoidal | transformer − bigram | encoding − no encoding |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for seed in report["config"]["seeds"]:
        losses = {
            run["treatment"]: run["history"][-1]["validation"]
            for run in report["runs"]
            if run["seed"] == seed
        }
        paired.append(
            f"| {seed} | {losses['bigram']:.4f} | {losses['transformer']:.4f} | "
            f"{losses['transformer_sinusoidal']:.4f} | "
            f"{losses['transformer'] - losses['bigram']:+.4f} | "
            f"{losses['transformer_sinusoidal'] - losses['transformer']:+.4f} |"
        )
    return "\n".join(summary), "\n".join(paired)


def run(steps=10000, seeds=(42, 43, 44), output="artifacts/position-encoding-six-blocks-10k"):
    torch.set_num_threads(2)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("choose an empty output directory to preserve previous results")
    tokens, vocabulary, _ = load_tiny_shakespeare_tokens(Path("data"))
    training, validation = split_token_stream(tokens)
    streams = {"train": torch.tensor(training), "validation": torch.tensor(validation)}
    evaluation_rng = random.Random(2026)
    target_positions = {
        split: evaluation_rng.sample(range(CONTEXT, len(stream)), 1024)
        for split, stream in streams.items()
    }
    fixed_batches = {
        split: evaluation_batches(stream, target_positions[split], CONTEXT, batch_size=32)
        for split, stream in streams.items()
    }
    source_paths = [
        __file__,
        "experiments/compare_context_lengths.py",
        "src/language_models.py",
        "src/dataset.py",
        "src/tokenizer.py",
        "data/tiny_shakespeare.txt",
        "data/tiny_shakespeare_bpe.json",
    ]
    config = {
        "steps": steps,
        "seeds": list(seeds),
        "treatments": list(TREATMENTS),
        "transformer_depth": 6,
        "width": 32,
        "heads": 2,
        "context": CONTEXT,
        "tokens_per_update": 256,
        "sequences_per_update": 8,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "evaluation_every": 1000,
        "evaluation_targets_per_split": 1024,
        "evaluation_target_positions": target_positions,
        "evaluation_protocol": "same targets and preceding 32 tokens; final-position cross entropy",
        "training_protocol": "same contiguous 256 shifted targets reshaped to 8 sequences of 32",
        "initialization": "identical non-position weights; independently owned parameters",
        "training_tokens": len(training),
        "validation_tokens": len(validation),
        "vocab_size": len(vocabulary),
        "device": "cpu",
        "torch_threads": 2,
        "torch_version": torch.__version__,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(Path(path).resolve().relative_to(Path.cwd())): hashlib.sha256(
                Path(path).read_bytes()
            ).hexdigest()
            for path in source_paths
        },
    }
    report = {"config": config, "runs": []}
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    for seed in seeds:
        models = initialize_models(len(vocabulary), seed)
        optimizers = {
            name: torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            for name, model in models.items()
        }
        histories = {name: [] for name in TREATMENTS}
        update_seconds = {name: 0.0 for name in TREATMENTS}
        batch_rng = random.Random(seed)
        for step in range(steps + 1):
            if step % 1000 == 0 or step in (2500, steps):
                for name, model in models.items():
                    row = {"step": step}
                    for split, batches in fixed_batches.items():
                        row[split] = evaluate(model, batches)
                    histories[name].append(row)
                print(f"seed {seed}: all three treatments evaluated at {step} updates", flush=True)
            if step in (1000, 2500, 5000, steps):
                for name, model in models.items():
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizers[name].state_dict(),
                            "seed": seed,
                            "treatment": name,
                            "step": step,
                            "config": config,
                            "history": histories[name],
                            "batch_rng_state": batch_rng.getstate(),
                            "torch_rng_state": torch.get_rng_state(),
                        },
                        directory / f"{name}-seed-{seed}-step-{step}.pt",
                    )
            if step == steps:
                break
            start = batch_rng.randrange(len(training) - 256)
            inputs, targets = training_batch(streams["train"], start, CONTEXT)
            for name, model in models.items():
                started = time.perf_counter()
                optimizers[name].zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, len(vocabulary)), targets.flatten())
                loss.backward()
                optimizers[name].step()
                update_seconds[name] += time.perf_counter() - started
        for name, model in models.items():
            report["runs"].append(
                {
                    "seed": seed,
                    "treatment": name,
                    "history": histories[name],
                    "update_seconds": update_seconds[name],
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                }
            )
        (directory / "results.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--output", default="artifacts/position-encoding-six-blocks-10k")
    args = parser.parse_args()
    run(steps=args.steps, output=args.output)
