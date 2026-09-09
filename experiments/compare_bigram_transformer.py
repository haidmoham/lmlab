"""Paired equal-update experiment; run with python -m experiments.compare_bigram_transformer."""

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import matplotlib
import torch
from torch.nn import functional as F

from src.dataset import load_tiny_shakespeare_tokens, split_token_stream
from src.language_models import (
    BigramLanguageModel,
    SingleTransformerLanguageModel,
    StackedTransformerLanguageModel,
)
from src.tokenizer import bpe_decode

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def logits_for(model, ids):
    result = model(ids)
    return result[0] if isinstance(result, tuple) else result


def sample_batch(stream, rng, batch_size=32, context=8):
    starts = [rng.randrange(len(stream) - context) for _ in range(batch_size)]
    return (
        torch.tensor([stream[i : i + context] for i in starts]),
        torch.tensor([stream[i + 1 : i + context + 1] for i in starts]),
    )


def loss_on(model, batch):
    x, y = batch
    logits = logits_for(model, x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    value = sum(loss_on(model, batch).item() for batch in batches) / len(batches)
    model.train()
    return value


@torch.no_grad()
def generate(model, prompt, count=100):
    model.eval()
    ids = torch.tensor([prompt])
    generator = torch.Generator().manual_seed(123)
    for _ in range(count):
        probabilities = logits_for(model, ids[:, -8:])[:, -1].softmax(-1)
        ids = torch.cat((ids, torch.multinomial(probabilities, 1, generator=generator)), dim=1)
    model.train()
    return ids[0].tolist()


def run(
    steps=1000, seeds=(42, 43, 44), output="artifacts/bigram-vs-transformer", include_stack=False
):
    torch.set_num_threads(2)
    output = Path(output)
    if (output / "results.json").exists():
        raise FileExistsError(f"results already exist in {output}; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    tokens, vocab, _ = load_tiny_shakespeare_tokens(Path("data"))
    train, validation = split_token_stream(tokens)
    # fixed evaluation windows shared across models, seeds, and checkpoints.
    eval_rng = random.Random(2026)
    fixed = {
        name: [sample_batch(stream, eval_rng) for _ in range(32)]
        for name, stream in [("train", train), ("validation", validation)]
    }
    report = {
        "config": {
            "include_stack": include_stack,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_sha256": {
                path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for path in [
                    "src/language_models.py",
                    "experiments/compare_bigram_transformer.py",
                    "src/dataset.py",
                    "src/tokenizer.py",
                    "data/tiny_shakespeare.txt",
                ]
            },
            "torch_version": torch.__version__,
            "steps": steps,
            "seeds": list(seeds),
            "context": 8,
            "batch_size": 32,
            "width": 32,
            "heads": 2,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "eval_batches": 32,
            "train_tokens": len(train),
            "validation_tokens": len(validation),
            "vocab_size": len(vocab),
        },
        "runs": [],
        "checkpoint_samples": [],
    }
    for seed in seeds:
        torch.manual_seed(seed)
        bigram = BigramLanguageModel(len(vocab), 32)
        transformer = SingleTransformerLanguageModel(len(vocab))
        # paired starting values for shared components, not shared parameter objects.
        transformer.token_embedding_table.load_state_dict(bigram.token_embedding_table.state_dict())
        transformer.lm_head.load_state_dict(bigram.lm_head.state_dict())
        models = {"bigram": bigram, "transformer": transformer}
        if include_stack:
            stacked = StackedTransformerLanguageModel(len(vocab), n_layers=2)
            # Copy values, never parameter objects: treatments train independently.
            for component in (
                "token_embedding_table",
                "position_embedding_table",
                "final_norm",
                "lm_head",
            ):
                source_component = getattr(transformer, component)
                target_component = getattr(stacked, component)
                target_component.load_state_dict(source_component.state_dict())
            stacked.stack.blocks[0].load_state_dict(transformer.block.state_dict())
            models["two_blocks"] = stacked
        optimizers = {
            name: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            for name, model in models.items()
        }
        histories = {name: [] for name in models}
        elapsed = {name: 0.0 for name in models}
        rng = random.Random(seed)
        for step in range(steps + 1):
            if step == 0 or step % 100 == 0 or step == steps:
                for name, model in models.items():
                    row = {
                        "step": step,
                        **{split: evaluate(model, batches) for split, batches in fixed.items()},
                    }
                    histories[name].append(row)
                    print(f"seed {seed} {name} {row}", flush=True)
            if step in (1000, 2500, 5000, 10000):
                for name, model in models.items():
                    ids = generate(model, train[:8])
                    report["checkpoint_samples"].append(
                        {
                            "seed": seed,
                            "model": name,
                            "step": step,
                            "prompt": bpe_decode(train[:8], vocab, errors="replace"),
                            "continuation": bpe_decode(ids[8:], vocab, errors="replace"),
                            "token_ids": ids,
                            "sampling_seed": 123,
                            "temperature": 1.0,
                            "validation_loss": histories[name][-1]["validation"],
                        }
                    )
                    torch.save(model.state_dict(), output / f"{name}-{seed}-step-{step}.pt")
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizers[name].state_dict(),
                            "step": step,
                            "seed": seed,
                            "batch_rng_state": rng.getstate(),
                            "torch_rng_state": torch.get_rng_state(),
                            "config": report["config"],
                            "history": histories[name],
                        },
                        output / f"{name}-{seed}-step-{step}-checkpoint.pt",
                    )
            if step == steps:
                break
            batch = sample_batch(train, rng)
            for name, model in models.items():
                start = time.perf_counter()
                optimizers[name].zero_grad(set_to_none=True)
                loss_on(model, batch).backward()
                optimizers[name].step()
                elapsed[name] += time.perf_counter() - start
        for name, model in models.items():
            report["runs"].append(
                {
                    "seed": seed,
                    "model": name,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "update_seconds": elapsed[name],
                    "history": histories[name],
                    "sample": bpe_decode(generate(model, train[:8]), vocab, errors="replace"),
                }
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizers[name].state_dict(),
                    "step": steps,
                    "seed": seed,
                    "batch_rng_state": rng.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "config": report["config"],
                    "history": histories[name],
                },
                output / f"{name}-{seed}-checkpoint.pt",
            )
            # retain a weights-only export for the notebook's existing generation cell.
            torch.save(model.state_dict(), output / f"{name}-{seed}.pt")
        (output / "results.json").write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, split in zip(axes, ("train", "validation")):
        colors = {"bigram": "#526477", "transformer": "#ac4827"}
        if include_stack:
            colors["two_blocks"] = "#31785c"
        for name, color in colors.items():
            runs = [r for r in report["runs"] if r["model"] == name]
            values = torch.tensor([[h[split] for h in r["history"]] for r in runs])
            x = [h["step"] for h in runs[0]["history"]]
            ax.plot(x, values.mean(0), label=name, color=color)
            ax.fill_between(x, values.min(0).values, values.max(0).values, alpha=0.15, color=color)
        ax.set(
            title=f"fixed {split} windows",
            xlabel="updates",
            ylabel="cross-entropy (nats/BPE token)",
        )
        ax.legend()
    fig.suptitle("matched treatment comparison: mean and seed range")
    fig.tight_layout()
    fig.savefig(output / "loss.png", dpi=160)
    plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", default="artifacts/bigram-vs-transformer")
    parser.add_argument("--include-stack", action="store_true")
    args = parser.parse_args()
    run(steps=args.steps, output=args.output, include_stack=args.include_stack)
