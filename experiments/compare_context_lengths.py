"""Compare context lengths with identical target tokens and matched token budgets."""

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from src.dataset import load_tiny_shakespeare_tokens, split_token_stream
from src.language_models import StackedTransformerLanguageModel
from src.tokenizer import bpe_decode

CONTEXT_LENGTHS = (8, 32, 128, 256)
TOKENS_PER_UPDATE = 256


def training_batch(token_stream, start, context):
    """Split the same contiguous 256 prediction targets at different boundaries."""
    inputs = token_stream[start : start + TOKENS_PER_UPDATE]
    targets = token_stream[start + 1 : start + TOKENS_PER_UPDATE + 1]
    return inputs.reshape(-1, context), targets.reshape(-1, context)


def evaluation_batches(token_stream, target_positions, context, batch_size=8):
    """Each target gets exactly context preceding tokens; score only its prediction."""
    batches = []
    for offset in range(0, len(target_positions), batch_size):
        positions = target_positions[offset : offset + batch_size]
        inputs = torch.stack(
            [token_stream[position - context : position] for position in positions]
        )
        targets = token_stream[positions]
        batches.append((inputs, targets))
    return batches


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    total_loss = 0.0
    target_count = 0
    for inputs, targets in batches:
        final_position_logits = model(inputs)[:, -1, :]
        total_loss += F.cross_entropy(final_position_logits, targets, reduction="sum").item()
        target_count += targets.numel()
    model.train()
    return total_loss / target_count


def initialize_models(vocab_size, seed):
    """Copy shared weights and position-table prefixes into independent models."""
    torch.manual_seed(seed)
    reference_model = StackedTransformerLanguageModel(vocab_size, n_layers=4, block_size=256)
    reference_weights = reference_model.state_dict()
    models = {}
    for context in CONTEXT_LENGTHS:
        model = StackedTransformerLanguageModel(vocab_size, n_layers=4, block_size=context)
        initial_weights = dict(reference_weights)
        initial_weights["position_embedding_table.weight"] = reference_weights[
            "position_embedding_table.weight"
        ][:context]
        model.load_state_dict(initial_weights)
        models[context] = model
    return models


@torch.no_grad()
def generate_sample(model, prompt_ids, context, vocabulary):
    model.eval()
    token_ids = torch.tensor([prompt_ids])
    sampling_generator = torch.Generator().manual_seed(123)
    for _ in range(100):
        logits = model(token_ids[:, -context:])[:, -1, :]
        next_token = torch.multinomial(logits.softmax(-1), 1, generator=sampling_generator)
        token_ids = torch.cat((token_ids, next_token), dim=1)
    model.train()
    continuation_ids = token_ids[0, len(prompt_ids) :].tolist()
    return {
        "prompt": bpe_decode(prompt_ids, vocabulary, errors="replace"),
        "prompt_ids": prompt_ids,
        "continuation": bpe_decode(continuation_ids, vocabulary, errors="replace"),
        "continuation_ids": continuation_ids,
        "sampling_seed": 123,
        "temperature": 1.0,
    }


def run(steps=10000, seeds=(42, 43, 44), output="artifacts/context-comparison-10k"):
    torch.set_num_threads(2)
    output_directory = Path(output)
    output_directory.mkdir(parents=True, exist_ok=True)
    if list(output_directory.glob("*.pt")) or (output_directory / "results.json").exists():
        raise FileExistsError("choose a new directory to preserve previous experiment artifacts")
    tokens, vocabulary, _ = load_tiny_shakespeare_tokens(Path("data"))
    training_tokens, validation_tokens = split_token_stream(tokens)
    streams = {
        "train": torch.tensor(training_tokens),
        "validation": torch.tensor(validation_tokens),
    }
    evaluation_rng = random.Random(2026)
    target_positions = {}
    for split, stream in streams.items():
        target_positions[split] = evaluation_rng.sample(range(256, len(stream)), 128)
    fixed_batches = {}
    for context in CONTEXT_LENGTHS:
        fixed_batches[context] = {}
        for split, stream in streams.items():
            fixed_batches[context][split] = evaluation_batches(
                stream, target_positions[split], context
            )
    source_paths = [
        "experiments/compare_context_lengths.py",
        "src/language_models.py",
        "src/dataset.py",
        "src/tokenizer.py",
        "data/tiny_shakespeare.txt",
        "data/tiny_shakespeare_bpe.json",
    ]
    source_hashes = {}
    for source_path in source_paths:
        source_hashes[source_path] = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    report = {
        "config": {
            "steps": steps,
            "seeds": list(seeds),
            "contexts": list(CONTEXT_LENGTHS),
            "depth": 4,
            "width": 32,
            "heads": 2,
            "tokens_per_update": TOKENS_PER_UPDATE,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "evaluation_every": 500,
            "evaluation_targets_per_split": 128,
            "evaluation_target_positions": target_positions,
            "training_tokens": len(training_tokens),
            "validation_tokens": len(validation_tokens),
            "vocab_size": len(vocabulary),
            "source_sha256": source_hashes,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "torch_version": torch.__version__,
            "device": "cpu",
            "torch_threads": 2,
            "training_protocol": "same contiguous 256 targets per update; reshape changes sequence boundaries",
            "evaluation_protocol": "same target positions; last-position loss with full preceding context",
            "initialization": "same non-position weights; shared learned-position prefix",
        },
        "runs": [],
        "checkpoint_samples": [],
        "cross_evaluation": [],
        "cross_samples": [],
    }
    prompt_ids = training_tokens[:256]
    for seed in seeds:
        models = initialize_models(len(vocabulary), seed)
        optimizers = {}
        histories = {}
        update_seconds = {}
        for context, model in models.items():
            optimizers[context] = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            histories[context] = []
            update_seconds[context] = 0.0
        batch_rng = random.Random(seed)
        for step in range(steps + 1):
            if step % 500 == 0 or step == steps:
                for context, model in models.items():
                    row = {"step": step}
                    for split, batches in fixed_batches[context].items():
                        row[split] = evaluate(model, batches)
                    histories[context].append(row)
                    print(
                        f"seed {seed}, context {context}: completed evaluation at update {step}",
                        flush=True,
                    )
            if step in (1000, 2500, 5000, 10000) or step == steps:
                for context, model in models.items():
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizers[context].state_dict(),
                        "step": step,
                        "seed": seed,
                        "context": context,
                        "batch_rng_state": batch_rng.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                        "history": histories[context],
                        "config": report["config"],
                    }
                    torch.save(
                        checkpoint,
                        output_directory / f"context-{context}-seed-{seed}-step-{step}.pt",
                    )
                    sample = generate_sample(model, prompt_ids, context, vocabulary)
                    sample.update(
                        {
                            "seed": seed,
                            "context": context,
                            "step": step,
                            "validation_loss": histories[context][-1]["validation"],
                        }
                    )
                    report["checkpoint_samples"].append(sample)
            if step == steps:
                break
            start = batch_rng.randrange(len(training_tokens) - TOKENS_PER_UPDATE)
            for context, model in models.items():
                inputs, targets = training_batch(streams["train"], start, context)
                started = time.perf_counter()
                optimizers[context].zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, len(vocabulary)), targets.reshape(-1))
                loss.backward()
                optimizers[context].step()
                update_seconds[context] += time.perf_counter() - started
        for training_context, model in models.items():
            for evaluation_context in CONTEXT_LENGTHS:
                if evaluation_context > training_context:
                    continue
                evaluation_row = {
                    "seed": seed,
                    "training_context": training_context,
                    "evaluation_context": evaluation_context,
                }
                for split, batches in fixed_batches[evaluation_context].items():
                    evaluation_row[split] = evaluate(model, batches)
                report["cross_evaluation"].append(evaluation_row)
                if seed == 42:
                    sample = generate_sample(model, prompt_ids, evaluation_context, vocabulary)
                    sample.update(evaluation_row)
                    sample["step"] = steps
                    report["cross_samples"].append(sample)
        for context, model in models.items():
            report["runs"].append(
                {
                    "seed": seed,
                    "context": context,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "history": histories[context],
                    "update_seconds": update_seconds[context],
                }
            )
        (output_directory / "results.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--output", default="artifacts/context-comparison-10k")
    arguments = parser.parse_args()
    run(steps=arguments.steps, output=arguments.output)
