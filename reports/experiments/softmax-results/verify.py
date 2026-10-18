"""Verify recorded softmax evidence on CPU without importing a GPU framework."""

import ast
import csv
import hashlib
import itertools
import json
import math
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    metadata = json.loads((directory / "provenance.json").read_text())
    measurements = list(csv.DictReader((directory / "measurements.csv").open()))
    samples = list(csv.DictReader((directory / "samples.csv").open()))
    assert len(measurements) == metadata["counts"]["provider_measurements"] == 648
    assert len(samples) == metadata["counts"]["raw_batch_samples"] == 6696
    grouped = defaultdict(list)
    orders = defaultdict(dict)
    for row in samples:
        key = (row["run"], int(row["rows"]), int(row["width"]), row["provider"])
        value = float(row["milliseconds_per_call"])
        assert math.isfinite(value) and value > 0
        grouped[key].append((int(row["repeat"]), value))
        order_key = (*key[:3], int(row["repeat"]))
        position = int(row["order_position"])
        assert position not in orders[order_key]
        orders[order_key][position] = row["provider"]
    seen = set()
    for row in measurements:
        key = (row["run"], int(row["rows"]), int(row["width"]), row["provider"])
        assert key not in seen
        seen.add(key)
        run = metadata["runs"][row["run"]]
        assert row["provider"] in run["providers"]
        assert row["dtype"] == run["settings"]["dtype"]
        assert sorted(repeat for repeat, _ in grouped[key]) == list(
            range(run["settings"]["repeats"])
        )
        values = [value * 1000 for _, value in grouped[key]]
        expected = {
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
        }
        byte_count = 2 * key[1] * key[2] * (4 if row["dtype"] == "float32" else 2)
        expected["effective_gbps"] = byte_count / (expected["median_us"] * 1000)
        for field, value in expected.items():
            assert math.isclose(float(row[field]), value, rel_tol=1e-10, abs_tol=1e-10)
    assert seen == set(grouped)
    for name, run in metadata["runs"].items():
        config = run["settings"]
        providers = tuple(run["providers"])
        shapes = set(itertools.product(config["rows"], config["widths"]))
        assert len(shapes) == 36
        assert {key[1:3] for key in seen if key[0] == name} == shapes
        for shape in shapes:
            schedule = [
                tuple(orders[(name, *shape, repeat)][p] for p in range(len(providers)))
                for repeat in range(config["repeats"])
            ]
            assert all(set(order) == set(providers) for order in schedule)
            if len(providers) == 3:
                permutations = list(itertools.permutations(providers))
                assert schedule == [
                    permutations[(i + config["order_offset"]) % 6] for i in range(12)
                ]
                assert Counter(schedule) == Counter({order: 2 for order in permutations})
                for a, b in itertools.combinations(providers, 2):
                    assert sum(order.index(a) < order.index(b) for order in schedule) == 6
            else:
                assert schedule == [providers if i % 2 == 0 else providers[::-1] for i in range(7)]
    source = metadata["three_way_measured_source"]
    patch = directory / source["patch"]
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == source["patch_sha256"]
    with tempfile.TemporaryDirectory(prefix="lmlab-softmax-evidence-") as temporary:
        reconstructed = Path(temporary)
        expected = metadata["runs"]["three-way/fp32-1"]["environment"]["source_sha256"]
        paths = set(expected) | set(source["files_in_patch"])
        for path in paths:
            content = subprocess.check_output(
                ["git", "show", f"{source['base_commit']}:{path}"], cwd=root
            )
            target = reconstructed / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=reconstructed, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=reconstructed, check=True)
        for path, digest in expected.items():
            assert hashlib.sha256((reconstructed / path).read_bytes()).hexdigest() == digest
        assert (
            hashlib.sha256((reconstructed / "tests/test_softmax.py").read_bytes()).hexdigest()
            == source["test_source_sha256"]
        )
        for path in source["files_in_patch"]:
            assert ast.dump(
                ast.parse((reconstructed / path).read_text()), include_attributes=False
            ) == ast.dump(ast.parse((root / path).read_text()), include_attributes=False)
    print(
        "verified 7 runs, 252 shape/run records, 648 summaries, 6696 raw samples, balanced orders, and measured-source reconstruction"
    )


if __name__ == "__main__":
    main()
