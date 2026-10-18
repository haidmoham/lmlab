# recorded softmax evidence

these compact records contain all seven completed runs, not a selected set of winning shapes. full machine-local JSON, logs, archives, profiler files, and notebooks remain outside Git. no calibration or profiling run is recorded as completed.

- `measurements.csv` has 648 provider measurements across 252 shape/run records. each median/min/max is in microseconds; effective_gbps is minimum logical I/O-equivalent GB/s, not measured DRAM traffic.
- `samples.csv` has all 6696 raw batch means in milliseconds per call. order_position is zero-based. the historical two-way positions are recovered from their recorded sample sequence; the three-way positions were explicitly recorded.
- `provenance.json` contains settings, tolerances, sanitized hardware/software snapshots, original raw-file hashes, and exact measured-source hashes. output directories, host paths, and GPU UUIDs are omitted.
- `three-way-measured.patch` reconstructs the exact measured three-way source on the recorded base commit. the working source was later formatted without changing its Python AST. the earlier two-way source is the base commit itself.
- `verify.py` independently checks summaries, raw samples, order balance, source reconstruction, and current-source AST equivalence on CPU.

```bash
uv run --locked python reports/experiments/softmax-results/verify.py
```

the verifier needs Git and the recorded base commit. it does not import PyTorch or launch GPU work. a source-only export can still inspect the CSV and metadata, but cannot reconstruct source history without that commit.

the reports distinguish observations from interpretation. ranges are batch-mean ranges, not confidence intervals. no limiter profile or serving-speed claim is supplied.
