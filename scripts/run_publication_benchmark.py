"""Fixed-protocol systems benchmark for publication tables.

Runs baseline vs vLLM only (library mode, single-stream + optional vLLM batch
throughput), records hardware metadata, p50/p95, and exports CSV.

Usage:
    python scripts/run_publication_benchmark.py --model distilgpt2 --rounds 50
    python scripts/run_publication_benchmark.py --model meta-llama/Llama-3.2-1B --rounds 50
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.scheduling_overhead import benchmark_engines as profile_host_wait
from analysis.throughput_benchmark import concurrency_sweep
from analysis.prefill_decode_split import compare_engines as profile_prefill_decode
from evaluation.runner import run_benchmark
from inference.setup_inference_engines import EngineConfig, setup_inference_engines
from scripts.export_results import export


def _hardware_metadata() -> dict:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            meta["cuda_device"] = torch.cuda.get_device_name(0)
            meta["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception as exc:
        meta["torch_error"] = str(exc)
    try:
        import vllm

        meta["vllm"] = getattr(vllm, "__version__", "unknown")
    except Exception:
        meta["vllm"] = None
    try:
        import transformers

        meta["transformers"] = transformers.__version__
    except Exception:
        meta["transformers"] = None
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Publication-protocol systems benchmark")
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--db", default="data/publication_run.db")
    parser.add_argument("--out", default="results/publication")
    parser.add_argument("--seed", type=int, default=0, help="Seed for game-driven prompts")
    parser.add_argument("--engines", nargs="+", default=["baseline", "vllm"])
    parser.add_argument("--skip-profilers", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _hardware_metadata()
    meta.update(
        {
            "model": args.model,
            "rounds": args.rounds,
            "engines": args.engines,
            "seed": args.seed,
            "prompt_seed": args.seed,
            "methodology": (
                "library-mode single-stream systems benchmark with game-driven prompts; "
                "vLLM throughput uses generate_batch (continuous batching); "
                "host_wait is wall-cpu, not serving-scheduler time; "
                "batch per-request latency may be estimated (see latency_estimated); "
                "preble/infercept only if remote BASE_URL is set"
            ),
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("Running systems benchmark...")
    result = run_benchmark(
        rounds=args.rounds,
        engines=args.engines,
        agents=[],
        db_path=args.db,
        model_name=args.model,
        n_seeds=1,
        allow_fallback=False,
        prompt_seed=args.seed,
    )
    print(f"run_id={result['run_id']} prompt_seed={result.get('prompt_seed')}")

    if not args.skip_profilers:
        pool = setup_inference_engines(
            EngineConfig(model_name=args.model, allow_fallback=False),
            engines=args.engines,
        )
        host = profile_host_wait(pool, measure_runs=min(30, args.rounds))
        # Local engines must report host_wait (wall-cpu), not a zero engine field.
        remote = {"preble", "infercept"}
        for name, report in host.items():
            if name in remote:
                continue
            if report.metric != "host_wait_ms":
                raise RuntimeError(
                    f"host-wait profiler used metric={report.metric!r} for {name}; "
                    f"expected host_wait_ms (scheduling_overhead_ms must be None when unreported)"
                )
        host_path = out_dir / "host_wait.json"
        host_path.write_text(json.dumps({k: v.to_dict() for k, v in host.items()}, indent=2))
        print(f"wrote {host_path}")

        splits = profile_prefill_decode(pool, measure_runs=min(30, args.rounds))
        split_path = out_dir / "prefill_decode.json"
        split_path.write_text(
            json.dumps({k: v.to_dict() for k, v in splits.items()}, indent=2)
        )
        print(f"wrote {split_path}")

        throughput = {}
        for name, engine in pool.items():
            reports = concurrency_sweep(
                engine, engine_name=name, levels=[1, 2, 4, 8], requests_per_level=min(32, args.rounds)
            )
            throughput[name] = [r.to_dict() for r in reports]
            for r in reports:
                print(r.summary())
        thr_path = out_dir / "throughput.json"
        thr_path.write_text(json.dumps(throughput, indent=2))
        print(f"wrote {thr_path}")

    written = export(Path(args.db), out_dir)
    for path in written:
        print(path)
    print("done")


if __name__ == "__main__":
    main()
