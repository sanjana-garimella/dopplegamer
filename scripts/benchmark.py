"""CLI for running Doppelgamer systems and agent benchmarks.

Usage:
    python scripts/benchmark.py agents --rounds 50
    python scripts/benchmark.py systems --engines baseline vllm --model mock
    python scripts/benchmark.py profiling --type throughput
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the root of the repo is in PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.runner import run_benchmark
from inference.setup_inference_engines import setup_inference_engines, EngineConfig
from analysis.scheduling_overhead import benchmark_engines as profile_scheduling
from analysis.throughput_benchmark import concurrency_sweep
from analysis.prefill_decode_split import compare_engines as profile_prefill_decode


def main():
    parser = argparse.ArgumentParser(description="Doppelgamer Benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", help="Benchmark category")

    # Agents Benchmark
    agents_parser = subparsers.add_parser("agents", help="Run agent performance benchmarks")
    agents_parser.add_argument("--rounds", type=int, default=100, help="Games per agent per seed")
    agents_parser.add_argument("--agents", nargs="+", help="Specific agents to test")
    agents_parser.add_argument("--seeds", type=int, default=3, help="Number of seeds for variance")
    agents_parser.add_argument("--games", nargs="+", default=["RPS+"], help="Game types to benchmark")
    agents_parser.add_argument("--db", default="data/game_data.db")

    # Systems Benchmark
    systems_parser = subparsers.add_parser("systems", help="Run inference engine benchmarks")
    systems_parser.add_argument("--engines", nargs="+", default=["baseline", "vllm"])
    systems_parser.add_argument("--model", default="mock")
    systems_parser.add_argument("--rounds", type=int, default=20)
    systems_parser.add_argument("--db", default="data/game_data.db")
    systems_parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="On real models, fall back to mock/baseline instead of failing loud",
    )

    # Specialized Profiling
    profiling_parser = subparsers.add_parser("profiling", help="Run specialized systems profiling")
    profiling_parser.add_argument("--type", choices=["scheduling", "throughput", "prefill_decode"], required=True)
    profiling_parser.add_argument("--model", default="mock")
    profiling_parser.add_argument("--engine", default="baseline")
    profiling_parser.add_argument("--allow-fallback", action="store_true")

    args = parser.parse_args()

    if args.command == "agents":
        print(f"Running agent benchmarks ({args.rounds} rounds, {args.seeds} seeds)...")
        results = run_benchmark(
            rounds=args.rounds,
            engines=[],
            agents=args.agents,
            n_seeds=args.seeds,
            games=args.games,
            db_path=args.db
        )
        print(f"Done. Run ID: {results['run_id']}")
        if results.get("game_breakdown"):
            print("Per-game breakdown:")
            for row in results["game_breakdown"]:
                print(
                    f"  {row['agent_name']} on {row['game_type']}: "
                    f"win_rate={row['win_rate']:.3f} "
                    f"games={row['games_played']} "
                    f"ci95=[{row['win_rate_stats']['ci95_low']:.3f}, {row['win_rate_stats']['ci95_high']:.3f}]"
                )

    elif args.command == "systems":
        print(f"Running inference benchmarks for {args.engines} using {args.model}...")
        # Systems benchmarking is integrated into run_benchmark when agents=[]
        results = run_benchmark(
            rounds=args.rounds,
            agents=[],  # Skip agents
            engines=args.engines,
            model_name=args.model,
            db_path=args.db,
            allow_fallback=args.allow_fallback,
        )
        print(f"Done. Results saved to {args.db}")

    elif args.command == "profiling":
        engine_pool = setup_inference_engines(
            EngineConfig(model_name=args.model, allow_fallback=args.allow_fallback),
            engines=[args.engine],
        )
        if args.engine not in engine_pool:
            print(f"Error: engine {args.engine} not found.")
            sys.exit(1)
        
        engine = engine_pool[args.engine]
        
        if args.type == "scheduling":
            print(
                f"Profiling host idle wait (wall-cpu; not serving scheduler) for {args.engine}..."
            )
            reports = profile_scheduling({args.engine: engine}, measure_runs=50)
            print(reports[args.engine].summary())
            
        elif args.type == "throughput":
            print(f"Running throughput sweep for {args.engine} (safe mode auto-selected)...")
            reports = concurrency_sweep(engine, engine_name=args.engine, levels=[1, 2, 4, 8])
            for r in reports:
                print(r.summary())
                
        elif args.type == "prefill_decode":
            print(f"Profiling prefill/decode latency split for {args.engine}...")
            summaries = profile_prefill_decode({args.engine: engine}, measure_runs=30)
            print(summaries[args.engine].summary())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
