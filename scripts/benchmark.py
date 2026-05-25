from __future__ import annotations

import argparse
import json
import statistics
from time import perf_counter

import httpx


def benchmark(url: str, payload: dict, runs: int) -> None:
    latencies: list[float] = []
    for _ in range(runs):
        start = perf_counter()
        response = httpx.post(url, json=payload, timeout=300.0)
        response.raise_for_status()
        latencies.append(perf_counter() - start)

    print({
        "runs": runs,
        "mean_seconds": round(statistics.mean(latencies), 4),
        "p95_seconds": round(sorted(latencies)[max(0, int(runs * 0.95) - 1)], 4),
        "min_seconds": round(min(latencies), 4),
        "max_seconds": round(max(latencies), 4),
    })


def benchmark_matrix(matrix_path: str) -> None:
    with open(matrix_path, "r", encoding="utf-8") as handle:
        matrix = json.load(handle)

    for case in matrix["cases"]:
        benchmark(case["url"], case["payload"], case.get("runs", 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt", default="A cinematic city skyline at dawn")
    parser.add_argument("--model-id", default="distilgpt2")
    parser.add_argument("--kind", choices=["text", "image"], default="text")
    parser.add_argument("--matrix", help="Path to a benchmark matrix JSON file.")
    args = parser.parse_args()

    if args.matrix:
        benchmark_matrix(args.matrix)
        return

    if args.kind == "text":
        payload = {
            "model_id": args.model_id,
            "prompt": args.prompt,
            "max_new_tokens": 64,
        }
    else:
        payload = {
            "model_id": args.model_id,
            "prompt": args.prompt,
            "num_inference_steps": 20,
            "height": 512,
            "width": 512,
        }

    benchmark(args.url, payload, args.runs)


if __name__ == "__main__":
    main()
