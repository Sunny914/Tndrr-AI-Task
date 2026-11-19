
import csv
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import requests


# 1. Project configuration

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434"
OLLAMA_EXE = (
    Path(os.environ["LOCALAPPDATA"])
    / "Programs"
    / "Ollama"
    / "ollama.exe"
)

MODELS = ["qwen-fp16", "qwen-q4"]
REPETITIONS = 3

OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_ctx": 2048,
    "num_predict": 128,
}

FIELDS = [
    "model",
    "prompt_id",
    "category",
    "run",
    "prompt",
    "response",
    "output_tokens",
    "decode_seconds",
    "tokens_per_second",
    "wall_seconds",
    "prompt_eval_seconds",
    "load_seconds",
    "done_reason",
    "gpu_memory_mib",
    "loaded_size_mib",
]


# 2. Read the dataset

def load_prompts():
    with open(ROOT / "prompts.json", encoding="utf-8") as f:
        prompts = json.load(f)

    assert len(prompts) == 10
    return prompts


# 3. Send an inference request

def generate(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": OPTIONS,
    }

    start = time.perf_counter()

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=300,
    )

    wall_seconds = time.perf_counter() - start
    response.raise_for_status()

    return response.json(), wall_seconds


# 4. Stop models to isolate measurements

def unload_models():
    for model in MODELS:
        subprocess.run(
            [str(OLLAMA_EXE), "stop", model],
            capture_output=True,
            text=True,
            check=False,
        )


# 5. Measure loaded-model memory

def get_model_memory(model):
    response = requests.get(
        f"{OLLAMA_URL}/api/ps",
        timeout=15,
    )
    response.raise_for_status()

    for item in response.json().get("models", []):
        if item["name"].split(":")[0] == model:
            return {
                "gpu_memory_mib": round(
                    item.get("size_vram", 0) / 1048576, 2
                ),
                "loaded_size_mib": round(
                    item["size"] / 1048576, 2
                ),
            }

    raise RuntimeError(f"Model not loaded: {model}")


# 6. Convert API response into a result row

def build_row(model, item, run, data, wall, memory):
    tokens = data.get("eval_count", 0)
    decode = data.get("eval_duration", 0) / 1e9

    tps = tokens / decode if decode > 0 else 0.0

    return {
        "model": model,
        "prompt_id": item["id"],
        "category": item["category"],
        "run": run,
        "prompt": item["prompt"],
        "response": data.get("response", ""),
        "output_tokens": tokens,
        "decode_seconds": round(decode, 6),
        "tokens_per_second": round(tps, 2),
        "wall_seconds": round(wall, 4),
        "prompt_eval_seconds": round(
            data.get("prompt_eval_duration", 0) / 1e9, 4
        ),
        "load_seconds": round(
            data.get("load_duration", 0) / 1e9, 4
        ),
        "done_reason": data.get("done_reason", ""),
        **memory,
    }


# 7. Aggregate results

def summarize(rows):
    summary = {}

    for model in MODELS:
        selected = [
            r for r in rows
            if r["model"] == model
        ]

        total_tokens = sum(
            r["output_tokens"] for r in selected
        )
        total_decode_time = sum(
            r["decode_seconds"] for r in selected
        )

        # Aggregate throughput is less sensitive
        # to very short responses.
        aggregate_tps = (
            total_tokens / total_decode_time
            if total_decode_time > 0 else 0
        )

        summary[model] = {
            "runs": len(selected),
            "aggregate_tokens_per_second": round(
                aggregate_tps, 2
            ),
            "median_wall_seconds": round(
                statistics.median(
                    r["wall_seconds"] for r in selected
                ), 3
            ),
            "gpu_memory_mib": selected[0]["gpu_memory_mib"],
            "loaded_size_mib": selected[0]["loaded_size_mib"],
            "truncated_responses": sum(
                r["done_reason"] == "length"
                for r in selected
            ),
        }

    return summary


# 8. Main experimental workflow

def main():
    prompts = load_prompts()
    rows = []

    output_csv = RESULTS_DIR / "benchmark_results.csv"

    # Confirm server availability.
    requests.get(
        f"{OLLAMA_URL}/api/version",
        timeout=10,
    ).raise_for_status()

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        for model in MODELS:
            print(f"\n===== BENCHMARKING {model} =====")

            # Avoid competing loaded models.
            unload_models()

            # Exclude cold start from measured runs.
            print("Warming up...")
            generate(
                model,
                "Explain what inference means in one sentence.",
            )

            memory = get_model_memory(model)
            print("Loaded-model memory:", memory)

            for item in prompts:
                for run in range(1, REPETITIONS + 1):
                    data, wall = generate(
                        model,
                        item["prompt"],
                    )

                    row = build_row(
                        model,
                        item,
                        run,
                        data,
                        wall,
                        memory,
                    )

                    rows.append(row)
                    writer.writerow(row)
                    file.flush()

                    print(
                        f"{model} | {item['id']} "
                        f"| run {run} "
                        f"| {row['tokens_per_second']} tok/s "
                        f"| {row['wall_seconds']}s"
                    )

            unload_models()

    summary = summarize(rows)

    with open(
        RESULTS_DIR / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n===== FINAL SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("\nResults saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
