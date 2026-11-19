# TNDRR AI/ML Engineering Assignment

Three completed experiments for the TNDRR AI/ML Engineering internship: parameter-efficient fine-tuning, framework-free retrieval-augmented generation, and local LLM quantization benchmarking.

This repository is an **experiment and documentation artifact**. It is not a production service.

Combined report (exactly four A4 pages): [`docs/TNDRR_Assignment_Report.pdf`](docs/TNDRR_Assignment_Report.pdf)

## Task summary

| Task | Folder | What was measured | Headline result |
| --- | --- | --- | --- |
| 1. LoRA fine-tuning | [`Task-1-FineTuning with LoRA/`](Task-1-FineTuning%20with%20LoRA/README.md) | AG News 4-class accuracy on 100 held-out articles | 28% → **84%** accuracy; Macro-F1 0.2058 → **0.8383** |
| 2. Framework-free RAG | [`Task-2-RAG-From-Scratch/`](Task-2-RAG-From-Scratch/README.md) | Page-based retrieval on 15 verified NIST AI RMF questions | Config B Hit@1 **0.600**, Hit@5 **0.933**, MRR **0.739** |
| 3. LLM inference | [`Task-3-Running-Model-Locally-Optimization-LLM-Inferrence/`](Task-3-Running-Model-Locally-Optimization-LLM-Inferrence/README.md) | FP16 vs Q4_K_M on 60 Ollama runs | FP16 **84.88 tok/s**, Q4 **69.0 tok/s**; VRAM **1003.94** vs **435.22 MiB** |

## Key measured results

**Task 1** — `Qwen2.5-0.5B-Instruct`, AG News, 800 / 100 / 100 stratified splits, rank-8 LoRA on `q_proj` and `v_proj`.

- Trainable adapter parameters: **540,672** (0.1093% of 494,573,440)
- Training runtime: **119.07 s** on NVIDIA Tesla T4
- Sixteen LoRA errors remain; three are documented in [`Task-1-FineTuning with LoRA/results/failure_cases.csv`](Task-1-FineTuning%20with%20LoRA/results/failure_cases.csv)
- 7B memory figures in `memory_estimates.json` are **theoretical parameter arithmetic**, not a 7B training run

**Task 2** — NIST AI RMF 1.0, **48** physical PDF pages, MiniLM embeddings, NumPy cosine top-5, local `qwen-q4`.

| Config | Tokens / overlap | Chunks | Hit@1 | Hit@5 | MRR | Quote Hit@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | 120 / 20 | 213 | 0.533 | 0.933 | 0.706 | 0.600 |
| B | 220 / 40 | 126 | 0.600 | 0.933 | 0.739 | 0.800 |

Two documented retrieval failures: **Q14** (Hit@5 miss, both configs) and **Q03** (Hit@1 and quote miss on config A).

**Task 3** — Same model family, GGUF FP16 vs Q4_K_M, sequential Ollama requests on a 4 GB GTX 1650.

- 10 prompts × 3 repetitions × 2 models = **60** measured runs
- Median wall latency: FP16 **2.528 s**, Q4 **2.574 s**
- The **20 concurrent users / 24 GiB GPU** design is a **proposal** ([`Task-3-Running-Model-Locally-Optimization-LLM-Inferrence/docs/DEPLOYMENT_DESIGN.md`](Task-3-Running-Model-Locally-Optimization-LLM-Inferrence/docs/DEPLOYMENT_DESIGN.md)). It was **not load-tested**.

## Repository structure

```text
Tndrr-Task-1,2,&3/
├── Task-1-FineTuning with LoRA/   # notebook, LoRA adapter, metrics
├── Task-2-RAG-From-Scratch/       # PDF, source pipeline, gold questions, results
├── Task-3-Running-Model-Locally-Optimization-LLM-Inferrence/  # benchmark, CSVs, design
├── docs/TNDRR_Assignment_Report.pdf
├── README.md
└── .gitignore
```

Large GGUF weights, virtualenvs, and embedding caches are excluded. Re-download GGUF files for Task 3 as described in that task README.

## Reproduction

Each task is independent. Use Python 3.13 and the task-local `requirements.txt`.

1. **Task 1** — Open [`Task-1-FineTuning with LoRA/notebooks/LoRA_FineTuning.ipynb`](Task-1-FineTuning%20with%20LoRA/notebooks/LoRA_FineTuning.ipynb). Metrics are already saved under [`Task-1-FineTuning with LoRA/results/`](Task-1-FineTuning%20with%20LoRA/results/). Reloading the adapter needs the Hugging Face base model; the adapter itself is in [`Task-1-FineTuning with LoRA/adapters/lora_adapter/`](Task-1-FineTuning%20with%20LoRA/adapters/lora_adapter/).
2. **Task 2** — `cd Task-2-RAG-From-Scratch`, install requirements, then `python run.py ingest` and `python run.py evaluate`. Gold labels: [`Task-2-RAG-From-Scratch/evaluation/gold_questions.csv`](Task-2-RAG-From-Scratch/evaluation/gold_questions.csv). Local Ollama `qwen-q4` is required only for `python run.py ask`.
3. **Task 3** — Install Ollama, download the two GGUF files into `models/` (gitignored), `ollama create` from the Modelfiles, then `python benchmark.py`. Saved measurements are in [`Task-3-Running-Model-Locally-Optimization-LLM-Inferrence/results/`](Task-3-Running-Model-Locally-Optimization-LLM-Inferrence/results/).

Do not treat Task 3 sequential GTX 1650 numbers as 24 GB multi-user capacity.

## Limitations

- Task 1 uses 100 test examples and one LoRA recipe; WORLD recall fell while overall accuracy rose.
- Task 2 page-level Hit@k overestimates evidence retrieval (quote Hit@5 is stricter).
- Task 3 quality scores in `quality_scored.csv` are **provisional**; the 20-user architecture is **untested**.
- No production APIs, frontends, or paid cloud services are included.

## Report

[`docs/TNDRR_Assignment_Report.pdf`](docs/TNDRR_Assignment_Report.pdf) — four A4 pages covering all three tasks, using only numbers stored in this repository.
