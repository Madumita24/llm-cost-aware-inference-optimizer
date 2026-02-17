# Cost-Aware LLM Inference Optimizer (CPU-only)

This repo builds a small inference “router” that automatically selects an execution plan for LLM inference under CPU constraints.

## Why this exists
LLM inference cost is dominated by latency and memory constraints. This project evaluates multiple inference plans (model choice × quantization × threading × KV-cache usage) and selects the cheapest plan that satisfies a latency SLA.

## Plans searched
Each plan is a combination of:
- Model: distilgpt2, gpt2 (configurable)
- Dynamic INT8 quantization for Linear layers (CPU) on/off
- Torch CPU threads: 1/2/4 (configurable)
- KV-cache (`use_cache`) on/off

## Metrics measured
- p50 / p95 latency (ms)
- tokens/sec (throughput)
- peak RSS memory (MB)

## Run
```bash
python benchmark.py --sla_ms 1500 --models distilgpt2 gpt2 --threads 1 2 4 --max_new_tokens 64
