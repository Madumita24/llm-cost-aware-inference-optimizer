import os
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class Plan:
    model_name: str
    quantize_int8: bool
    num_threads: int
    use_cache: bool


@dataclass
class RunMetrics:
    plan: Dict[str, Any]
    prompt_chars: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tokens_per_sec: float
    rss_mb: float


# def set_threads(n: int) -> None:
#     # Controls intra-op parallelism for CPU ops
#     torch.set_num_threads(n)
#     torch.set_num_interop_threads(max(1, min(4, n)))

_INTEROP_SET = False

def set_threads(n: int) -> None:
    global _INTEROP_SET
    # Intra-op threads can change between runs
    torch.set_num_threads(n)

    # Inter-op threads must be set only once (and early)
    if not _INTEROP_SET:
        torch.set_num_interop_threads(max(1, min(4, n)))
        _INTEROP_SET = True



def rss_mb() -> float:
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)


def maybe_quantize_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    # Dynamic quantization for CPU Linear layers
    model.eval()
    return torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )


def load_model_and_tokenizer(model_name: str, quantize_int8: bool):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    if quantize_int8:
        model = maybe_quantize_dynamic(model)

    return model, tok


@torch.inference_mode()
def run_once(
    model,
    tok,
    prompt: str,
    max_new_tokens: int,
    use_cache: bool,
) -> Dict[str, Any]:
    inputs = tok(prompt, return_tensors="pt")
    input_tokens = int(inputs["input_ids"].shape[1])

    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=use_cache,
        pad_token_id=tok.eos_token_id,
    )
    t1 = time.perf_counter()

    output_tokens = int(out.shape[1] - input_tokens)
    latency_s = (t1 - t0)
    tps = (output_tokens / latency_s) if latency_s > 0 and output_tokens > 0 else 0.0

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_s * 1000.0,
        "tokens_per_sec": tps,
    }


def benchmark_plan(
    plan: Plan,
    prompts: List[str],
    max_new_tokens: int,
    warmup: int = 1,
) -> List[RunMetrics]:
    set_threads(plan.num_threads)

    model, tok = load_model_and_tokenizer(plan.model_name, plan.quantize_int8)

    
    for _ in range(warmup):
        _ = run_once(model, tok, prompts[0], max_new_tokens=max_new_tokens, use_cache=plan.use_cache)

    results: List[RunMetrics] = []
    for pr in prompts:
        before_rss = rss_mb()
        r = run_once(model, tok, pr, max_new_tokens=max_new_tokens, use_cache=plan.use_cache)
        after_rss = rss_mb()

        results.append(
            RunMetrics(
                plan=asdict(plan),
                prompt_chars=len(pr),
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                latency_ms=r["latency_ms"],
                tokens_per_sec=r["tokens_per_sec"],
                rss_mb=max(before_rss, after_rss),
            )
        )

    return results
