import argparse
import statistics
from typing import List, Dict, Any

import pandas as pd

from router import Plan, benchmark_plan


def read_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def pctl(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for plan_key, g in df.groupby(["model_name", "quantize_int8", "num_threads", "use_cache"]):
        lat = g["latency_ms"].tolist()
        tps = g["tokens_per_sec"].tolist()
        mem = g["rss_mb"].tolist()
        out_toks = g["output_tokens"].tolist()

        rows.append({
            "model_name": plan_key[0],
            "quantize_int8": plan_key[1],
            "num_threads": plan_key[2],
            "use_cache": plan_key[3],
            "p50_latency_ms": pctl(lat, 50),
            "p95_latency_ms": pctl(lat, 95),
            "avg_tokens_per_sec": statistics.mean(tps) if tps else 0.0,
            "peak_rss_mb": max(mem) if mem else 0.0,
            "avg_output_tokens": statistics.mean(out_toks) if out_toks else 0.0,
        })
    return pd.DataFrame(rows)


def choose_best(summary_df: pd.DataFrame, sla_ms: float) -> pd.DataFrame:
    # Filter those that meet p50 SLA; if none, pick fastest p50 anyway.
    ok = summary_df[summary_df["p50_latency_ms"] <= sla_ms].copy()
    if len(ok) == 0:
        return summary_df.sort_values(["p50_latency_ms", "peak_rss_mb"], ascending=[True, True]).head(1)

    # "Cost" heuristic: prefer higher throughput, then lower memory
    ok = ok.sort_values(["avg_tokens_per_sec", "peak_rss_mb"], ascending=[False, True])
    return ok.head(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts.txt")
    ap.add_argument("--models", nargs="+", default=["distilgpt2", "gpt2"])
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--sla_ms", type=float, default=1500.0)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    args = ap.parse_args()

    prompts = read_prompts(args.prompts)
    if len(prompts) < 5:
        raise SystemExit("Please add at least 5 prompts to prompts.txt for a more stable benchmark.")

    all_rows: List[Dict[str, Any]] = []

    plans = []
    for m in args.models:
        for q in [False, True]:
            for t in args.threads:
                for cache in [False, True]:
                    plans.append(Plan(model_name=m, quantize_int8=q, num_threads=t, use_cache=cache))

    print(f"Running {len(plans)} plans × {len(prompts)} prompts ...")

    for i, plan in enumerate(plans, 1):
        print(f"[{i}/{len(plans)}] {plan}")
        runs = benchmark_plan(plan, prompts, max_new_tokens=args.max_new_tokens, warmup=1)
        for r in runs:
            all_rows.append({
                **r.plan,
                "prompt_chars": r.prompt_chars,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
                "tokens_per_sec": r.tokens_per_sec,
                "rss_mb": r.rss_mb,
            })

    df = pd.DataFrame(all_rows)
    df.to_csv("results.csv", index=False)

    s = summarize(df)
    s = s.sort_values(["p50_latency_ms", "peak_rss_mb"], ascending=[True, True])
    print("\n=== Summary (sorted by p50 latency, then memory) ===")
    print(s.to_string(index=False))

    best = choose_best(s, args.sla_ms)
    print(f"\n=== Chosen plan (SLA p50 <= {args.sla_ms:.0f} ms) ===")
    print(best.to_string(index=False))

    s.to_csv("summary.csv", index=False)
    best.to_csv("chosen_plan.csv", index=False)
    print("\nSaved: results.csv, summary.csv, chosen_plan.csv")


if __name__ == "__main__":
    main()
