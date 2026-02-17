import pandas as pd
import streamlit as st
import subprocess
import os

st.set_page_config(page_title="Cost-Aware LLM Inference Optimizer", layout="wide")
st.title("Cost-Aware LLM Inference Optimizer (CPU MVP)")

st.markdown("Select constraints and run a benchmark. The app will pick the best plan that meets your SLA.")

col1, col2, col3 = st.columns(3)

with col1:
    models = st.multiselect("Models", ["distilgpt2", "gpt2"], default=["distilgpt2", "gpt2"])
    max_new_tokens = st.slider("max_new_tokens", 16, 256, 64, step=16)

with col2:
    sla_ms = st.slider("p50 latency SLA (ms)", 200, 5000, 1500, step=100)
    threads = st.multiselect("Threads to try", [1, 2, 4, 8], default=[1, 2, 4])

with col3:
    prompts_file = st.text_input("Prompts file", "prompts.txt")
    st.caption("Tip: add 10–30 realistic prompts for stable results.")

run = st.button("Run Benchmark")

if run:
    if not models:
        st.error("Pick at least one model.")
        st.stop()
    if not threads:
        st.error("Pick at least one thread count.")
        st.stop()
    if not os.path.exists(prompts_file):
        st.error(f"Cannot find prompts file: {prompts_file}")
        st.stop()

    cmd = [
        "python", "benchmark.py",
        "--sla_ms", str(sla_ms),
        "--max_new_tokens", str(max_new_tokens),
        "--prompts", prompts_file,
        "--models", *models,
        "--threads", *map(str, threads),
    ]

    st.code(" ".join(cmd), language="bash")
    with st.spinner("Benchmark running..."):
        p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        st.error("Benchmark failed.")
        st.text(p.stderr)
        st.stop()

    st.success("Done!")
    st.text(p.stdout)

    # Load outputs
    if os.path.exists("summary.csv"):
        summary = pd.read_csv("summary.csv")
        st.subheader("Plan Summary")
        st.dataframe(summary, use_container_width=True)

        st.subheader("Charts")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("p50 latency (ms)")
            st.bar_chart(summary.set_index(summary.index)["p50_latency_ms"])
        with c2:
            st.caption("tokens/sec")
            st.bar_chart(summary.set_index(summary.index)["avg_tokens_per_sec"])
        with c3:
            st.caption("peak RSS (MB)")
            st.bar_chart(summary.set_index(summary.index)["peak_rss_mb"])

    if os.path.exists("chosen_plan.csv"):
        chosen = pd.read_csv("chosen_plan.csv")
        st.subheader("Chosen Plan")
        st.dataframe(chosen, use_container_width=True)

    if os.path.exists("results.csv"):
        results = pd.read_csv("results.csv")
        st.subheader("Raw Results (per prompt)")
        st.dataframe(results.head(200), use_container_width=True)
