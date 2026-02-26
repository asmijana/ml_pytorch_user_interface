from __future__ import annotations

import subprocess
from pathlib import Path
import yaml
import pandas as pd
import streamlit as st
import json
import sys

st.markdown(
    "<style>html { overflow-y: scroll; }</style>",
    unsafe_allow_html=True
)
st.sidebar.write("Streamlit Python:", sys.executable)
try:
    import torch
    st.sidebar.write("Torch OK:", torch.__version__)
except Exception as e:
    st.sidebar.error(f"Torch import failed: {e}")
    st.stop()

# ---------------------------
# Helpers
# ---------------------------
def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def save_yaml(cfg: dict, path: Path) -> None:
    ensure_parent(path)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def parse_hidden_dims(s: str) -> list[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def run_training(config_path: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "src.main", "--config", str(config_path)]
    return subprocess.run(cmd, capture_output=True, text=True)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="PyTorch MLP Regression UI", layout="wide")
st.markdown(
    """
    <style>
      html { scrollbar-gutter: stable both-edges; }
      body { overflow-x: hidden; }
      div.block-container { max-width: 1400px; width: 100%; }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("PyTorch MLP Regression UI → writes configs/baseline.yaml → trains → plots")

cfg_path = Path("configs/baseline.yaml")

with st.sidebar:
    st.header("Experiment")
    exp_name = st.text_input("experiment.name", value="baseline_mlp_california_housing")

    st.header("Data")
    data_out_dir = st.text_input("data.out_dir", value="data")
    test_size = st.slider("data.test_size", 0.05, 0.30, 0.20, 0.01)
    val_size = st.slider("data.val_size", 0.05, 0.30, 0.10, 0.01)
    split_seed = st.number_input("data.random_state", value=42, step=1)

    st.header("Model")
    hidden_dims_str = st.text_input("model.hidden_dims (e.g., 128,64)", value="128,64")
    dropout = st.slider("model.dropout", 0.0, 0.8, 0.10, 0.05)

    st.header("Train")
    lr = st.number_input("train.lr", value=1e-3, format="%.6f")
    batch_size = st.number_input("train.batch_size", value=256, step=32)
    max_epochs = st.number_input("train.max_epochs", value=200, step=10)
    weight_decay = st.number_input("train.weight_decay", value=0.0, format="%.6f")
    patience = st.number_input("train.patience", value=20, step=1)
    min_delta = st.number_input("train.min_delta", value=0.0, format="%.6f")
    num_workers = st.number_input("train.num_workers", value=0, step=1)

    st.header("Runtime")
    seed = st.number_input("runtime.seed", value=7, step=1)
    device = st.selectbox("runtime.device", ["auto", "cpu", "cuda"], index=0)
    use_amp = st.checkbox("runtime.use_amp", value=False)

    st.header("Actions")
    write_only = st.button("Write baseline.yaml")
    run_clicked = st.button("Write baseline.yaml + Run training", type="primary")

# Build config dict entirely from UI (no baseline.yaml read)
cfg = {
    "experiment": {"name": str(exp_name)},
    "data": {
        "out_dir": str(data_out_dir),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "random_state": int(split_seed),
    },
    "model": {
        "hidden_dims": parse_hidden_dims(hidden_dims_str),
        "dropout": float(dropout),
    },
    "train": {
        "lr": float(lr),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "weight_decay": float(weight_decay),
        "patience": int(patience),
        "min_delta": float(min_delta),
        "num_workers": int(num_workers),
    },
    "runtime": {
        "seed": int(seed),
        "device": str(device),
        "use_amp": bool(use_amp),
    },
}

run_dir = Path("runs") / cfg["experiment"]["name"]
st.caption(f"Artifacts will be written to: `{run_dir}`")

# Sanity check
if cfg["data"]["test_size"] + cfg["data"]["val_size"] >= 0.95:
    st.error("Invalid split sizes: test_size + val_size is too large. Reduce them.")
    st.stop()

# Write YAML (button)
if write_only:
    save_yaml(cfg, cfg_path)
    st.success(f"Wrote {cfg_path}")

# Write + train
if run_clicked:
    save_yaml(cfg, cfg_path)

    st.success(f"Wrote {cfg_path}")
    with st.spinner("Training..."):
        result = run_training(cfg_path)

    st.subheader("Training logs")
    if result.returncode != 0:
        st.error("Training failed.")
        st.code(result.stderr or "(no stderr)", language="text")
        st.stop()
    else:
        st.code(result.stdout[-12000:] if result.stdout else "(no stdout)", language="text")

# Show results if present
st.subheader("Results")
metrics_csv = run_dir / "metrics.csv"
loss_png = run_dir / "loss_curve.png"
parity_train = run_dir / "parity_train.png"
parity_val = run_dir / "parity_val.png"
parity_test = run_dir / "parity_test.png"
test_summary = run_dir / "test_summary.json"

if metrics_csv.exists():
    df = pd.read_csv(metrics_csv)
    st.write("Metrics (last 10 rows):")
    st.dataframe(df.tail(10), use_container_width=True, height=320)
    st.line_chart(df.set_index("epoch")[["train_loss", "val_loss"]])

    #if loss_png.exists():
    #    st.image(str(loss_png), caption="Loss curve (saved by src.main)", use_container_width=True)
else:
    st.info("No metrics.csv yet. Click **Write baseline.yaml + Run training**.")

cols = st.columns(3)
for c, pth, title in zip(cols, [parity_train, parity_val, parity_test], ["Train parity", "Validation parity", "Test parity"]):
    with c:
        if pth.exists():
            st.image(str(pth), caption=title, use_container_width=True)
        else:
            st.caption(f"{title}: (not found)")

if test_summary.exists():
    st.subheader("Test summary")
    with test_summary.open("r") as f:
        st.json(json.load(f))
