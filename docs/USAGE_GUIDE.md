# Dynamic-ESM Usage Guide

A comprehensive guide for setting up, training, and deploying Dynamic-ESM.

---

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/yuanyangshi/dynamic-esm.git
cd dynamic-esm

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## 2. Quickstart: CLI Workflow

### Step 1: Preprocess Multimodal Trajectories
```bash
python scripts/run_preprocessing.py \
    --input-dir ./data \
    --output-dir ./output/processed_pt \
    --num-frames 5
```

### Step 2: Pretrain 4D-QM Spatiotemporal Backbone
```bash
python scripts/run_pretrain.py \
    --data-dir ./output/processed_pt \
    --checkpoint-dir ./output/checkpoints \
    --epochs 30
```

### Step 3: End-to-End Multimodal Fine-Tuning
```bash
python scripts/run_finetune.py \
    --data-dir ./output/processed_pt \
    --pretrain-dir ./output/checkpoints \
    --checkpoint-dir ./output/checkpoints \
    --figures-dir ./output/figures \
    --epochs 30
```

### Step 4: CASF-2016 Benchmarking & Physical Ablation
```bash
python scripts/run_benchmark.py \
    --predictions-file ./output/checkpoints/val_empirical_predictions.npz \
    --output-dir ./output
```

### Step 5: Biological Interpretability & PyMOL Visualization
```bash
python scripts/run_interpretability.py \
    --pdb-dir ./data/pdbs \
    --output-dir ./output
```

### Step 6: Virtual Screening & AF3 Induced-Fit Rectification
```bash
python scripts/run_screening.py \
    --predictions-file ./output/checkpoints/val_empirical_predictions.npz \
    --output-dir ./output
```

## 3. Python API Integration

You can directly import and integrate Dynamic-ESM modules into your custom computational pipelines:

### 3.1 Initializing from YAML Configuration
```python
from dynamic_esm.config import Config
from dynamic_esm.models import DynamicESM

# Load hyperparameters directly from YAML
cfg = Config.from_yaml("configs/default_config.yaml")

# Initialize model architecture
model = DynamicESM(
    esm_model_name=cfg.ESM_MODEL_NAME,
    gnn_hidden_dim=cfg.GNN_HIDDEN_DIM,
    cross_attn_dim=cfg.CROSS_ATTN_DIM,
    lora_rank=cfg.LORA_RANK,
)
```

### 3.2 End-to-End Prediction with Uncertainty Quantification
```python
import torch

# Put model in evaluation mode
model.eval()

# Forward pass with atomic features, coordinate trajectory, and sequence embeddings
with torch.no_grad():
    # x: [N, 654], pos: [T, N, 3], seq_embeds: [1, L, 480]
    gamma, v, alpha, beta = model(x, pos, seq_embeds=seq_embeds, return_attn=False)
    
    # Compute evidential predictive uncertainties via NIG formulation
    aleatoric, epistemic, total = model.head.compute_uncertainty(v, alpha, beta)
    y_pred = gamma  # Mean predicted pK_d / -log(K_d)
    
print(f"Predicted Affinity: {y_pred.item():.3f} | Epistemic Uncertainty: {epistemic.item():.4f}")
```

