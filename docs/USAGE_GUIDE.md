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

# Forward pass with sequence tokens and dynamic graph structures
with torch.no_grad():
    # tokens: [B, L], node_x: [B, T, N, 3], node_attr: [B, T, N, F], edge_index: [2, E]
    outputs = model(tokens, node_x, node_attr, edge_index)
    
    # Extract predicted binding affinity and evidential uncertainties
    y_pred = outputs["pred"]                   # Mean predicted pK_d / -log(K_d)
    sigma2_alea = outputs["aleatoric_var"]     # Data/experimental noise
    sigma2_epis = outputs["epistemic_var"]     # Model epistemic uncertainty
    
print(f"Predicted Affinity: {y_pred.item():.3f} | Epistemic Uncertainty: {sigma2_epis.item():.4f}")
```

