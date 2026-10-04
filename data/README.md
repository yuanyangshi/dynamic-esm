# Dynamic-ESM Dataset Sources & Download Instructions

This directory contains instructions and direct download links for all datasets utilized in **Dynamic-ESM**. To ensure repository lightness and compliance with open-source distribution standards, raw dataset files are not tracked in git and should be downloaded using the links or scripts below.

---

## 0. Pre-trained Checkpoints & Preprocessed Dynamic Data (Direct Cloud Downloads)

For quick inference or reproduction without reprocessing terabyte-scale raw trajectory files:

- **Pre-trained Checkpoints & Workspace (NB3)**:
  - **Kaggle Dataset**: [https://www.kaggle.com/datasets/shixinguo/nb3-workspace](https://www.kaggle.com/datasets/shixinguo/nb3-workspace)
  - Contains: `dynamic_esm_main_best.ckpt` (Epoch 47 global optimal checkpoint, validation loss = 4.888) and full training logs.
  - Recommended destination: Save to `./output/checkpoints/dynamic_esm_main_best.ckpt` or root `./output/`.

- **Preprocessed 4D-QM Trajectories & Feature Archives (NB1)**:
  - **Kaggle Execution Output**: [https://www.kaggle.com/code/shixinguo/nb1-20260809/output](https://www.kaggle.com/code/shixinguo/nb1-20260809/output)
  - Contains: Preprocessed PyTorch Geometric `.pt` graph chunks, extracted ESM-2 representations, and MMseqs2 30% sequence identity cluster splits.
  - Recommended destination: Extract to `./data/processed_pt/` or `./output/processed_pt/`.

---

## 1. MISATO Molecular Dynamics & Quantum Chemistry Dataset

The primary training dataset consists of multi-nanosecond molecular dynamics trajectories and quantum chemical (QM) polarizability calculations from the **MISATO** database.

- **Zenodo Official Repository**: [https://zenodo.org/records/7711953](https://zenodo.org/records/7711953)
  - DOI: `10.5281/zenodo.7711953`
  - Required files:
    - `MD_split_*.hdf5` (Trajectories containing atomic coordinates and periodic boundary information)
    - `QM_clean_norm.hdf5` or `QM.hdf5` (GFN2-xTB semi-empirical atomic partial charges and polarizabilities)
    - `train_keys.txt`, `val_keys.txt`, `test_keys.txt`
- **MISATO Project GitHub**: [https://github.com/sab39/misato-dataset](https://github.com/sab39/misato-dataset)

---

## 2. CASF-2016 Gold-Standard Benchmarking Dataset

Used for scoring, ranking, docking, and screening power validation.

- **PDBbind / CASF-2016 Official Portal**: [http://www.pdbbind.org.cn/casf.php](http://www.pdbbind.org.cn/casf.php)
- **Direct Package**: `CASF-2016.tar.gz` (contains the 285 co-crystallized complexes of the core set)
- **Reference**: Su, M. et al. *Comparative Assessment of Scoring Functions (CASF-2016)*, J. Chem. Inf. Model. 2019, 59, 895–913.

---

## 3. Clinical Case Studies (RCSB Protein Data Bank)

Used for mechanistic interpretability and clinical resistance mutation hotspot mapping (EGFR, Abl, KRAS, SARS-CoV-2 Mpro):

| System / Target | PDB ID | Inhibitor | Direct Download Link |
| :--- | :---: | :---: | :--- |
| **Abl Kinase** | `1IEP` | Imatinib | [https://files.rcsb.org/download/1IEP.pdb](https://files.rcsb.org/download/1IEP.pdb) |
| **EGFR Kinase** | `2JIT` | Gefitinib | [https://files.rcsb.org/download/2JIT.pdb](https://files.rcsb.org/download/2JIT.pdb) |
| **Abl T315I Mutant** | `3QRJ` | Ponatinib | [https://files.rcsb.org/download/3QRJ.pdb](https://files.rcsb.org/download/3QRJ.pdb) |
| **EGFR T790M/C797S** | `6LUD` | Osimertinib | [https://files.rcsb.org/download/6LUD.pdb](https://files.rcsb.org/download/6LUD.pdb) |
| **KRAS G12C** | `6OIM` | Sotorasib | [https://files.rcsb.org/download/6OIM.pdb](https://files.rcsb.org/download/6OIM.pdb) |
| **SARS-CoV-2 Mpro** | `7VH8` | Nirmatrelvir | [https://files.rcsb.org/download/7VH8.pdb](https://files.rcsb.org/download/7VH8.pdb) |

---

## 4. Automated Download CLI Script

You can automatically download all clinical case study PDB structures into `data/pdbs/` by running:

```bash
python scripts/download_data.py --download-pdbs
```

---

## 5. Storage Requirements & Licensing

| Dataset / Resource | Est. Storage | Access / License | Primary Reference / Source |
| :--- | :---: | :--- | :--- |
| **MISATO Trajectories & QM** | ~20 GB (zip) / ~45 GB (raw) | CC-BY 4.0 | [10.5281/zenodo.7711953](https://zenodo.org/records/7711953) |
| **CASF-2016 Core Benchmark** | ~420 MB | Academic Research | Su et al., *J. Chem. Inf. Model.* 2019 |
| **Clinical Case Studies (PDBs)**| ~15 MB | CC0 1.0 (Public Domain) | RCSB Protein Data Bank |

---

## 6. Expected Directory Layout

After downloading datasets, the `data/` directory should be arranged as follows:

```text
data/
├── misato/                      # MISATO MD and QM database files
│   ├── MD_split_0.hdf5          # Multi-frame coordinate trajectories
│   ├── ...
│   ├── QM.hdf5                  # Quantum chemical partial charges & polarizabilities
│   ├── train_keys.txt           # Training split keys
│   ├── val_keys.txt             # Validation split keys
│   └── test_keys.txt            # Test split keys
├── casf2016/                    # CASF-2016 benchmark package
│   ├── coreset/                 # 285 core complex structures & crystal ligands
│   └── power_scoring/           # Scoring and ranking assessment scripts
└── pdbs/                        # Clinical case studies for interpretability
    ├── 1IEP.pdb                 # Abl Kinase (Imatinib)
    ├── 2JIT.pdb                 # EGFR Kinase (Gefitinib)
    ├── 3QRJ.pdb                 # Abl T315I mutant (Ponatinib)
    ├── 6LUD.pdb                 # EGFR T790M/C797S mutant (Osimertinib)
    ├── 6OIM.pdb                 # KRAS G12C (Sotorasib)
    └── 7VH8.pdb                 # SARS-CoV-2 Mpro (Nirmatrelvir)
```
