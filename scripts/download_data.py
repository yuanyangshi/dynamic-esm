#!/usr/bin/env python3
"""
CLI Script: Dynamic-ESM Dataset Downloader.
Fetches clinical case study PDB structures directly from RCSB Protein Data Bank
and provides instructions for obtaining MISATO MD trajectories and CASF-2016 benchmarks.
"""

import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dynamic_esm.utils.logging import get_logger

logger = get_logger("dynamic_esm.cli.download")

RCSB_PDB_DOWNLOADS = {
    "1IEP": "https://files.rcsb.org/download/1IEP.pdb",
    "2JIT": "https://files.rcsb.org/download/2JIT.pdb",
    "3QRJ": "https://files.rcsb.org/download/3QRJ.pdb",
    "6LUD": "https://files.rcsb.org/download/6LUD.pdb",
    "6OIM": "https://files.rcsb.org/download/6OIM.pdb",
    "7VH8": "https://files.rcsb.org/download/7VH8.pdb",
}


def download_case_study_pdbs(target_dir: str = "./data/pdbs"):
    """Download the 6 clinical case study PDB structures from RCSB PDB."""
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Downloading clinical case study PDBs to {target_dir}...")

    for pdb_id, url in RCSB_PDB_DOWNLOADS.items():
        dst_path = os.path.join(target_dir, f"{pdb_id}.pdb")
        if os.path.exists(dst_path):
            logger.info(f"Already exists: {dst_path}")
            continue

        try:
            logger.info(f"Fetching {pdb_id} from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp, open(dst_path, "wb") as f_out:
                f_out.write(resp.read())
            logger.info(f"Saved: {dst_path}")
        except Exception as e:
            logger.error(f"Failed to download {pdb_id}: {e}")

    logger.info("Case study PDB download routine completed.")


def print_dataset_sources():
    """Print official download links and instructions for external datasets."""
    print("=" * 70)
    print("Dynamic-ESM: Official Dataset & Weight Download Sources")
    print("=" * 70)
    print("0. Pre-trained Weights & Processed Datasets (Kaggle Cloud):")
    print("   - Checkpoints & Workspace (NB3): https://www.kaggle.com/datasets/shixinguo/nb3-workspace")
    print("   - Processed Features & Graphs (NB1): https://www.kaggle.com/code/shixinguo/nb1-20260809/output\n")
    print("1. MISATO MD Trajectories & QM Calculations (Zenodo):")
    print("   URL: https://zenodo.org/records/7711953")
    print("   DOI: 10.5281/zenodo.7711953")
    print("   Files to download into ./data:")
    print("     - MD_split_*.hdf5")
    print("     - QM_clean_norm.hdf5 (or QM.hdf5)")
    print("     - train_keys.txt, val_keys.txt, test_keys.txt\n")
    print("2. CASF-2016 Gold-Standard Benchmarking:")
    print("   URL: http://www.pdbbind.org.cn/casf.php")
    print("   File: CASF-2016.tar.gz (285 core set complexes)\n")
    print("3. RCSB Protein Data Bank:")
    print("   URL: https://www.rcsb.org/")
    print("   Run `python scripts/download_data.py --download-pdbs` to fetch automatically.")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic-ESM dataset download assistant.")
    parser.add_argument("--download-pdbs", action="store_true", help="Download case study PDBs from RCSB.")
    parser.add_argument("--info", action="store_true", help="Print official dataset sources and instructions.")
    parser.add_argument("--target-dir", type=str, default="./data/pdbs", help="Directory to save downloaded PDBs.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.download_pdbs:
        download_case_study_pdbs(target_dir=args.target_dir)
    else:
        print_dataset_sources()


if __name__ == "__main__":
    main()
