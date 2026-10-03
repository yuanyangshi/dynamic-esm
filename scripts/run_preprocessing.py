#!/usr/bin/env python3
"""
CLI Script: Dynamic-ESM Data Preprocessing Pipeline.
Transforms raw MISATO HDF5 and QM datasets into PyTorch Geometric graph representations.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import glob
import os
import sys

from dynamic_esm.config import Config
from dynamic_esm.data.preprocessor import MISATOPreprocessor
from dynamic_esm.utils.logging import get_logger

logger = get_logger("dynamic_esm.cli.preprocessing")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MISATO HDF5 & QM data into PyG graphs.")
    parser.add_argument("--input-dir", type=str, default="./data", help="Directory containing MISATO HDF5 files.")
    parser.add_argument("--output-dir", type=str, default="./data/processed_pt", help="Output directory for processed graphs.")
    parser.add_argument("--esm-model", type=str, default="esm2_t12_35M_UR50D", help="ESM-2 model architecture.")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of MD trajectory frames to sample.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing MISATO Preprocessor...")
    preprocessor = MISATOPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        esm_model_name=args.esm_model,
        num_frames=args.num_frames,
    )

    split_files = sorted(glob.glob(os.path.join(args.input_dir, "MD_split_*.hdf5")))
    if not split_files:
        logger.warning(f"No MD_split_*.hdf5 files found in {args.input_dir}. Please place MISATO files there.")
        return

    logger.info(f"Found {len(split_files)} split files to process.")
    for split_path in split_files:
        preprocessor.run_split(split_path)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
