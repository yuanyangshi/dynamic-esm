"""
MISATO HDF5 and QM Dataset Preprocessor for Dynamic-ESM.
Extracts 4D molecular dynamics trajectories, QM charges/dipoles, and ESM-2 embeddings.
"""

import glob
import json
import math
import os
import shutil
import time
import zipfile
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch_geometric.data import Data

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    AllChem = None

import esm
from dynamic_esm.utils.logging import get_logger

logger = get_logger("dynamic_esm.preprocessor")


class MISATOPreprocessor:
    """
    Transforms raw MISATO MD trajectories and QM HDF5 files into PyG Graph objects.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        esm_model_name: str = "esm2_t12_35M_UR50D",
        num_frames: int = 5,
        pocket_radius: float = 12.0,
        load_esm: bool = False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.esm_model_name = esm_model_name
        self.num_frames = num_frames
        self.pocket_radius = pocket_radius

        os.makedirs(self.output_dir, exist_ok=True)

        # Check QM file
        qm_path = os.path.join(self.input_dir, "QM_clean_norm.hdf5")
        if not os.path.exists(qm_path):
            qm_path = os.path.join(self.input_dir, "QM.hdf5")

        self.qm_data_available = os.path.exists(qm_path)
        if self.qm_data_available:
            self.qm_file = h5py.File(qm_path, "r")
            logger.info(f"Loaded QM file: {qm_path}")
        else:
            self.qm_file = None
            logger.warning("QM file not found. QM features will be zero-padded.")

        # Load ESM-2 optionally
        self.esm_model = None
        self.esm_alphabet = None
        self.esm_batch_converter = None
        if load_esm:
            try:
                logger.info(f"Loading ESM-2: {self.esm_model_name}...")
                self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet(self.esm_model_name)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.esm_model = self.esm_model.to(self.device).eval()
                self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
                logger.info("ESM-2 loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load ESM-2 model ({e}). Fallback to offline embeddings.")

        self.split_dict = self._load_splits()

    def close(self):
        """Close open HDF5 file handles."""
        if self.qm_file is not None:
            try:
                self.qm_file.close()
            except Exception:
                pass

    def _load_splits(self) -> Dict[str, str]:
        split_dict = {}
        for subset in ["train", "val", "test"]:
            key_file = os.path.join(self.input_dir, f"{subset}_keys.txt")
            if os.path.exists(key_file):
                with open(key_file, "r") as f:
                    for line in f:
                        k = line.strip()
                        if k:
                            split_dict[k] = subset
        return split_dict

    def _extract_trajectory_array(self, complex_group) -> Optional[np.ndarray]:
        for k in ["trajectory_coordinates", "trajectoryCoordinates", "trajectory", "coordinates"]:
            if k in complex_group:
                return complex_group[k][:]
        return None

    def _extract_atom_features(self, complex_group, num_atoms: int) -> torch.Tensor:
        supported_Z = [6, 7, 8, 9, 15, 16, 17, 35, 53]
        one_hot = np.zeros((num_atoms, len(supported_Z) + 1), dtype=np.float32)

        Z = None
        for k in ["atoms_element", "atoms_type", "Z", "atomic_numbers"]:
            if k in complex_group:
                Z = complex_group[k][:]
                break

        if Z is not None:
            for i, z_val in enumerate(Z):
                if z_val in supported_Z:
                    one_hot[i, supported_Z.index(z_val)] = 1.0
                else:
                    one_hot[i, -1] = 1.0
        else:
            one_hot[:, 0] = 1.0  # Default carbon fallback

        return torch.tensor(one_hot, dtype=torch.float32)

    def _extract_qm_features(
        self, pdb_id: str, num_atoms: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        charges = torch.zeros(num_atoms, 1, dtype=torch.float32)
        dipoles = torch.zeros(num_atoms, 3, dtype=torch.float32)

        if not self.qm_data_available or pdb_id not in self.qm_file:
            return charges, dipoles

        qm_group = self.qm_file[pdb_id]
        if "atom_properties" in qm_group and "atom_properties_values" in qm_group["atom_properties"]:
            vals = qm_group["atom_properties"]["atom_properties_values"][:]
            n_qm = min(vals.shape[0], num_atoms)
            # Take first column as charge
            charges[:n_qm, 0] = torch.tensor(vals[:n_qm, 0], dtype=torch.float32)
            if vals.shape[1] >= 4:
                dipoles[:n_qm, :] = torch.tensor(vals[:n_qm, 1:4], dtype=torch.float32)

        return charges, dipoles

    def _extract_ligand_fingerprint(self, complex_group) -> torch.Tensor:
        if Chem is None:
            return torch.zeros(2048, dtype=torch.float32)

        smiles = None
        for k in ["ligand_smiles", "smiles"]:
            if k in complex_group:
                val = complex_group[k][()]
                smiles = val.decode("utf-8") if isinstance(val, bytes) else str(val)
                break

        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                return torch.tensor(list(fp), dtype=torch.float32)

        return torch.zeros(2048, dtype=torch.float32)

    def process_complex(self, pdb_id: str, complex_group) -> Optional[Data]:
        trajectory = self._extract_trajectory_array(complex_group)
        if trajectory is None:
            return None

        total_frames = trajectory.shape[0]
        num_atoms = trajectory.shape[1]

        if total_frames < self.num_frames:
            frame_indices = np.arange(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        atom_feats = self._extract_atom_features(complex_group, num_atoms)
        charges, dipoles = self._extract_qm_features(pdb_id, num_atoms)
        ligand_fp = self._extract_ligand_fingerprint(complex_group)

        # Base node features: atom elements (10) + charge (1) + dipoles (3)
        base_x = torch.cat([atom_feats, charges, dipoles], dim=-1)
        pos_trajectory = torch.tensor(trajectory[frame_indices], dtype=torch.float32)

        graph_data = Data(
            x=base_x,
            pos=pos_trajectory,
            charges=charges,
            ligand_fp=ligand_fp.unsqueeze(0),
            pdb_id=pdb_id,
        )
        return graph_data

    def run_split(self, split_file_path: str) -> None:
        split_name = os.path.basename(split_file_path).split(".")[0]
        logger.info(f"Processing split: {split_name}")
        archive_name = os.path.join(self.output_dir, f"{split_name}.zip")

        try:
            with h5py.File(split_file_path, "r") as f:
                pdb_ids = list(f.keys())
                logger.info(f"Found {len(pdb_ids)} complexes in {split_name}")

                with zipfile.ZipFile(archive_name, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for i, pdb_id in enumerate(pdb_ids):
                        graph_data = self.process_complex(pdb_id, f[pdb_id])
                        if graph_data is not None:
                            subset = self.split_dict.get(pdb_id, "train")
                            tmp_path = os.path.join(self.output_dir, f"{subset}_{pdb_id}.pt")
                            torch.save(graph_data, tmp_path)
                            zipf.write(tmp_path, arcname=os.path.basename(tmp_path))
                            os.remove(tmp_path)

                        if (i + 1) % 100 == 0 or (i + 1) == len(pdb_ids):
                            logger.info(f"Processed {i + 1}/{len(pdb_ids)} complexes.")
        except Exception as e:
            logger.error(f"Error processing split {split_name}: {e}")
