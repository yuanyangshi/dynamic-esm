"""
Unit tests for raw MISATO HDF5 and QM data preprocessor.
"""

import os
import tempfile
import h5py
import numpy as np
from dynamic_esm.data.preprocessor import MISATOPreprocessor


def test_misato_preprocessor_synthetic_h5():
    with tempfile.TemporaryDirectory() as tmp_dir:
        preprocessor = MISATOPreprocessor(input_dir=tmp_dir, output_dir=tmp_dir, load_esm=False)
        h5_path = os.path.join(tmp_dir, "test.h5")

        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("1abc")
            grp.create_dataset("trajectory_coordinates", data=np.random.randn(10, 20, 3).astype(np.float32))
            grp.create_dataset("atoms_element", data=np.full(20, 6, dtype=np.int32))
            grp.create_dataset("ligand_smiles", data="CC(=O)OC1=CC=CC=C1C(=O)O".encode("utf-8"))

            graph = preprocessor.process_complex("1abc", grp)
            assert graph is not None
            assert graph.x.shape == (20, 14)  # 10 element one-hot + 1 charge + 3 dipoles
            assert graph.pos.shape == (5, 20, 3)
            assert graph.pdb_id == "1abc"
            assert hasattr(graph, "ligand_fp")
            assert graph.ligand_fp.shape == (1, 2048)

        preprocessor.close()
