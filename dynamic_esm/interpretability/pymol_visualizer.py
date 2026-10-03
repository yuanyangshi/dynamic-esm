"""
Automated PyMOL Script Generation & B-Factor Attention Mapping.
Generates 3D publication-grade structural visualization scripts (.pml).
"""

import os
from typing import Dict, List, Optional
import numpy as np

from dynamic_esm.utils.logging import get_logger

logger = get_logger("dynamic_esm.pymol")


def export_pdb_with_attention(
    in_pdb_path: str,
    out_pdb_path: str,
    residue_attention_map: Dict[int, float],
) -> str:
    """
    Reads original PDB file, replaces the B-factor column with normalized attention scores,
    and writes the annotated structure for PyMOL heatmap rendering.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_pdb_path)), exist_ok=True)

    with open(in_pdb_path, "r", encoding="utf-8", errors="ignore") as f_in, open(
        out_pdb_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    res_num = int(line[22:26].strip())
                    attn = residue_attention_map.get(res_num, 0.0)
                    # B-factor is at columns 61-66 (6 characters, formatted as %6.2f)
                    b_factor_str = f"{attn * 100.0:6.2f}"
                    new_line = line[:60] + b_factor_str + line[66:]
                    f_out.write(new_line)
                except Exception:
                    f_out.write(line)
            else:
                f_out.write(line)

    logger.info(f"Exported attention-annotated PDB to: {out_pdb_path}")
    return out_pdb_path


def generate_pymol_script(
    pdb_path: str,
    out_pml_path: str,
    critical_residues: Optional[List[int]] = None,
    image_name: Optional[str] = None,
) -> str:
    """
    Generates a publication-grade PyMOL command script (.pml).
    Colors the binding pocket according to attention B-factors (Blue -> White -> Red).
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_pml_path)), exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdb_path))[0]
    critical_residues = critical_residues or []
    res_sel_str = "+".join(str(r) for r in critical_residues)

    pml_content = f"""# ==============================================================================
# Dynamic-ESM Automated PyMOL Visualizer
# System: {base_name}
# ==============================================================================
reinitialize
bg_color white
set ray_shadows, 1
set ray_trace_mode, 1
set antialias, 2
set cartoon_transparency, 0.2

# Load Structure
load {os.path.basename(pdb_path)}, {base_name}

# Surface and Cartoon Display
hide everything, {base_name}
show cartoon, {base_name}
color gray85, {base_name}

# Color Binding Pocket by Model Attention (B-factor: 0 -> 100)
spectrum b, blue_white_red, {base_name}, minimum=0, maximum=100

# Highlight Ligand
select ligand, {base_name} and organic
show sticks, ligand
set stick_radius, 0.28, ligand
color green, ligand

# Highlight Clinically Critical Mutation Residues
select hotspots, {base_name} and resi {res_sel_str}
show sticks, hotspots
show surface, hotspots
set transparency, 0.4, hotspots
label hotspots and name CA, "%s%s" % (resn, resi)
set label_size, 14
set label_color, black
set label_font_id, 7

# Pocket Zoom & View
orient ligand
zoom ligand, 8.0

# Render High-Resolution Raster Image (300 DPI)
{f"ray 2400, 1800" if image_name else "# ray 2400, 1800"}
{f"png {image_name}, dpi=300" if image_name else "# png output.png, dpi=300"}
"""

    with open(out_pml_path, "w", encoding="utf-8") as f:
        f.write(pml_content)

    logger.info(f"PyMOL visualization script written: {out_pml_path}")
    return out_pml_path
