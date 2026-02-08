"""
Configuration for LSD Binding Affinity Evaluation Pipeline.

Defines molecules, receptors, docking parameters, and file paths for
AutoDock Vina docking of LSD and structural variants against serotonin
receptor subtypes.
"""

import os
from pathlib import Path

# === Directory paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = Path(__file__).resolve().parent

SDF_DIR = PROJECT_ROOT / "molecule_data" / "sdf_files"
PDB_DIR = PROJECT_ROOT / "receptors" / "pdb_files"

PREPARED_LIGANDS_DIR = BASELINE_DIR / "prepared" / "ligands"
PREPARED_RECEPTORS_DIR = BASELINE_DIR / "prepared" / "receptors"
RESULTS_DIR = BASELINE_DIR / "results"
VIS_DIR = BASELINE_DIR / "visualizations"

# Ensure output directories exist
for d in [PREPARED_LIGANDS_DIR, PREPARED_RECEPTORS_DIR, RESULTS_DIR, VIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Molecules ===
# 8 lysergamide variants: LSD + 7 structural analogs
# Note: LSA_5143.sdf is saccharin (wrong molecule); use Ergine_442072.sdf instead
MOLECULES = {
    "LSD":         {"sdf": "LSD_5761.sdf",          "mw": 323.4},
    "1P-LSD":      {"sdf": "1P-LSD_119025985.sdf",  "mw": 393.5},
    "ALD-52":      {"sdf": "ALD-52_201111.sdf",      "mw": 365.4},
    "ETH-LAD":     {"sdf": "ETH-LAD_44457783.sdf",   "mw": 337.5},
    "AL-LAD":      {"sdf": "AL-LAD_15227511.sdf",    "mw": 349.5},
    "PRO-LAD":     {"sdf": "PRO-LAD_44457803.sdf",   "mw": 351.5},
    "Ergine":      {"sdf": "Ergine_442072.sdf",       "mw": 267.3},
    "Ergometrine": {"sdf": "Ergometrine_443884.sdf",  "mw": 325.4},
}

# === Receptors ===
# 11 primary serotonin receptor subtypes (best resolution each)
# + 2 validation structures for cross-checking
#
# Chain IDs: cryo-EM structures typically use chain R for receptor,
# X-ray structures use chain A.
# ligand_resname: 3-letter code of co-crystallized ligand in PDB
# ligand_chain: chain the ligand resides on

RECEPTORS = {
    # --- Primary structures (11 subtypes) ---
    "5-HT1A_7E2Y": {
        "pdb": "7E2Y.pdb",
        "subtype": "5-HT1A",
        "method": "EM",
        "resolution": 3.0,
        "receptor_chain": "R",
        "ligand_resname": "SRO",   # serotonin
        "ligand_chain": "R",
        "ligand_resseq": 501,
        "is_validation": False,
    },
    "5-HT1B_4IAR": {
        "pdb": "4IAR.pdb",
        "subtype": "5-HT1B",
        "method": "X-ray",
        "resolution": 2.7,
        "receptor_chain": "A",
        "ligand_resname": "ERM",   # ergotamine
        "ligand_chain": "A",
        "ligand_resseq": None,     # will search
        "is_validation": False,
    },
    "5-HT1D_7E32": {
        "pdb": "7E32.pdb",
        "subtype": "5-HT1D",
        "method": "EM",
        "resolution": 2.9,
        "receptor_chain": "R",
        "ligand_resname": "SRO",   # serotonin
        "ligand_chain": "R",
        "ligand_resseq": None,
        "is_validation": False,
    },
    "5-HT1E_7E33": {
        "pdb": "7E33.pdb",
        "subtype": "5-HT1E",
        "method": "EM",
        "resolution": 2.9,
        "receptor_chain": "R",
        "ligand_resname": "HVU",
        "ligand_chain": "R",
        "ligand_resseq": 501,
        "is_validation": False,
    },
    "5-HT1F_7EXD": {
        "pdb": "7EXD.pdb",
        "subtype": "5-HT1F",
        "method": "EM",
        "resolution": 3.4,
        "receptor_chain": "R",
        "ligand_resname": "05X",   # lasmiditan
        "ligand_chain": "R",
        "ligand_resseq": None,
        "is_validation": False,
    },
    "5-HT2A_7WC6": {
        "pdb": "7WC6.pdb",
        "subtype": "5-HT2A",
        "method": "X-ray",
        "resolution": 2.6,
        "receptor_chain": "A",
        "ligand_resname": "7LD",   # LSD
        "ligand_chain": "A",
        "ligand_resseq": 1205,
        "is_validation": False,
    },
    "5-HT2B_4IB4": {
        "pdb": "4IB4.pdb",
        "subtype": "5-HT2B",
        "method": "X-ray",
        "resolution": 2.7,
        "receptor_chain": "A",
        "ligand_resname": "ERM",   # ergotamine
        "ligand_chain": "A",
        "ligand_resseq": None,
        "is_validation": False,
    },
    "5-HT2C_6BQH": {
        "pdb": "6BQH.pdb",
        "subtype": "5-HT2C",
        "method": "X-ray",
        "resolution": 2.7,
        "receptor_chain": "A",
        "ligand_resname": "E2J",   # ritanserin
        "ligand_chain": "A",
        "ligand_resseq": None,
        "is_validation": False,
    },
    "5-HT4_7XT8": {
        "pdb": "7XT8.pdb",
        "subtype": "5-HT4",
        "method": "EM",
        "resolution": 3.1,
        "receptor_chain": "R",
        "ligand_resname": "SRO",   # serotonin
        "ligand_chain": "R",
        "ligand_resseq": 408,
        "is_validation": False,
    },
    "5-HT6_7XTC": {
        "pdb": "7XTC.pdb",
        "subtype": "5-HT6",
        "method": "EM",
        "resolution": 3.2,
        "receptor_chain": "R",
        "ligand_resname": "8K3",
        "ligand_chain": "R",
        "ligand_resseq": 501,
        "is_validation": False,
    },
    "5-HT7_7XTA": {
        "pdb": "7XTA.pdb",
        "subtype": "5-HT7",
        "method": "EM",
        "resolution": 3.2,
        "receptor_chain": "R",
        "ligand_resname": "SRO",   # serotonin
        "ligand_chain": "R",
        "ligand_resseq": 401,
        "is_validation": False,
    },
    # --- Validation structures ---
    "5-HT2A_6WHA": {
        "pdb": "6WHA.pdb",
        "subtype": "5-HT2A",
        "method": "EM",
        "resolution": 3.36,
        "receptor_chain": "A",
        "ligand_resname": "U0G",   # LSD
        "ligand_chain": "A",
        "ligand_resseq": 501,
        "is_validation": True,
    },
    "5-HT2B_5TUD": {
        "pdb": "5TUD.pdb",
        "subtype": "5-HT2B",
        "method": "X-ray",
        "resolution": 3.0,
        "receptor_chain": "A",
        "ligand_resname": "ERM",   # ergotamine (LSD analog)
        "ligand_chain": "A",
        "ligand_resseq": 2001,
        "is_validation": True,
    },
}

# === Docking parameters (AutoDock Vina) ===
VINA_SEED = 42
VINA_EXHAUSTIVENESS = 32
VINA_N_POSES = 20
VINA_BOX_SIZE = [22.0, 22.0, 22.0]  # Angstroms

# === Output files ===
RESULTS_CSV = RESULTS_DIR / "docking_results.csv"
RESULTS_JSON = RESULTS_DIR / "docking_results.json"
BINDING_SITES_JSON = PREPARED_RECEPTORS_DIR / "binding_sites.json"
