#!/usr/bin/env python3
"""
Prepare receptors for docking: PDB → cleaned PDB → PDBQT.

For each receptor defined in config.RECEPTORS:
1. Parse PDB with BioPython, extract receptor chain only
   (removes G-proteins, antibodies, nanobodies, water, ions)
2. Extract binding site center from co-crystallized ligand HETATM coordinates
3. Convert cleaned PDB → PDBQT via OpenBabel pybel (add hydrogens, Gasteiger charges)
4. Save PDBQT to prepared/receptors/ and binding sites to binding_sites.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from Bio import PDB as BioPDB
from Bio.PDB import PDBIO, Select
from openbabel import pybel

from config import (
    RECEPTORS, PDB_DIR, PREPARED_RECEPTORS_DIR,
    BINDING_SITES_JSON, VINA_BOX_SIZE,
)

warnings.filterwarnings("ignore", category=BioPDB.PDBExceptions.PDBConstructionWarning)


class ChainSelect(Select):
    """Select only ATOM records from specified chain, excluding water and ions."""

    WATER_RESNAMES = {"HOH", "WAT", "TIP", "TIP3", "SPC"}
    ION_RESNAMES = {
        "NA", "CL", "MG", "ZN", "CA", "FE", "MN", "CO", "NI", "CU",
        "K", "BR", "IOD", "SO4", "PO4", "ACT", "GOL", "EDO", "DMS",
    }

    def __init__(self, chain_id):
        super().__init__()
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id

    def accept_residue(self, residue):
        resname = residue.get_resname().strip()
        hetflag = residue.get_id()[0]
        # Keep standard amino acids (ATOM records) only
        if hetflag == " ":
            return True
        # Exclude water and common ions/solvents
        if resname in self.WATER_RESNAMES or resname in self.ION_RESNAMES:
            return False
        return False  # Exclude all other HETATM (ligands, lipids, etc.)


def extract_binding_site(pdb_path: str, receptor_info: dict) -> dict | None:
    """Extract binding site center from co-crystallized ligand coordinates."""
    parser = BioPDB.PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", pdb_path)

    ligand_resname = receptor_info["ligand_resname"]
    ligand_chain = receptor_info["ligand_chain"]
    ligand_resseq = receptor_info.get("ligand_resseq")

    coords = []
    for model in structure:
        for chain in model:
            if chain.get_id() != ligand_chain:
                continue
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname != ligand_resname:
                    continue
                # If specific resseq given, match it
                if ligand_resseq is not None and residue.get_id()[1] != ligand_resseq:
                    continue
                for atom in residue:
                    coords.append(atom.get_vector().get_array())

    if not coords:
        return None

    coords = np.array(coords)
    center = coords.mean(axis=0)
    return {
        "center_x": float(round(center[0], 3)),
        "center_y": float(round(center[1], 3)),
        "center_z": float(round(center[2], 3)),
        "size_x": VINA_BOX_SIZE[0],
        "size_y": VINA_BOX_SIZE[1],
        "size_z": VINA_BOX_SIZE[2],
        "n_ligand_atoms": len(coords),
    }


def clean_pdb(pdb_path: str, chain_id: str, output_path: str) -> bool:
    """Extract receptor chain from PDB, removing non-protein atoms."""
    parser = BioPDB.PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", pdb_path)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path), ChainSelect(chain_id))
    return True


def pdb_to_pdbqt(pdb_path: str, pdbqt_path: str) -> bool:
    """Convert cleaned PDB to rigid receptor PDBQT using OpenBabel.

    OpenBabel writes PDBQT with ROOT/BRANCH tags (ligand format).
    For rigid receptors, Vina expects only ATOM/HETATM lines with
    Gasteiger charges — no torsion tree. We strip those tags.
    """
    mol = next(pybel.readfile("pdb", pdb_path))
    mol.addh()
    mol.calccharges("gasteiger")

    # Write to temp ligand-style PDBQT first
    temp_path = pdbqt_path + ".tmp"
    mol.write("pdbqt", temp_path, overwrite=True)

    # Strip ROOT/BRANCH/TORSDOF lines to make rigid receptor PDBQT
    # Note: OpenBabel may write "BRANCH10001001" without spaces
    skip_prefixes = ("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")
    with open(temp_path) as f_in, open(pdbqt_path, "w") as f_out:
        for line in f_in:
            stripped = line.strip()
            if not stripped:
                continue
            if any(stripped.startswith(p) for p in skip_prefixes):
                continue
            # Skip REMARK lines about torsions
            if line.startswith("REMARK") and "torsion" in line.lower():
                continue
            f_out.write(line)

    Path(temp_path).unlink(missing_ok=True)
    return True


def prepare_receptor(name: str, info: dict, binding_sites: dict) -> bool:
    """Full receptor preparation pipeline for one structure."""
    pdb_path = PDB_DIR / info["pdb"]
    clean_pdb_path = PREPARED_RECEPTORS_DIR / f"{name}_clean.pdb"
    pdbqt_path = PREPARED_RECEPTORS_DIR / f"{name}.pdbqt"

    if pdbqt_path.exists() and name in binding_sites:
        print(f"  [SKIP] {name}: already prepared")
        return True

    if not pdb_path.exists():
        print(f"  [ERROR] {name}: PDB not found at {pdb_path}")
        return False

    # Step 1: Extract binding site center from co-crystallized ligand
    site = extract_binding_site(str(pdb_path), info)
    if site is None:
        print(f"  [ERROR] {name}: Could not find ligand {info['ligand_resname']} "
              f"on chain {info['ligand_chain']} in {info['pdb']}")
        return False

    binding_sites[name] = site
    print(f"  [SITE] {name}: center=({site['center_x']:.1f}, {site['center_y']:.1f}, "
          f"{site['center_z']:.1f}), {site['n_ligand_atoms']} ligand atoms")

    # Step 2: Clean PDB - extract receptor chain only
    try:
        clean_pdb(str(pdb_path), info["receptor_chain"], str(clean_pdb_path))
    except Exception as e:
        print(f"  [ERROR] {name}: PDB cleaning failed: {e}")
        return False

    # Step 3: Convert to PDBQT
    try:
        pdb_to_pdbqt(str(clean_pdb_path), str(pdbqt_path))
    except Exception as e:
        print(f"  [ERROR] {name}: PDBQT conversion failed: {e}")
        return False

    # Verify output
    if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        print(f"  [ERROR] {name}: PDBQT file empty or not created")
        return False

    print(f"  [OK] {name}: {info['subtype']} ({info['method']}, {info['resolution']}Å) → {pdbqt_path.name}")
    return True


def main():
    print("=" * 60)
    print("RECEPTOR PREPARATION: PDB → PDBQT")
    print("=" * 60)
    print(f"Input directory:  {PDB_DIR}")
    print(f"Output directory: {PREPARED_RECEPTORS_DIR}")
    print(f"Receptors: {len(RECEPTORS)}")
    print()

    # Load existing binding sites if any
    binding_sites = {}
    if BINDING_SITES_JSON.exists():
        binding_sites = json.loads(BINDING_SITES_JSON.read_text())

    success, fail = 0, 0
    for name, info in RECEPTORS.items():
        ok = prepare_receptor(name, info, binding_sites)
        if ok:
            success += 1
        else:
            fail += 1

    # Save binding sites
    BINDING_SITES_JSON.write_text(json.dumps(binding_sites, indent=2))
    print()
    print(f"Binding sites saved to: {BINDING_SITES_JSON}")
    print(f"Results: {success} succeeded, {fail} failed out of {len(RECEPTORS)}")

    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
