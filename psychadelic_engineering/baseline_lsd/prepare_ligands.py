#!/usr/bin/env python3
"""
Prepare ligands for docking: SDF → PDBQT via RDKit + Meeko.

For each molecule defined in config.MOLECULES:
1. Load SDF with RDKit SDMolSupplier
2. Add hydrogens, generate/optimize 3D coords with MMFF94
3. Convert to PDBQT via Meeko (MoleculePreparation → PDBQTWriterLegacy)
4. Save to prepared/ligands/{name}.pdbqt
"""

import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from meeko import MoleculePreparation, PDBQTWriterLegacy

from config import MOLECULES, SDF_DIR, PREPARED_LIGANDS_DIR


def prepare_ligand(name: str, info: dict) -> bool:
    """Prepare a single ligand from SDF to PDBQT."""
    sdf_path = SDF_DIR / info["sdf"]
    out_path = PREPARED_LIGANDS_DIR / f"{name}.pdbqt"

    if out_path.exists():
        print(f"  [SKIP] {name}: {out_path.name} already exists")
        return True

    if not sdf_path.exists():
        print(f"  [ERROR] {name}: SDF not found at {sdf_path}")
        return False

    # Load molecule from SDF
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next(iter(supplier), None)
    if mol is None:
        print(f"  [ERROR] {name}: Failed to parse SDF")
        return False

    # Remove existing hydrogens and re-add for clean state
    mol = Chem.RemoveHs(mol)
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates if not present, or optimize existing ones
    if mol.GetNumConformers() == 0:
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result == -1:
            print(f"  [ERROR] {name}: 3D embedding failed")
            return False

    # MMFF94 force field optimization
    try:
        opt_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        if opt_result == -1:
            print(f"  [WARN] {name}: MMFF optimization did not converge, using current geometry")
    except Exception as e:
        print(f"  [WARN] {name}: MMFF optimization failed ({e}), trying UFF")
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
        except Exception as e2:
            print(f"  [WARN] {name}: UFF also failed ({e2}), using unoptimized geometry")

    # Sanity check molecular weight
    mw = Descriptors.MolWt(mol)
    expected_mw = info.get("mw", 0)
    if expected_mw and abs(mw - expected_mw) > 50:
        print(f"  [WARN] {name}: MW mismatch — computed {mw:.1f}, expected ~{expected_mw}")

    # Convert to PDBQT via Meeko
    try:
        preparator = MoleculePreparation()
        mol_setup_list = preparator.prepare(mol)
        if not mol_setup_list:
            print(f"  [ERROR] {name}: Meeko preparation returned empty list")
            return False

        mol_setup = mol_setup_list[0]
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setup)
        if not is_ok:
            print(f"  [ERROR] {name}: PDBQT writing failed: {error_msg}")
            return False

        out_path.write_text(pdbqt_string)
        print(f"  [OK] {name}: MW={mw:.1f}, atoms={mol.GetNumAtoms()}, saved to {out_path.name}")
        return True

    except Exception as e:
        print(f"  [ERROR] {name}: Meeko conversion failed: {e}")
        return False


def main():
    print("=" * 60)
    print("LIGAND PREPARATION: SDF → PDBQT")
    print("=" * 60)
    print(f"Input directory:  {SDF_DIR}")
    print(f"Output directory: {PREPARED_LIGANDS_DIR}")
    print(f"Molecules: {len(MOLECULES)}")
    print()

    success, fail = 0, 0
    for name, info in MOLECULES.items():
        ok = prepare_ligand(name, info)
        if ok:
            success += 1
        else:
            fail += 1

    print()
    print(f"Results: {success} succeeded, {fail} failed out of {len(MOLECULES)}")
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
