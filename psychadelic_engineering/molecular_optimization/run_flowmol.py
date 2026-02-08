#!/usr/bin/env python
"""FlowMol: flow matching molecular generation for LSD optimization.

Attempts native FlowMol sampling (most likely to succeed natively);
falls back to random molecular mutations starting from LSD.

Output: molecular_optimization/FlowMol/*.sdf
"""

import os
import sys
import subprocess
import random
import copy
import numpy as np

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Descriptors

rdBase.DisableLog("rdApp.error")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDF_PATH = os.path.join(BASE_DIR, "molecule_data", "sdf_files", "LSD_5761.sdf")
OUTPUT_DIR = os.path.join(BASE_DIR, "molecular_optimization", "FlowMol")
FLOWMOL_DIR = os.path.join(BASE_DIR, "generative_molecular_design", "FlowMol")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_lsd_mol(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for mol in suppl:
        if mol is not None:
            return mol
    raise RuntimeError(f"Could not load molecule from {sdf_path}")


def get_lsd_center(sdf_path):
    mol = get_lsd_mol(sdf_path)
    mol_noH = Chem.RemoveHs(mol)
    conf = mol_noH.GetConformer()
    positions = conf.GetPositions()
    return positions.mean(axis=0).tolist()


def save_mol_sdf(mol, output_dir, idx):
    path = os.path.join(output_dir, f"{idx:03d}.sdf")
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.close()
    return path


def install_packages(pkgs):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + pkgs,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


# ---------------------------------------------------------------------------
# Native FlowMol pipeline
# ---------------------------------------------------------------------------

def try_native():
    """Attempt native FlowMol sampling."""
    print("[FlowMol] Attempting native pipeline...")

    # Install FlowMol dependencies
    if not install_packages(["pytorch-lightning"]):
        print("[FlowMol] Failed to install pytorch-lightning")
        # Continue — might still work

    # Try installing FlowMol package itself
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "-e", FLOWMOL_DIR],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        print("[FlowMol] pip install -e FlowMol/ failed, trying import anyway...")

    # Also try dgl and torch-geometric
    install_packages(["torch-geometric"])
    install_packages(["dgl"])

    try:
        import flowmol
        model = flowmol.load_pretrained('flowmol3')
        print("[FlowMol] Model loaded successfully!")
    except Exception as e:
        print(f"[FlowMol] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # Sample molecules with n_atoms=24 (matching LSD heavy atom count)
        lsd_mol = get_lsd_mol(SDF_PATH)
        lsd_noH = Chem.RemoveHs(lsd_mol)
        n_atoms = lsd_noH.GetNumHeavyAtoms()
        print(f"[FlowMol] LSD has {n_atoms} heavy atoms, sampling with n_atoms={n_atoms}")

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        # Sample 100 molecules
        print("[FlowMol] Sampling 100 molecules with 250 timesteps...")
        sampled_mols = model.sample(
            n_samples=100,
            n_atoms=n_atoms,
            n_timesteps=250,
        )

        saved = 0
        for i, smol in enumerate(sampled_mols):
            try:
                rdmol = smol.rdkit_mol
                if rdmol is None:
                    continue
                smi = Chem.MolToSmiles(rdmol)
                if smi is None or '.' in smi:
                    continue
                # Ensure 3D coords
                if rdmol.GetNumConformers() == 0:
                    rdmol = Chem.AddHs(rdmol)
                    AllChem.EmbedMolecule(rdmol, AllChem.ETKDGv3())
                    AllChem.MMFFOptimizeMolecule(rdmol, maxIters=500)
                    rdmol = Chem.RemoveHs(rdmol)
                save_mol_sdf(rdmol, OUTPUT_DIR, saved)
                saved += 1
            except Exception:
                continue

        print(f"[FlowMol] Native: saved {saved} molecules")
        return saved > 0

    except Exception as e:
        print(f"[FlowMol] Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Fallback: random molecular mutations
# ---------------------------------------------------------------------------

def atom_substitution(mol, rng):
    """Replace a non-ring carbon with N/O, or N with O, etc."""
    rwmol = Chem.RWMol(copy.deepcopy(mol))
    atoms = list(range(rwmol.GetNumAtoms()))
    rng.shuffle(atoms)

    for aidx in atoms:
        atom = rwmol.GetAtomWithIdx(aidx)
        an = atom.GetAtomicNum()
        is_ring = atom.IsInRing()
        degree = atom.GetDegree()

        # Non-ring C -> N (if degree <= 3)
        if an == 6 and not is_ring and degree <= 3:
            atom.SetAtomicNum(7)
            try:
                Chem.SanitizeMol(rwmol)
                return Chem.MolFromSmiles(Chem.MolToSmiles(rwmol))
            except Exception:
                atom.SetAtomicNum(6)

        # Non-ring C -> O (if degree <= 2)
        if an == 6 and not is_ring and degree <= 2:
            atom.SetAtomicNum(8)
            try:
                Chem.SanitizeMol(rwmol)
                return Chem.MolFromSmiles(Chem.MolToSmiles(rwmol))
            except Exception:
                atom.SetAtomicNum(6)

        # N -> O (if degree <= 2)
        if an == 7 and not is_ring and degree <= 2:
            atom.SetAtomicNum(8)
            try:
                Chem.SanitizeMol(rwmol)
                return Chem.MolFromSmiles(Chem.MolToSmiles(rwmol))
            except Exception:
                atom.SetAtomicNum(7)

        # O -> S
        if an == 8 and not is_ring and degree <= 2:
            atom.SetAtomicNum(16)
            try:
                Chem.SanitizeMol(rwmol)
                return Chem.MolFromSmiles(Chem.MolToSmiles(rwmol))
            except Exception:
                atom.SetAtomicNum(8)

    return None


def atom_addition(mol, rng):
    """Insert a C/N/O atom at a random bond."""
    rwmol = Chem.RWMol(copy.deepcopy(mol))
    bonds = list(range(rwmol.GetNumBonds()))
    rng.shuffle(bonds)

    for bidx in bonds:
        bond = rwmol.GetBondWithIdx(bidx)
        if bond.IsInRing():
            continue
        if bond.GetBondTypeAsDouble() != 1.0:
            continue

        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()

        # Add new atom
        new_atom_num = rng.choice([6, 6, 6, 7, 8])  # bias toward C
        new_idx = rwmol.AddAtom(Chem.Atom(new_atom_num))

        # Remove old bond, add two new bonds through new atom
        rwmol.RemoveBond(begin, end)
        rwmol.AddBond(begin, new_idx, Chem.BondType.SINGLE)
        rwmol.AddBond(new_idx, end, Chem.BondType.SINGLE)

        try:
            Chem.SanitizeMol(rwmol)
            return Chem.MolFromSmiles(Chem.MolToSmiles(rwmol))
        except Exception:
            return None

    return None


def atom_deletion(mol, rng):
    """Remove a terminal (degree-1) atom."""
    rwmol = Chem.RWMol(copy.deepcopy(mol))
    terminal_atoms = []

    for aidx in range(rwmol.GetNumAtoms()):
        atom = rwmol.GetAtomWithIdx(aidx)
        if atom.GetDegree() == 1 and not atom.IsInRing():
            terminal_atoms.append(aidx)

    if not terminal_atoms:
        return None

    rng.shuffle(terminal_atoms)
    to_remove = terminal_atoms[0]

    rwmol.RemoveAtom(to_remove)
    try:
        Chem.SanitizeMol(rwmol)
        result = Chem.MolFromSmiles(Chem.MolToSmiles(rwmol))
        if result and result.GetNumHeavyAtoms() >= 10:
            return result
    except Exception:
        pass
    return None


def bond_type_change(mol, rng):
    """Change single<->double bond at eligible positions."""
    rwmol = Chem.RWMol(copy.deepcopy(mol))
    bonds = list(range(rwmol.GetNumBonds()))
    rng.shuffle(bonds)

    for bidx in bonds:
        bond = rwmol.GetBondWithIdx(bidx)
        if bond.IsInRing():
            continue
        if bond.GetIsAromatic():
            continue

        bt = bond.GetBondType()
        begin_atom = rwmol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        end_atom = rwmol.GetAtomWithIdx(bond.GetEndAtomIdx())

        if bt == Chem.BondType.SINGLE:
            # Try to make double
            if begin_atom.GetDegree() <= 3 and end_atom.GetDegree() <= 3:
                bond.SetBondType(Chem.BondType.DOUBLE)
                try:
                    Chem.SanitizeMol(rwmol)
                    return Chem.MolFromSmiles(Chem.MolToSmiles(rwmol))
                except Exception:
                    bond.SetBondType(Chem.BondType.SINGLE)

        elif bt == Chem.BondType.DOUBLE:
            bond.SetBondType(Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(rwmol)
                return Chem.MolFromSmiles(Chem.MolToSmiles(rwmol))
            except Exception:
                bond.SetBondType(Chem.BondType.DOUBLE)

    return None


MUTATION_OPS = [atom_substitution, atom_addition, atom_deletion, bond_type_change]


def mutate_molecule(mol, n_mutations=1, rng=None):
    """Apply n random mutations to a molecule."""
    if rng is None:
        rng = random.Random()

    current = mol
    for _ in range(n_mutations):
        op = rng.choice(MUTATION_OPS)
        result = op(current, rng)
        if result is not None:
            current = result
        else:
            # Try another op
            ops = [o for o in MUTATION_OPS if o != op]
            rng.shuffle(ops)
            for alt_op in ops:
                result = alt_op(current, rng)
                if result is not None:
                    current = result
                    break

    if Chem.MolToSmiles(current) == Chem.MolToSmiles(mol):
        return None
    return current


def run_fallback():
    """Generate LSD variants via random molecular mutations."""
    print("[FlowMol] Running fallback: random molecular mutations")

    mol = get_lsd_mol(SDF_PATH)
    mol_noH = Chem.RemoveHs(mol)
    lsd_smi = Chem.MolToSmiles(mol_noH)
    lsd_mw = Descriptors.ExactMolWt(mol_noH)

    random.seed(42)
    np.random.seed(42)
    rng = random.Random(42)

    generated = {}  # smiles -> mol
    idx = 0

    # --- Part A: Single mutations ---
    print("  Part A: Single mutations from LSD...")
    attempts = 0
    while idx < 35 and attempts < 500:
        attempts += 1
        variant = mutate_molecule(mol_noH, n_mutations=1, rng=rng)
        if variant is None:
            continue
        smi = Chem.MolToSmiles(variant)
        if smi == lsd_smi or smi in generated:
            continue
        # MW filter
        mw = Descriptors.ExactMolWt(variant)
        if abs(mw - lsd_mw) > lsd_mw * 0.5:
            continue
        generated[smi] = variant
        # Embed 3D
        variant_3d = Chem.AddHs(variant)
        params = AllChem.ETKDGv3()
        params.randomSeed = idx + 42
        res = AllChem.EmbedMolecule(variant_3d, params)
        if res != 0:
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(variant_3d, params)
        if res == 0:
            AllChem.MMFFOptimizeMolecule(variant_3d, maxIters=500)
            variant_3d = Chem.RemoveHs(variant_3d)
            save_mol_sdf(variant_3d, OUTPUT_DIR, idx)
            idx += 1

    # --- Part B: Double mutations ---
    print("  Part B: Double mutations from LSD...")
    attempts = 0
    while idx < 65 and attempts < 500:
        attempts += 1
        variant = mutate_molecule(mol_noH, n_mutations=2, rng=rng)
        if variant is None:
            continue
        smi = Chem.MolToSmiles(variant)
        if smi == lsd_smi or smi in generated:
            continue
        mw = Descriptors.ExactMolWt(variant)
        if abs(mw - lsd_mw) > lsd_mw * 0.6:
            continue
        generated[smi] = variant
        variant_3d = Chem.AddHs(variant)
        params = AllChem.ETKDGv3()
        params.randomSeed = idx + 200
        res = AllChem.EmbedMolecule(variant_3d, params)
        if res != 0:
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(variant_3d, params)
        if res == 0:
            AllChem.MMFFOptimizeMolecule(variant_3d, maxIters=500)
            variant_3d = Chem.RemoveHs(variant_3d)
            save_mol_sdf(variant_3d, OUTPUT_DIR, idx)
            idx += 1

    # --- Part C: Triple mutations ---
    print("  Part C: Triple mutations from LSD...")
    attempts = 0
    while idx < 85 and attempts < 500:
        attempts += 1
        variant = mutate_molecule(mol_noH, n_mutations=3, rng=rng)
        if variant is None:
            continue
        smi = Chem.MolToSmiles(variant)
        if smi == lsd_smi or smi in generated:
            continue
        mw = Descriptors.ExactMolWt(variant)
        if abs(mw - lsd_mw) > lsd_mw * 0.7:
            continue
        generated[smi] = variant
        variant_3d = Chem.AddHs(variant)
        params = AllChem.ETKDGv3()
        params.randomSeed = idx + 400
        res = AllChem.EmbedMolecule(variant_3d, params)
        if res != 0:
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(variant_3d, params)
        if res == 0:
            AllChem.MMFFOptimizeMolecule(variant_3d, maxIters=500)
            variant_3d = Chem.RemoveHs(variant_3d)
            save_mol_sdf(variant_3d, OUTPUT_DIR, idx)
            idx += 1

    # --- Part D: Iterative mutations on generated variants ---
    print("  Part D: Iterative mutations on generated variants...")
    existing_smiles = list(generated.keys())
    rng2 = random.Random(789)
    attempts = 0
    while idx < 100 and attempts < 1000:
        attempts += 1
        base_smi = rng2.choice(existing_smiles)
        base_mol = Chem.MolFromSmiles(base_smi)
        if base_mol is None:
            continue
        n_mut = rng2.choice([1, 2])
        variant = mutate_molecule(base_mol, n_mutations=n_mut, rng=rng2)
        if variant is None:
            continue
        smi = Chem.MolToSmiles(variant)
        if smi in generated or smi == lsd_smi:
            continue
        mw = Descriptors.ExactMolWt(variant)
        if abs(mw - lsd_mw) > lsd_mw * 0.8:
            continue
        hac = variant.GetNumHeavyAtoms()
        if hac < 10 or hac > 40:
            continue
        generated[smi] = variant
        existing_smiles.append(smi)
        variant_3d = Chem.AddHs(variant)
        params = AllChem.ETKDGv3()
        params.randomSeed = idx + 600
        res = AllChem.EmbedMolecule(variant_3d, params)
        if res != 0:
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(variant_3d, params)
        if res == 0:
            AllChem.MMFFOptimizeMolecule(variant_3d, maxIters=500)
            variant_3d = Chem.RemoveHs(variant_3d)
            save_mol_sdf(variant_3d, OUTPUT_DIR, idx)
            idx += 1

    return idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("FlowMol: Flow Matching Molecular Generation")
    print("=" * 60)
    print(f"  LSD SDF:    {SDF_PATH}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print()

    method = "native"
    success = try_native()

    if not success:
        method = "fallback"
        count = run_fallback()
    else:
        count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".sdf")])

    # Summary
    sdf_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".sdf")]
    valid = 0
    for f in sdf_files:
        suppl = Chem.SDMolSupplier(os.path.join(OUTPUT_DIR, f), removeHs=True)
        for mol in suppl:
            if mol is not None:
                valid += 1

    print()
    print("=" * 60)
    print(f"FlowMol Summary")
    print(f"  Method:           {method}")
    print(f"  Total SDF files:  {len(sdf_files)}")
    print(f"  Valid molecules:  {valid}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
