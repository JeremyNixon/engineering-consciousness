#!/usr/bin/env python
"""TargetDiff: diffusion-based molecular generation for LSD optimization.

Attempts native TargetDiff diffusion sampling; falls back to bioisosteric
atom replacements using TargetDiff's atom vocabulary.

Output: molecular_optimization/TargetDiff/*.sdf
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
PDB_PATH = os.path.join(BASE_DIR, "receptors", "pdb_files", "6WHA.pdb")
OUTPUT_DIR = os.path.join(BASE_DIR, "molecular_optimization", "TargetDiff")
TARGETDIFF_DIR = os.path.join(BASE_DIR, "generative_molecular_design", "targetdiff")

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


def extract_pocket_pdb(pdb_path, center, radius=10.0, out_path=None):
    if out_path is None:
        out_path = os.path.join(OUTPUT_DIR, "pocket_6WHA.pdb")
    cx, cy, cz = center
    kept_residues = set()

    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                dist = ((x - cx)**2 + (y - cy)**2 + (z - cz)**2)**0.5
                if dist <= radius:
                    chain = line[21]
                    resnum = line[22:27].strip()
                    kept_residues.add((chain, resnum))

    with open(pdb_path) as f:
        lines = []
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21]
                resnum = line[22:27].strip()
                if (chain, resnum) in kept_residues:
                    lines.append(line)
            elif line.startswith("END"):
                lines.append(line)

    with open(out_path, "w") as f:
        f.writelines(lines)
    return out_path


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
# Native TargetDiff pipeline
# ---------------------------------------------------------------------------

def try_native():
    """Attempt native TargetDiff diffusion sampling."""
    print("[TargetDiff] Attempting native pipeline...")

    if not install_packages(["torch-geometric", "easydict", "pyyaml"]):
        print("[TargetDiff] Failed to install dependencies")
        return False
    install_packages(["gdown"])

    # Check/download checkpoint
    pretrained_dir = os.path.join(TARGETDIFF_DIR, "pretrained_models")
    os.makedirs(pretrained_dir, exist_ok=True)
    ckpt_path = os.path.join(pretrained_dir, "pretrained_diffusion.pt")

    if not os.path.isfile(ckpt_path):
        print("[TargetDiff] Downloading pretrained checkpoint...")
        try:
            import gdown
            gdown.download(
                "https://drive.google.com/uc?id=1-ftOFCBz4KXBGE0fhqAviYkVH2KPGA90",
                ckpt_path, quiet=False,
            )
        except Exception as e:
            print(f"[TargetDiff] Checkpoint download failed: {e}")
            return False

    if not os.path.isfile(ckpt_path):
        print("[TargetDiff] Checkpoint not available")
        return False

    sys.path.insert(0, TARGETDIFF_DIR)
    try:
        import torch
        from torch_geometric.transforms import Compose
        import utils.misc as misc
        import utils.transforms as trans
        from datasets.pl_data import ProteinLigandData, torchify_dict
        from models.molopt_score_model import ScorePosNet3D
        from scripts.sample_diffusion import sample_diffusion_ligand
        from utils.data import PDBProtein
        from utils import reconstruct
    except ImportError as e:
        print(f"[TargetDiff] Import failed: {e}")
        return False

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        center = get_lsd_center(SDF_PATH)
        pocket_path = extract_pocket_pdb(PDB_PATH, center, radius=10.0)

        # Load pocket
        pocket_dict = PDBProtein(pocket_path).to_dict_atom()
        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=torchify_dict(pocket_dict),
            ligand_dict={
                'element': torch.empty([0], dtype=torch.long),
                'pos': torch.empty([0, 3], dtype=torch.float),
                'atom_feature': torch.empty([0, 8], dtype=torch.float),
                'bond_index': torch.empty([2, 0], dtype=torch.long),
                'bond_type': torch.empty([0], dtype=torch.long),
            }
        )

        # Load checkpoint and model
        ckpt = torch.load(ckpt_path, map_location=device)
        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
        ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
        transform = Compose([protein_featurizer])
        data = transform(data)

        model = ScorePosNet3D(
            ckpt['config'].model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        ).to(device)
        model.load_state_dict(ckpt['model'], strict=False)

        # Sample
        num_samples = 100
        all_pred_pos, all_pred_v, *_ = sample_diffusion_ligand(
            model, data, num_samples,
            batch_size=50, device=device,
            num_steps=100,
            center_pos_mode='protein',
            sample_num_atoms='ref',
        )

        # Reconstruct molecules
        saved = 0
        for pred_pos, pred_v in zip(all_pred_pos, all_pred_v):
            pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode='add_aromatic')
            try:
                pred_aromatic = trans.is_aromatic_from_index(pred_v, mode='add_aromatic')
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
                if '.' not in smiles:
                    save_mol_sdf(mol, OUTPUT_DIR, saved)
                    saved += 1
            except Exception:
                continue

        print(f"[TargetDiff] Native: saved {saved} molecules")
        return saved > 0

    except Exception as e:
        print(f"[TargetDiff] Native pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Fallback: bioisosteric atom replacements
# ---------------------------------------------------------------------------

# TargetDiff atom vocabulary
TARGETDIFF_ATOMS = [6, 7, 8, 9, 15, 16, 17]  # C, N, O, F, P, S, Cl

# Bioisosteric replacement rules: (from_atomic_num, to_atomic_num, context)
BIOISOSTERE_RULES = [
    # Classic bioisosteric replacements
    (7, 8, "NH_to_O"),       # NH -> O
    (8, 7, "O_to_NH"),       # O -> NH
    (6, 7, "CH_to_N"),       # C -> N (in aromatic context)
    (7, 6, "N_to_CH"),       # N -> C
    (8, 16, "O_to_S"),       # O -> S
    (16, 8, "S_to_O"),       # S -> O
    (1, 9, "H_to_F"),        # H -> F (terminal)
    (6, 9, "CH3_to_F"),      # methyl -> F (approximate)
    (6, 17, "C_to_Cl"),      # C -> Cl (terminal)
    (7, 15, "N_to_P"),       # N -> P
]


def apply_bioisosteric_swap(mol, n_swaps=1, rng=None):
    """Apply random bioisosteric atom swaps to a molecule."""
    if rng is None:
        rng = random.Random()

    rwmol = Chem.RWMol(copy.deepcopy(mol))

    swaps_done = 0
    attempts = 0
    max_attempts = 50

    while swaps_done < n_swaps and attempts < max_attempts:
        attempts += 1

        # Pick a random atom
        atom_indices = list(range(rwmol.GetNumAtoms()))
        rng.shuffle(atom_indices)

        swapped = False
        for aidx in atom_indices:
            atom = rwmol.GetAtomWithIdx(aidx)
            atomic_num = atom.GetAtomicNum()

            # Find applicable rules
            applicable = [r for r in BIOISOSTERE_RULES if r[0] == atomic_num]
            if not applicable:
                continue

            rule = rng.choice(applicable)
            from_num, to_num, context = rule

            # Validate swap context
            is_ring = atom.IsInRing()
            is_aromatic = atom.GetIsAromatic()
            n_neighbors = atom.GetDegree()

            # Don't break ring systems with incompatible valence
            if is_ring:
                # Only allow C<->N in rings (classic ring bioisostere)
                if not ((from_num == 6 and to_num == 7) or
                        (from_num == 7 and to_num == 6)):
                    continue

            # Don't swap atoms with too many bonds for target element
            try:
                max_valence = {6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1}
                if n_neighbors > max_valence.get(to_num, 4):
                    continue
            except KeyError:
                continue

            # Apply swap
            atom.SetAtomicNum(to_num)
            if is_aromatic and to_num in (6, 7):
                atom.SetIsAromatic(True)

            # Adjust hydrogen count
            try:
                Chem.SanitizeMol(rwmol)
                swapped = True
                swaps_done += 1
                break
            except Exception:
                # Revert
                atom.SetAtomicNum(from_num)
                if is_aromatic:
                    atom.SetIsAromatic(True)
                continue

        if not swapped:
            break

    # Final sanitization
    try:
        Chem.SanitizeMol(rwmol)
        smiles = Chem.MolToSmiles(rwmol)
        result = Chem.MolFromSmiles(smiles)
        return result
    except Exception:
        return None


def run_fallback():
    """Generate LSD variants via bioisosteric replacements."""
    print("[TargetDiff] Running fallback: bioisosteric atom replacements")

    mol = get_lsd_mol(SDF_PATH)
    mol_noH = Chem.RemoveHs(mol)
    center = get_lsd_center(SDF_PATH)

    # Extract pocket for reference
    extract_pocket_pdb(PDB_PATH, center, radius=10.0)

    random.seed(42)
    np.random.seed(42)

    generated = {}  # smiles -> mol
    idx = 0

    lsd_smi = Chem.MolToSmiles(mol_noH)
    print(f"  Starting from LSD: {lsd_smi}")

    # --- Part A: Single bioisosteric swaps ---
    print("  Part A: Single bioisosteric swaps...")
    rng = random.Random(42)
    attempts = 0
    while idx < 40 and attempts < 500:
        attempts += 1
        variant = apply_bioisosteric_swap(mol_noH, n_swaps=1, rng=rng)
        if variant is None:
            continue
        smi = Chem.MolToSmiles(variant)
        if smi == lsd_smi or smi in generated:
            continue
        # MW filter: within 50% of LSD MW
        lsd_mw = Descriptors.ExactMolWt(mol_noH)
        var_mw = Descriptors.ExactMolWt(variant)
        if abs(var_mw - lsd_mw) > lsd_mw * 0.5:
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

    # --- Part B: Double bioisosteric swaps ---
    print("  Part B: Double bioisosteric swaps...")
    attempts = 0
    while idx < 70 and attempts < 500:
        attempts += 1
        variant = apply_bioisosteric_swap(mol_noH, n_swaps=2, rng=rng)
        if variant is None:
            continue
        smi = Chem.MolToSmiles(variant)
        if smi == lsd_smi or smi in generated:
            continue
        lsd_mw = Descriptors.ExactMolWt(mol_noH)
        var_mw = Descriptors.ExactMolWt(variant)
        if abs(var_mw - lsd_mw) > lsd_mw * 0.5:
            continue
        generated[smi] = variant
        variant_3d = Chem.AddHs(variant)
        params = AllChem.ETKDGv3()
        params.randomSeed = idx + 100
        res = AllChem.EmbedMolecule(variant_3d, params)
        if res != 0:
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(variant_3d, params)
        if res == 0:
            AllChem.MMFFOptimizeMolecule(variant_3d, maxIters=500)
            variant_3d = Chem.RemoveHs(variant_3d)
            save_mol_sdf(variant_3d, OUTPUT_DIR, idx)
            idx += 1

    # --- Part C: Halogenation variants (F, Cl at various positions) ---
    print("  Part C: Targeted halogenation variants...")

    lsd_base = "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(C)C1"

    halogen_variants = [
        # Fluorinated variants
        "CCN(CC)C(=O)[C@@H]1C=C2c3cc(F)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(C)C1F",
        "CCN(CC)C(=O)[C@@H]1C=C2c3c(F)ccc4[nH]cc(c34)C[C@H]2N(C)C1",
        "CCN(CC)C(=O)[C@@H]1C=C2c3cc(F)c(F)c4[nH]cc(c34)C[C@H]2N(C)C1",
        "CCN(CC)C(=O)[C@@H]1C=C2c3c(F)cc(F)c4[nH]cc(c34)C[C@H]2N(C)C1",
        # Chlorinated
        "CCN(CC)C(=O)[C@@H]1C=C2c3cc(Cl)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        "CCN(CC)C(=O)[C@@H]1C=C2c3c(Cl)ccc4[nH]cc(c34)C[C@H]2N(C)C1",
        # Mixed halogens
        "CCN(CC)C(=O)[C@@H]1C=C2c3cc(F)cc4[nH]cc(c34)C[C@H]2N(C)C1Cl",
        "CCN(CC)C(=O)[C@@H]1C=C2c3cc(Cl)cc4[nH]cc(c34)C[C@H]2N(C)C1F",
    ]

    for smi in halogen_variants:
        mol_v = Chem.MolFromSmiles(smi)
        if mol_v is None:
            continue
        can_smi = Chem.MolToSmiles(mol_v)
        if can_smi in generated or can_smi == lsd_smi:
            continue
        mol_v = Chem.AddHs(mol_v)
        params = AllChem.ETKDGv3()
        params.randomSeed = idx + 200
        res = AllChem.EmbedMolecule(mol_v, params)
        if res != 0:
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(mol_v, params)
        if res == 0:
            AllChem.MMFFOptimizeMolecule(mol_v, maxIters=500)
            mol_v = Chem.RemoveHs(mol_v)
            generated[can_smi] = mol_v
            save_mol_sdf(mol_v, OUTPUT_DIR, idx)
            idx += 1

    # --- Part D: Heteroatom swaps in ergoline core ---
    print("  Part D: Heteroatom swaps in ergoline core...")

    heteroatom_variants = [
        # O replacing NH in indole
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4occ(c34)C[C@H]2N(C)C1",
        # S replacing NH in indole
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4[s]cc(c34)C[C@H]2N(C)C1",
        # Aza variants (N in benzene ring)
        "CCN(CC)C(=O)[C@@H]1C=C2c3ccnc4[nH]cc(c34)C[C@H]2N(C)C1",
        "CCN(CC)C(=O)[C@@H]1C=C2c3cncc4[nH]cc(c34)C[C@H]2N(C)C1",
        "CCN(CC)C(=O)[C@@H]1C=C2c3ncccc4[nH]cc(c34)C[C@H]2N(C)C1",
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccnc34[nH]cc4C[C@H]2N(C)C1",
        # Thioamide
        "CCN(CC)C(=S)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(C)C1",
        # Amidine (C=N instead of C=O)
        "CCN(CC)C(=N)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(C)C1",
        # P replacing N in piperidine
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2P(C)C1",
    ]

    for smi in heteroatom_variants:
        mol_v = Chem.MolFromSmiles(smi)
        if mol_v is None:
            continue
        can_smi = Chem.MolToSmiles(mol_v)
        if can_smi in generated or can_smi == lsd_smi:
            continue
        mol_v = Chem.AddHs(mol_v)
        params = AllChem.ETKDGv3()
        params.randomSeed = idx + 300
        res = AllChem.EmbedMolecule(mol_v, params)
        if res != 0:
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(mol_v, params)
        if res == 0:
            AllChem.MMFFOptimizeMolecule(mol_v, maxIters=500)
            mol_v = Chem.RemoveHs(mol_v)
            generated[can_smi] = mol_v
            save_mol_sdf(mol_v, OUTPUT_DIR, idx)
            idx += 1

    # --- Part E: Additional swaps on existing variants to reach target ---
    print("  Part E: Iterative bioisosteric swaps on variants...")
    existing_smiles = list(generated.keys())
    rng2 = random.Random(123)
    attempts = 0
    while idx < 100 and attempts < 1000:
        attempts += 1
        # Pick a random existing variant as starting point
        base_smi = rng2.choice(existing_smiles)
        base_mol = Chem.MolFromSmiles(base_smi)
        if base_mol is None:
            continue
        variant = apply_bioisosteric_swap(base_mol, n_swaps=1, rng=rng2)
        if variant is None:
            continue
        smi = Chem.MolToSmiles(variant)
        if smi in generated or smi == lsd_smi:
            continue
        # MW filter
        var_mw = Descriptors.ExactMolWt(variant)
        lsd_mw = Descriptors.ExactMolWt(mol_noH)
        if abs(var_mw - lsd_mw) > lsd_mw * 0.7:
            continue
        generated[smi] = variant
        existing_smiles.append(smi)
        variant_3d = Chem.AddHs(variant)
        params = AllChem.ETKDGv3()
        params.randomSeed = idx + 500
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
    print("TargetDiff: Diffusion-Based Molecular Generation")
    print("=" * 60)
    print(f"  LSD SDF:    {SDF_PATH}")
    print(f"  Receptor:   {PDB_PATH}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print()

    center = get_lsd_center(SDF_PATH)
    print(f"  LSD centroid: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
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
    print(f"TargetDiff Summary")
    print(f"  Method:           {method}")
    print(f"  Total SDF files:  {len(sdf_files)}")
    print(f"  Valid molecules:  {valid}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
