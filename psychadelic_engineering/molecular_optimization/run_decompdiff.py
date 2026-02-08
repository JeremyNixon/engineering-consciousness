#!/usr/bin/env python
"""DecompDiff: decomposed diffusion for LSD optimization.

Attempts native DecompDiff sampling; falls back to BRICS decomposition
and Murcko scaffold substitution (mirroring DecompDiff's scaffold+arms
philosophy).

Output: molecular_optimization/DecompDiff/*.sdf
"""

import os
import sys
import subprocess
import random
import itertools
import numpy as np

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Descriptors, BRICS, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold

rdBase.DisableLog("rdApp.error")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDF_PATH = os.path.join(BASE_DIR, "molecule_data", "sdf_files", "LSD_5761.sdf")
PDB_PATH = os.path.join(BASE_DIR, "receptors", "pdb_files", "6WHA.pdb")
OUTPUT_DIR = os.path.join(BASE_DIR, "molecular_optimization", "DecompDiff")
DECOMPDIFF_DIR = os.path.join(BASE_DIR, "generative_molecular_design", "DecompDiff")

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
# Native DecompDiff pipeline
# ---------------------------------------------------------------------------

def try_native():
    """Attempt native DecompDiff sampling."""
    print("[DecompDiff] Attempting native pipeline...")

    if not install_packages(["torch-geometric", "easydict", "pyyaml"]):
        print("[DecompDiff] Failed to install dependencies")
        return False
    install_packages(["gdown", "torch-scatter"])

    # Check/download checkpoint
    ckpt_dir = os.path.join(DECOMPDIFF_DIR, "pretrained")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "uni_o2_bond.pt")

    if not os.path.isfile(ckpt_path):
        print("[DecompDiff] Downloading pretrained checkpoint...")
        try:
            import gdown
            gdown.download(
                "https://drive.google.com/uc?id=1GHJKgLx4vn8vlbVIUl5n9bxQEjXmMz5a",
                ckpt_path, quiet=False,
            )
        except Exception as e:
            print(f"[DecompDiff] Checkpoint download failed: {e}")
            return False

    if not os.path.isfile(ckpt_path):
        print("[DecompDiff] Checkpoint not available")
        return False

    sys.path.insert(0, DECOMPDIFF_DIR)
    try:
        import torch
        from easydict import EasyDict

        # DecompDiff requires preprocessed LMDB datasets, decomposition metadata,
        # and alphaspace2. This is the most complex setup of the four tools.
        from models.decomp_diffusion import DecompScorePosNet3D  # noqa: F811
        from scripts.sample_diffusion import sample_diffusion_ligand_decomp
    except ImportError as e:
        print(f"[DecompDiff] Import failed: {e}")
        print("[DecompDiff] Native pipeline requires complex preprocessing — falling back")
        return False

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        center = get_lsd_center(SDF_PATH)
        pocket_path = extract_pocket_pdb(PDB_PATH, center, radius=10.0)

        ckpt = torch.load(ckpt_path, map_location=device)

        # This would need decomposed ligand data, which requires extensive preprocessing
        # Attempt anyway:
        model = DecompScorePosNet3D(ckpt['config'].model).to(device)
        model.load_state_dict(ckpt['model'])

        # DecompDiff needs: protein_pos, protein_atom_feature, ligand decomposition data
        # This preprocessing chain is not available without the full LMDB pipeline
        raise RuntimeError("DecompDiff requires preprocessed decomposition data (LMDB)")

    except Exception as e:
        print(f"[DecompDiff] Native pipeline error: {e}")
        return False


# ---------------------------------------------------------------------------
# Fallback: BRICS decomposition + Murcko scaffold substitution
# ---------------------------------------------------------------------------

# Fragment library for arm substitutions
ARM_FRAGMENTS = [
    # Small alkyl
    "C", "CC", "CCC", "C(C)C", "CCCC", "C(C)(C)C",
    # Cyclic
    "C1CC1", "C1CCC1", "C1CCCC1", "C1CCCCC1",
    # Heteroatom-containing
    "CO", "CN", "CF", "CCl", "CCO", "CCN",
    "C(=O)N", "C(=O)O", "CS", "C(=O)",
    # Amino groups
    "NC", "NCC", "N(C)C", "N(CC)CC",
    "N1CCCC1", "N1CCCCC1", "N1CCOCC1",
    # Hydroxyl/ether
    "OC", "OCC", "OCCC",
    # Amides
    "NC(=O)C", "C(=O)NC", "NC(=O)CC",
    # Aromatic
    "c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1",
    # Halogenated
    "CF", "CCF", "C(F)(F)F", "CCl", "CBr",
]


def brics_decompose_and_recombine(mol, max_results=50, rng=None):
    """BRICS decompose LSD and recombine with fragment library."""
    if rng is None:
        rng = random.Random()

    smi = Chem.MolToSmiles(mol)
    frags = list(BRICS.BRICSDecompose(mol, returnMols=False))
    print(f"  BRICS fragments: {frags}")

    if len(frags) < 2:
        print("  BRICS produced fewer than 2 fragments, using manual decomposition")
        return []

    # Use BRICSBuild to recombine with variations
    # First convert BRICS fragments to mol objects
    frag_mols = [Chem.MolFromSmiles(f) for f in frags]
    frag_mols = [f for f in frag_mols if f is not None]

    # Also add some fragments from our library
    extra_frags = []
    for arm in ARM_FRAGMENTS:
        m = Chem.MolFromSmiles(arm)
        if m is not None:
            extra_frags.append(m)

    # Combine original fragments with extra fragments
    all_frags = frag_mols + extra_frags

    results = []
    lsd_mw = Descriptors.ExactMolWt(mol)
    lsd_hac = mol.GetNumHeavyAtoms()

    # Try BRICSBuild
    try:
        builder = BRICS.BRICSBuild(frag_mols)
        for i, product in enumerate(builder):
            if i > 500:
                break
            if product is None:
                continue
            try:
                Chem.SanitizeMol(product)
                product_smi = Chem.MolToSmiles(product)
                if '.' in product_smi:
                    continue
                product = Chem.MolFromSmiles(product_smi)
                if product is None:
                    continue
                mw = Descriptors.ExactMolWt(product)
                hac = product.GetNumHeavyAtoms()
                # Filter: similar size to LSD
                if abs(hac - lsd_hac) > 10:
                    continue
                if mw > lsd_mw * 1.5 or mw < lsd_mw * 0.5:
                    continue
                ring_count = Descriptors.RingCount(product)
                if ring_count < 2:  # LSD has 4+ rings
                    continue
                results.append(product)
                if len(results) >= max_results:
                    break
            except Exception:
                continue
    except Exception as e:
        print(f"  BRICSBuild failed: {e}")

    return results


def murcko_scaffold_variants(mol, n_variants=50, rng=None):
    """Extract Murcko scaffold and substitute side chains."""
    if rng is None:
        rng = random.Random()

    results = []
    lsd_smi = Chem.MolToSmiles(mol)
    lsd_mw = Descriptors.ExactMolWt(mol)

    # Get Murcko scaffold
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smi = Chem.MolToSmiles(scaffold)
    print(f"  Murcko scaffold: {scaffold_smi}")

    # The ergoline scaffold — manually enumerate variants with different arms
    # LSD has: diethylamide arm, N-methyl on piperidine
    # Scaffold attachment points are the amide and N-methyl positions

    # Strategy: take the ergoline core and attach different substituents
    # at the amide position and N6 position

    amide_groups = [
        "C(=O)N(C)C",        # dimethylamide
        "C(=O)N(CC)CC",      # diethylamide (original)
        "C(=O)NC",           # N-methylamide
        "C(=O)N",            # primary amide
        "C(=O)NCC",          # N-ethylamide
        "C(=O)N1CCCC1",      # pyrrolidine amide
        "C(=O)N1CCCCC1",     # piperidine amide
        "C(=O)N1CCOCC1",     # morpholine amide
        "C(=O)N1CCN(C)CC1",  # N-methyl piperazine amide
        "C(=O)NC(C)C",       # N-isopropyl amide
        "C(=O)NC1CC1",       # N-cyclopropyl amide
        "C(=O)NCC=C",        # N-allyl amide
        "C(=O)N(C)CC",       # N-methyl-N-ethyl amide
        "C(=O)OC",           # methyl ester
        "C(=O)OCC",          # ethyl ester
        "C(=O)O",            # carboxylic acid
        "C(=O)NNCC",         # hydrazide
        "C(=N)N(CC)CC",      # amidine
        "C(=S)N(CC)CC",      # thioamide
        "CO",                # hydroxymethyl (reduced)
        "CN(CC)CC",          # aminomethyl
    ]

    n6_groups = [
        "C",     # methyl (original)
        "CC",    # ethyl
        "CCC",   # propyl
        "C(C)C", # isopropyl
        "CC=C",  # allyl
        "CC#C",  # propargyl
        "",      # no substituent (NH)
        "C1CC1", # cyclopropylmethyl
    ]

    # Build combinatorial variants using the ergoline template
    # Template: {amide}[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N({n6})C1
    ergoline_template = "[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N({n6})C1"

    seen = set()
    for amide in amide_groups:
        for n6 in n6_groups:
            if len(results) >= n_variants:
                break
            if n6:
                core = ergoline_template.replace("{n6}", n6)
            else:
                core = "[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2NC1"
            smi = amide + core
            mol_v = Chem.MolFromSmiles(smi)
            if mol_v is None:
                continue
            can_smi = Chem.MolToSmiles(mol_v)
            if can_smi in seen or can_smi == lsd_smi:
                continue
            seen.add(can_smi)

            # MW/size filter
            mw = Descriptors.ExactMolWt(mol_v)
            if mw > lsd_mw * 1.5 or mw < lsd_mw * 0.4:
                continue

            results.append(mol_v)

        if len(results) >= n_variants:
            break

    return results


def run_fallback():
    """Generate LSD variants via BRICS decomposition + Murcko scaffold substitution."""
    print("[DecompDiff] Running fallback: BRICS decomposition + Murcko scaffold substitution")

    mol = get_lsd_mol(SDF_PATH)
    mol_noH = Chem.RemoveHs(mol)
    center = get_lsd_center(SDF_PATH)

    # Extract pocket
    extract_pocket_pdb(PDB_PATH, center, radius=10.0)

    random.seed(42)
    np.random.seed(42)
    rng = random.Random(42)

    generated = {}  # smiles -> mol
    idx = 0

    # --- Part A: BRICS decomposition and recombination ---
    print("  Part A: BRICS decomposition and recombination...")
    brics_results = brics_decompose_and_recombine(mol_noH, max_results=50, rng=rng)
    print(f"  BRICS generated {len(brics_results)} candidates")

    for mol_v in brics_results:
        if idx >= 50:
            break
        smi = Chem.MolToSmiles(mol_v)
        if smi in generated:
            continue
        generated[smi] = mol_v

        mol_v_3d = Chem.AddHs(mol_v)
        params = AllChem.ETKDGv3()
        params.randomSeed = idx + 42
        res = AllChem.EmbedMolecule(mol_v_3d, params)
        if res != 0:
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(mol_v_3d, params)
        if res == 0:
            AllChem.MMFFOptimizeMolecule(mol_v_3d, maxIters=500)
            mol_v_3d = Chem.RemoveHs(mol_v_3d)
            save_mol_sdf(mol_v_3d, OUTPUT_DIR, idx)
            idx += 1

    # --- Part B: Murcko scaffold substitution ---
    print("  Part B: Murcko scaffold substitution (scaffold+arms)...")
    murcko_results = murcko_scaffold_variants(mol_noH, n_variants=60, rng=rng)
    print(f"  Murcko generated {len(murcko_results)} candidates")

    for mol_v in murcko_results:
        if idx >= 100:
            break
        smi = Chem.MolToSmiles(mol_v)
        if smi in generated:
            continue
        generated[smi] = mol_v

        mol_v_3d = Chem.AddHs(mol_v)
        params = AllChem.ETKDGv3()
        params.randomSeed = idx + 200
        res = AllChem.EmbedMolecule(mol_v_3d, params)
        if res != 0:
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(mol_v_3d, params)
        if res == 0:
            AllChem.MMFFOptimizeMolecule(mol_v_3d, maxIters=500)
            mol_v_3d = Chem.RemoveHs(mol_v_3d)
            save_mol_sdf(mol_v_3d, OUTPUT_DIR, idx)
            idx += 1

    # --- Part C: Ring decoration variants to fill remaining slots ---
    if idx < 100:
        print(f"  Part C: Additional ring-decorated variants (currently {idx})...")

        ring_deco_variants = [
            # 5-substituted ergoline with various amides
            "CN(C)C(=O)[C@@H]1C=C2c3cc(OC)cc4[nH]cc(c34)C[C@H]2N(C)C1",
            "CN(C)C(=O)[C@@H]1C=C2c3cc(F)cc4[nH]cc(c34)C[C@H]2N(C)C1",
            "O=C(N1CCCC1)[C@@H]1C=C2c3cc(OC)cc4[nH]cc(c34)C[C@H]2N(C)C1",
            "O=C(N1CCCC1)[C@@H]1C=C2c3cc(F)cc4[nH]cc(c34)C[C@H]2N(C)C1",
            "CNC(=O)[C@@H]1C=C2c3cc(OC)cc4[nH]cc(c34)C[C@H]2N(C)C1",
            "CNC(=O)[C@@H]1C=C2c3cc(F)cc4[nH]cc(c34)C[C@H]2N(C)C1",
            "NC(=O)[C@@H]1C=C2c3cc(Cl)cc4[nH]cc(c34)C[C@H]2N(C)C1",
            "NC(=O)[C@@H]1C=C2c3cc(Br)cc4[nH]cc(c34)C[C@H]2N(C)C1",
            # N1-substituted variants
            "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4n(C)cc(c34)C[C@H]2N(C)C1",
            "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4n(CC)cc(c34)C[C@H]2N(C)C1",
            "CN(C)C(=O)[C@@H]1C=C2c3cccc4n(C)cc(c34)C[C@H]2N(C)C1",
            "CN(C)C(=O)[C@@H]1C=C2c3cccc4n(CC=C)cc(c34)C[C@H]2N(C)C1",
            # Multiple modifications
            "O=C(NC1CC1)[C@@H]1C=C2c3cc(F)cc4n(C)cc(c34)C[C@H]2N(CC)C1",
            "O=C(N1CCOCC1)[C@@H]1C=C2c3cc(OC)cc4[nH]cc(c34)C[C@H]2N(C)C1",
            "CCN(CC)C(=O)[C@@H]1C=C2c3cc(O)cc4[nH]cc(c34)C[C@H]2N(CC)C1",
            "CCCN(CCC)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(C)C1",
            "CC(C)NC(=O)[C@@H]1C=C2c3cc(F)cc4[nH]cc(c34)C[C@H]2N(CC)C1",
            "O=C(N1CCN(C)CC1)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(C)C1",
            "C=CCNC(=O)[C@@H]1C=C2c3cc(OC)cc4[nH]cc(c34)C[C@H]2N(C)C1",
            "O=C(N1CCC1)[C@@H]1C=C2c3cc(F)cc4n(C)cc(c34)C[C@H]2N(C)C1",
        ]

        for smi in ring_deco_variants:
            if idx >= 100:
                break
            mol_v = Chem.MolFromSmiles(smi)
            if mol_v is None:
                continue
            can_smi = Chem.MolToSmiles(mol_v)
            if can_smi in generated:
                continue
            generated[can_smi] = mol_v
            mol_v_3d = Chem.AddHs(mol_v)
            params = AllChem.ETKDGv3()
            params.randomSeed = idx + 400
            res = AllChem.EmbedMolecule(mol_v_3d, params)
            if res != 0:
                params.useRandomCoords = True
                res = AllChem.EmbedMolecule(mol_v_3d, params)
            if res == 0:
                AllChem.MMFFOptimizeMolecule(mol_v_3d, maxIters=500)
                mol_v_3d = Chem.RemoveHs(mol_v_3d)
                save_mol_sdf(mol_v_3d, OUTPUT_DIR, idx)
                idx += 1

    return idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DecompDiff: Decomposed Diffusion Molecular Generation")
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
    print(f"DecompDiff Summary")
    print(f"  Method:           {method}")
    print(f"  Total SDF files:  {len(sdf_files)}")
    print(f"  Valid molecules:  {valid}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
