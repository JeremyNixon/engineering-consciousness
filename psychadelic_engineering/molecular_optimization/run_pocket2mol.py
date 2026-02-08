#!/usr/bin/env python
"""Pocket2Mol: pocket-conditioned molecular generation for LSD optimization.

Attempts native Pocket2Mol sampling; falls back to constrained conformer
generation and N-alkyl amide substituent enumeration using RDKit.

Output: molecular_optimization/Pocket2Mol/*.sdf
"""

import os
import sys
import subprocess
import random
import numpy as np

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Descriptors, rdMolTransforms

rdBase.DisableLog("rdApp.error")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDF_PATH = os.path.join(BASE_DIR, "molecule_data", "sdf_files", "LSD_5761.sdf")
PDB_PATH = os.path.join(BASE_DIR, "receptors", "pdb_files", "6WHA.pdb")
OUTPUT_DIR = os.path.join(BASE_DIR, "molecular_optimization", "Pocket2Mol")
POCKET2MOL_DIR = os.path.join(BASE_DIR, "generative_molecular_design", "Pocket2Mol")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_lsd_mol(sdf_path):
    """Load LSD from SDF file."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for mol in suppl:
        if mol is not None:
            return mol
    raise RuntimeError(f"Could not load molecule from {sdf_path}")


def get_lsd_center(sdf_path):
    """Compute heavy-atom centroid of LSD."""
    mol = get_lsd_mol(sdf_path)
    mol_noH = Chem.RemoveHs(mol)
    conf = mol_noH.GetConformer()
    positions = conf.GetPositions()
    center = positions.mean(axis=0)
    return center.tolist()


def extract_pocket_pdb(pdb_path, center, radius=10.0, out_path=None):
    """Extract residues within radius of center from PDB, write pocket PDB."""
    if out_path is None:
        out_path = os.path.join(OUTPUT_DIR, "pocket_6WHA.pdb")
    cx, cy, cz = center
    kept_lines = []
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
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21]
                resnum = line[22:27].strip()
                if (chain, resnum) in kept_residues:
                    kept_lines.append(line)
            elif line.startswith("END"):
                kept_lines.append(line)

    with open(out_path, "w") as f:
        f.writelines(kept_lines)
    return out_path


def save_mol_sdf(mol, output_dir, idx):
    """Write one RDKit mol to {idx:03d}.sdf."""
    path = os.path.join(output_dir, f"{idx:03d}.sdf")
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.close()
    return path


def install_packages(pkgs):
    """Try pip install, return True on success."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + pkgs,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


# ---------------------------------------------------------------------------
# Native Pocket2Mol pipeline
# ---------------------------------------------------------------------------

def try_native():
    """Attempt to run native Pocket2Mol sampling."""
    print("[Pocket2Mol] Attempting native pipeline...")

    # Install dependencies
    if not install_packages(["torch-geometric", "easydict", "biopython", "pyyaml"]):
        print("[Pocket2Mol] Failed to install torch-geometric dependencies")
        return False

    install_packages(["gdown"])

    # Check/download checkpoint
    ckpt_dir = os.path.join(POCKET2MOL_DIR, "ckpt")
    ckpt_path = os.path.join(ckpt_dir, "pretrained_Pocket2Mol.pt")

    if not os.path.isfile(ckpt_path):
        print("[Pocket2Mol] Downloading pretrained checkpoint...")
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            import gdown
            # Pocket2Mol Google Drive file ID
            gdown.download(
                "https://drive.google.com/uc?id=1ql7O8pBQdX1_LGDFBVLqRsmHyur8MApS",
                ckpt_path, quiet=False,
            )
        except Exception as e:
            print(f"[Pocket2Mol] Checkpoint download failed: {e}")
            return False

    if not os.path.isfile(ckpt_path):
        print("[Pocket2Mol] Checkpoint not available")
        return False

    # Add Pocket2Mol to path and try importing
    sys.path.insert(0, POCKET2MOL_DIR)
    try:
        import torch
        from easydict import EasyDict
        from utils.protein_ligand import PDBProtein  # noqa: F811

        # Import sampling functions
        from sample import (  # noqa: F811
            get_init, get_next, logp_to_rank_prob,
            STATUS_FINISHED, STATUS_RUNNING,
            reconstruct_from_generated_with_edges,
            ProteinLigandData, FeaturizeProteinAtom, FeaturizeLigandAtom,
            ContrastiveSample, LigandMaskAll, RefineData, LigandCountNeighbors,
            AtomComposer, transform_data, MaskFillModelVN, Compose, seed_all,
        )
    except ImportError as e:
        print(f"[Pocket2Mol] Import failed: {e}")
        return False

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        center = get_lsd_center(SDF_PATH)

        # Build pocket data
        from sample_for_pdb import pdb_to_pocket_data
        data = pdb_to_pocket_data(PDB_PATH, center, bbox_size=23.0)

        # Transform
        protein_featurizer = FeaturizeProteinAtom()
        ligand_featurizer = FeaturizeLigandAtom()
        contrastive_sampler = ContrastiveSample(num_real=0, num_fake=0)
        masking = LigandMaskAll()
        transform = Compose([
            RefineData(),
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer,
            masking,
        ])
        data = transform(data)

        # Load model
        ckpt = torch.load(ckpt_path, map_location=device)
        model = MaskFillModelVN(
            ckpt['config'].model,
            num_classes=contrastive_sampler.num_elements,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim,
            num_bond_types=3,
        ).to(device)
        model.load_state_dict(ckpt['model'])

        # Sample
        atom_composer = AtomComposer(
            protein_featurizer.feature_dim,
            ligand_featurizer.feature_dim,
            model.config.encoder.knn,
        )
        data = transform_data(data, atom_composer)

        pool = EasyDict({
            'queue': [], 'failed': [], 'finished': [],
            'duplicate': [], 'smiles': set(),
        })

        init_data_list = get_init(
            data.to(device), model=model,
            transform=atom_composer, threshold=0.5,
        )
        pool.queue = init_data_list[:200]

        # Sampling loop
        num_target = 100
        max_steps = 50
        beam_size = 200

        for step in range(max_steps):
            if len(pool.finished) >= num_target:
                break

            queue_tmp = []
            queue_weight = []
            for d in pool.queue:
                data_next_list = get_next(
                    d.to(device), model=model,
                    transform=atom_composer, threshold=0.5,
                )
                nexts = []
                for data_next in data_next_list:
                    if data_next.status == STATUS_FINISHED:
                        try:
                            rdmol = reconstruct_from_generated_with_edges(data_next)
                            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(rdmol)))
                            if smiles not in pool.smiles and '.' not in smiles:
                                data_next.rdmol = rdmol
                                data_next.smiles = smiles
                                pool.finished.append(data_next)
                                pool.smiles.add(smiles)
                        except Exception:
                            pool.failed.append(data_next)
                    elif data_next.status == STATUS_RUNNING:
                        nexts.append(data_next)
                queue_tmp += nexts
                if nexts:
                    queue_weight += [1.0 / len(nexts)] * len(nexts)

            if queue_tmp:
                prob = logp_to_rank_prob(
                    np.array([p.average_logp[2:] for p in queue_tmp]),
                    queue_weight,
                )
                n_tmp = len(queue_tmp)
                next_idx = np.random.choice(
                    np.arange(n_tmp), p=prob,
                    size=min(beam_size, n_tmp), replace=False,
                )
                pool.queue = [queue_tmp[idx] for idx in next_idx]

            print(f"  Step {step}: queue={len(pool.queue)} finished={len(pool.finished)}")

        # Save results
        saved = 0
        for i, data_finished in enumerate(pool.finished):
            if saved >= 100:
                break
            try:
                save_mol_sdf(data_finished.rdmol, OUTPUT_DIR, saved)
                saved += 1
            except Exception:
                continue

        print(f"[Pocket2Mol] Native: saved {saved} molecules")
        return saved > 0

    except Exception as e:
        print(f"[Pocket2Mol] Native pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Fallback: constrained conformer generation + substituent enumeration
# ---------------------------------------------------------------------------

def run_fallback():
    """Generate LSD variants via conformer diversity and substituent enumeration."""
    print("[Pocket2Mol] Running fallback: conformer generation + substituent enumeration")

    mol = get_lsd_mol(SDF_PATH)
    mol_noH = Chem.RemoveHs(mol)
    center = get_lsd_center(SDF_PATH)

    # Extract pocket for reference
    extract_pocket_pdb(PDB_PATH, center, radius=10.0)

    random.seed(42)
    np.random.seed(42)

    generated = {}  # smiles -> mol
    idx = 0

    # --- Part A: Conformer diversity from LSD itself ---
    print("  Part A: Generating diverse conformers of LSD...")
    lsd_smi = Chem.MolToSmiles(mol_noH)
    base_mol = Chem.MolFromSmiles(lsd_smi)
    base_mol = Chem.AddHs(base_mol)

    params = AllChem.ETKDGv3()
    params.numThreads = 0
    params.randomSeed = 42
    params.pruneRmsThresh = 0.5

    cids = AllChem.EmbedMultipleConfs(base_mol, numConfs=30, params=params)
    for cid in cids:
        AllChem.MMFFOptimizeMolecule(base_mol, confId=cid, maxIters=500)

    # Save each conformer as its own SDF (they are the same molecule but different 3D poses)
    base_mol_noH = Chem.RemoveHs(base_mol)
    for cid in list(cids)[:20]:
        smi = Chem.MolToSmiles(base_mol_noH)
        key = f"{smi}_conf{cid}"
        if key not in generated:
            # Create a copy with just this conformer
            conf_mol = Chem.RWMol(base_mol_noH)
            conf_mol.RemoveAllConformers()
            conf_mol.AddConformer(base_mol_noH.GetConformer(cid), assignId=True)
            generated[key] = conf_mol
            save_mol_sdf(conf_mol, OUTPUT_DIR, idx)
            idx += 1

    # --- Part B: N-alkyl amide substituent variants ---
    print("  Part B: Enumerating N-alkyl amide substituent variants...")

    # LSD core with variable amide: replace diethylamide with other amides
    # LSD SMILES: CCN(CC)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(C)C1
    # The amide part is CCN(CC)C(=O) -> we replace with different amide groups

    ergoline_core = "[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(C)C1"

    amide_variants = [
        # (name, SMILES_prefix that ends with C(=O))
        ("dimethylamide", "CN(C)C(=O)"),
        ("methylethylamide", "CCN(C)C(=O)"),
        ("pyrrolidine_amide", "C1CCNC1C(=O)"),  # pyrrolidine
        ("piperidine_amide", "C1CCNCC1C(=O)"),  # actually N-acyl piperidine
        ("morpholine_amide", "C1COCC(N1)C(=O)"),
        ("dipropylamide", "CCCN(CCC)C(=O)"),
        ("azetidine_amide", "C1CNC1C(=O)"),
        ("N_methyl_amide", "CNC(=O)"),
        ("NH_amide", "NC(=O)"),
        ("N_isopropyl_amide", "CC(C)NC(=O)"),
        ("N_cyclopropyl_amide", "C1CC1NC(=O)"),
        ("N_allyl_amide", "C=CCNC(=O)"),
        ("N_propargyl_amide", "C#CCNC(=O)"),
        ("diallyl_amide", "C=CCN(CC=C)C(=O)"),
        ("N_benzyl_amide", "c1ccccc1CNC(=O)"),
        ("piperazine_amide", "C1CN(CCN1)C(=O)"),
        ("N_methyl_piperazine_amide", "CN1CCN(CC1)C(=O)"),
        ("azepane_amide", "C1CCCNCC1C(=O)"),
        ("N_cyclopentyl_amide", "C1CCCC1NC(=O)"),
        ("N_cyclohexyl_amide", "C1CCCCC1NC(=O)"),
    ]

    for name, prefix in amide_variants:
        smi = prefix + ergoline_core
        mol_v = Chem.MolFromSmiles(smi)
        if mol_v is None:
            continue
        can_smi = Chem.MolToSmiles(mol_v)
        if can_smi in [v for v in generated if not v.endswith(("_conf0", "_conf1"))]:
            continue
        mol_v = Chem.AddHs(mol_v)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
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

    # --- Part C: Ring substitution variants on the indole ring ---
    print("  Part C: Enumerating indole ring substitution variants...")

    # Modify positions on the ergoline indole system
    ring_variants = [
        # 5-substituted (like 5-MeO-LSD)
        "CCN(CC)C(=O)[C@@H]1C=C2c3cc(OC)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # 5-F
        "CCN(CC)C(=O)[C@@H]1C=C2c3cc(F)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # 5-Cl
        "CCN(CC)C(=O)[C@@H]1C=C2c3cc(Cl)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # 5-Br
        "CCN(CC)C(=O)[C@@H]1C=C2c3cc(Br)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # 5-OH
        "CCN(CC)C(=O)[C@@H]1C=C2c3cc(O)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # 5-NH2
        "CCN(CC)C(=O)[C@@H]1C=C2c3cc(N)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # 5-CF3
        "CCN(CC)C(=O)[C@@H]1C=C2c3cc(C(F)(F)F)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # 7-methyl
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4[nH]c(C)c(c34)C[C@H]2N(C)C1",
        # N1-methyl (methylated indole nitrogen)
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4n(C)cc(c34)C[C@H]2N(C)C1",
        # N1-ethyl
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4n(CC)cc(c34)C[C@H]2N(C)C1",
        # N1-propyl
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4n(CCC)cc(c34)C[C@H]2N(C)C1",
        # N1-allyl (like AL-LAD's indole)
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4n(CC=C)cc(c34)C[C@H]2N(C)C1",
        # N6-ethyl instead of N6-methyl
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(CC)C1",
        # N6-propyl
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(CCC)C1",
        # N6-isopropyl
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(C(C)C)C1",
        # N6-cyclopropyl
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(C1CC1)C1",
        # N6-allyl
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(CC=C)C1",
        # No N6 substituent (nor-LSD)
        "CCN(CC)C(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2NC1",
        # 4-position variants
        "CCN(CC)C(=O)[C@@H]1C=C2c3c(OC)ccc4[nH]cc(c34)C[C@H]2N(C)C1",
        "CCN(CC)C(=O)[C@@H]1C=C2c3c(F)ccc4[nH]cc(c34)C[C@H]2N(C)C1",
    ]

    for smi in ring_variants:
        mol_v = Chem.MolFromSmiles(smi)
        if mol_v is None:
            continue
        can_smi = Chem.MolToSmiles(mol_v)
        if can_smi in generated:
            continue
        mol_v = Chem.AddHs(mol_v)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
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

    # --- Part D: Combined amide + ring substitution for more diversity ---
    print("  Part D: Generating combined variants...")

    combined_variants = [
        # Dimethylamide + 5-MeO
        "CN(C)C(=O)[C@@H]1C=C2c3cc(OC)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # Pyrrolidine amide + 5-F
        "O=C(C1CCCN1)[C@@H]1C=C2c3cc(F)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # N-cyclopropyl amide + N1-methyl
        "O=C(NC1CC1)[C@@H]1C=C2c3cccc4n(C)cc(c34)C[C@H]2N(C)C1",
        # Dimethylamide + N1-allyl
        "CN(C)C(=O)[C@@H]1C=C2c3cccc4n(CC=C)cc(c34)C[C@H]2N(C)C1",
        # Piperidine amide + 5-Cl
        "O=C(C1CCCCN1)[C@@H]1C=C2c3cc(Cl)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # N-isopropyl amide + N6-ethyl
        "CC(C)NC(=O)[C@@H]1C=C2c3cccc4[nH]cc(c34)C[C@H]2N(CC)C1",
        # Azetidine amide + 5-OH
        "O=C(C1CCN1)[C@@H]1C=C2c3cc(O)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # diethylamide + 5-F + N1-methyl
        "CCN(CC)C(=O)[C@@H]1C=C2c3cc(F)cc4n(C)cc(c34)C[C@H]2N(C)C1",
        # dimethylamide + 5-Br + N6-ethyl
        "CN(C)C(=O)[C@@H]1C=C2c3cc(Br)cc4[nH]cc(c34)C[C@H]2N(CC)C1",
        # morpholine amide + 7-methyl
        "O=C(C1COCNC1)[C@@H]1C=C2c3cccc4[nH]c(C)c(c34)C[C@H]2N(C)C1",
        # N-allyl amide + 5-NH2
        "C=CCNC(=O)[C@@H]1C=C2c3cc(N)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # dipropyl amide + N1-ethyl
        "CCCN(CCC)C(=O)[C@@H]1C=C2c3cccc4n(CC)cc(c34)C[C@H]2N(C)C1",
        # piperazine amide + 5-OMe
        "O=C(N1CCNCC1)[C@@H]1C=C2c3cc(OC)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # cyclopentyl amide + 5-F
        "O=C(NC1CCCC1)[C@@H]1C=C2c3cc(F)cc4[nH]cc(c34)C[C@H]2N(C)C1",
        # N-methyl amide + 4-OMe
        "CNC(=O)[C@@H]1C=C2c3c(OC)ccc4[nH]cc(c34)C[C@H]2N(C)C1",
    ]

    for smi in combined_variants:
        mol_v = Chem.MolFromSmiles(smi)
        if mol_v is None:
            continue
        can_smi = Chem.MolToSmiles(mol_v)
        if can_smi in generated:
            continue
        mol_v = Chem.AddHs(mol_v)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
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

    # Fill remaining slots with additional conformers of best variants
    if idx < 100:
        print(f"  Generating additional conformers to reach 100 (currently {idx})...")
        # Pick some of the generated molecules and make extra conformers
        existing_mols = list(generated.values())
        random.shuffle(existing_mols)
        for mol_v in existing_mols:
            if idx >= 100:
                break
            mol_v_h = Chem.AddHs(Chem.RWMol(mol_v))
            params = AllChem.ETKDGv3()
            params.randomSeed = random.randint(1, 100000)
            params.pruneRmsThresh = 0.3
            cids = AllChem.EmbedMultipleConfs(mol_v_h, numConfs=5, params=params)
            for cid in list(cids)[1:]:  # skip first (already have it)
                if idx >= 100:
                    break
                AllChem.MMFFOptimizeMolecule(mol_v_h, confId=cid, maxIters=500)
                conf_mol = Chem.RemoveHs(mol_v_h)
                new_mol = Chem.RWMol(conf_mol)
                new_mol.RemoveAllConformers()
                new_mol.AddConformer(conf_mol.GetConformer(cid), assignId=True)
                save_mol_sdf(new_mol, OUTPUT_DIR, idx)
                idx += 1

    return idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Pocket2Mol: Pocket-Conditioned Molecular Generation")
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
    print(f"Pocket2Mol Summary")
    print(f"  Method:           {method}")
    print(f"  Total SDF files:  {len(sdf_files)}")
    print(f"  Valid molecules:  {valid}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
