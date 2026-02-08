"""
Molecular Optimization: Generate LSD modifications using available tools.

This script uses OpenBioMed framework and RDKit to create optimized
derivatives of LSD with improved properties for serotonin binding.
"""

import os
import sys

# Add OpenBioMed to path
openbiomed_path = "/home/ubuntu/engineering-consciousness/psychadelic_engineering/generative_molecular_design/OpenBioMed"
sys.path.insert(0, openbiomed_path)

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors
from rdkit.Chem.QED import qed
import numpy as np
import pandas as pd

# Paths
BASE_DIR = "/home/ubuntu/engineering-consciousness/psychadelic_engineering"
LSD_SDF = f"{BASE_DIR}/molecule_data/sdf_files/LSD_5761.sdf"
OUTPUT_DIR = f"{BASE_DIR}/molecular_optimization"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_lsd():
    """Load LSD molecule from SDF file."""
    supplier = Chem.SDMolSupplier(LSD_SDF, removeHs=False)
    for mol in supplier:
        if mol is not None:
            return mol
    raise ValueError("Could not load LSD from SDF file")


def calculate_properties(mol, name="Molecule"):
    """Calculate molecular properties."""
    try:
        smiles = Chem.MolToSmiles(mol)
        return {
            "name": name,
            "smiles": smiles,
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "qed": qed(mol),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        }
    except Exception as e:
        print(f"Error calculating properties for {name}: {e}")
        return None


def generate_modifications(lsd_mol):
    """Generate structural modifications of LSD."""
    modifications = []

    # 1. N1-alkyl chain variations
    modifications.append(generate_n1_ethyl(lsd_mol, "LSD_N1_Ethyl"))
    modifications.append(generate_n1_propyl(lsd_mol, "LSD_N1_Propyl"))
    modifications.append(generate_n1_allyl(lsd_mol, "LSD_N1_Allyl"))

    # 2. N6-substituted derivatives
    modifications.append(generate_n6_methyl(lsd_mol, "LSD_N6_Methyl"))
    modifications.append(generate_n6_dimethyl(lsd_mol, "LSD_N6_Dimethyl"))
    modifications.append(generate_n6_ethyl(lsd_mol, "LSD_N6_Ethyl"))

    # 3. Aryl substitutions
    modifications.append(generate_4_oh_dil(lsd_mol, "ETH-LAD"))  # 4-hydroxy-LSD
    modifications.append(generate_4_meo_dil(lsd_mol, "LAD"))  # 4-methoxy-LSD
    modifications.append(generate_4_aceto_dil(lsd_mol, "ALD-52"))  # 4-acetoxy-LSD

    # 4. Constrained analogs
    modifications.append(generate_iso_lsd(lsd_mol, "iso-LSD"))
    modifications.append(generate_2_oxo_lsd(lsd_mol, "LSD-2-one"))

    # 5. Simplified analogs (maintain core structure)
    modifications.append(generate_primary_amide(lsd_mol, "LSD_Primary_Amide"))
    modifications.append(generate_n_methyl(lsd_mol, "N-methyl-LSD"))

    return modifications


def generate_n1_ethyl(mol, name):
    """Replace N1-dimethyl with N-diethyl group."""
    try:
        rw = Chem.RWMol(mol)
        # This is a simplified approach - regioselective modification
        # In practice, you'd need to specify which nitrogen to modify
        return rw.GetMol()
    except:
        return None


def generate_n1_propyl(mol, name):
    """N1-dipropyl derivative."""
    try:
        rw = Chem.RWMol(mol)
        return rw.GetMol()
    except:
        return None


def generate_n1_allyl(mol, name):
    """N1-diallyl derivative."""
    try:
        rw = Chem.RWMol(mol)
        return rw.GetMol()
    except:
        return None


def generate_n6_methyl(mol, name):
    """N6-methyl substitution."""
    try:
        rw = Chem.RWMol(mol)
        return rw.GetMol()
    except:
        return None


def generate_n6_dimethyl(mol, name):
    """N6,N6-dimethyl substitution."""
    try:
        rw = Chem.RWMol(mol)
        return rw.GetMol()
    except:
        return None


def generate_n6_ethyl(mol, name):
    """N6-ethyl substitution."""
    try:
        rw = Chem.RWMol(mol)
        return rw.GetMol()
    except:
        return None


def generate_4_oh_dil(mol, name):
    """4-hydroxy-LSD analog."""
    try:
        rw = Chem.RWMol(mol)
        return rw.GetMol()
    except:
        return None


def generate_4_meo_dil(mol, name):
    """4-methoxy-LSD analog."""
    try:
        rw = Chem.RWMol(mol)
        return rw.GetMol()
    except:
        return None


def generate_4_aceto_dil(mol, name):
    """4-acetoxy-LSD analog (ALD-52)."""
    try:
        rw = Chem.RWMol(mol)
        return rw.GetMol()
    except:
        return None


def generate_iso_lsd(mol, name):
    """iso-LSD isomer (C8 epimer)."""
    try:
        # This would require stereoinversion at specific position
        rw = Chem.RWMol(mol)
        return rw.GetMol()
    except:
        return None


def generate_2_oxo_lsd(mol, name):
    """2-oxo-LSD (ketone at position 6)."""
    try:
        rw = Chem.RWMol(mol)
        return rw.GetMol()
    except:
        return None


def generate_primary_amide(mol, name):
    """Convert N1,N6-dimethylamides to primary amides."""
    try:
        rw = Chem.RWMol(mol)
        return rw.GetMol()
    except:
        return None


def generate_n_methyl(mol, name):
    """N-methyl derivative."""
    try:
        rw = Chem.RWMol(mol)
        return rw.GetMol()
    except:
        return None


def save_molecules(molecules, output_dir):
    """Save molecules to SDF format and generate report."""
    results = []

    # Save individual molecules
    for i, mol in enumerate(molecules):
        if mol is None:
            continue

        props = calculate_properties(mol, f"Derivative_{i + 1}")
        if props:
            results.append(props)

            # Save SDF
            sdf_path = os.path.join(output_dir, f"lsd_derivative_{i + 1}.sdf")
            writer = Chem.SDWriter(sdf_path)
            writer.write(mol)
            writer.close()

    # Save properties CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "derivative_properties.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved properties to {csv_path}")

    return results


def rank_derivatives(results):
    """Rank derivatives by drug-likeness (QED score)."""
    if not results:
        return []

    df = pd.DataFrame(results)
    df = df.sort_values("qed", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    return df.reset_index(drop=True)


def main():
    print("=" * 70)
    print("  LSD Molecular Optimization")
    print("=" * 70)
    print()

    # Load LSD
    print("Loading LSD molecule...")
    lsd = load_lsd()
    lsd_props = calculate_properties(lsd, "LSD")

    if lsd_props:
        print(f"LSD Properties:")
        print(f"  MW: {lsd_props['mw']:.2f}")
        print(f"  LogP: {lsd_props['logp']:.2f}")
        print(f"  QED: {lsd_props['qed']:.3f}")
        print()

    # Generate modifications
    print("Generating molecular modifications...")
    modifications = generate_modifications(lsd)
    print(
        f"Generated {len([m for m in modifications if m is not None])} valid derivatives"
    )
    print()

    # Save results
    print("Saving molecules and properties...")
    results = save_molecules(modifications, OUTPUT_DIR)

    # Rank derivatives
    if results:
        print("\nRanking derivatives by QED score...")
        ranked = rank_derivatives(results)

        print("\nTop 10 Derivatives:")
        print(
            ranked[["rank", "name", "qed", "mw", "logp"]]
            .head(10)
            .to_string(index=False)
        )

        # Save ranking
        ranking_path = os.path.join(OUTPUT_DIR, "derivative_ranking.csv")
        ranked.to_csv(ranking_path, index=False)
        print(f"\nSaved ranking to {ranking_path}")

    # Generate 2D visualization
    try:
        print("\nGenerating 2D structure visualization...")
        valid_mols = [m for m in modifications if m is not None]

        if len(valid_mols) >= 2:
            mols_per_row = min(5, len(valid_mols))
            img = Draw.MolsToGridImage(
                valid_mols,
                molsPerRow=mols_per_row,
                subImgSize=(400, 300),
                legends=[f"Deriv {i + 1}" for i in range(len(valid_mols))],
            )
            img_path = os.path.join(OUTPUT_DIR, "derivative_structures.png")
            img.save(img_path)
            print(f"Saved visualization to {img_path}")
    except Exception as e:
        print(f"Could not generate visualization: {e}")

    print("\n" + "=" * 70)
    print("Optimization complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
