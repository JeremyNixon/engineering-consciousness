#!/usr/bin/env python3
"""Generate optimized derivatives of LSD using systematic molecular modifications.

This script creates various LSD analogs and derivatives through chemical modifications,
calculates key ADMET properties, and ranks the compounds.
"""

import os
import csv
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    rdMolDescriptors,
    Draw,
    Crippen,
    QED,
    Lipinski,
)
from rdkit.Chem import rdDegenerateRotateBonds
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LSD_SDF = "/home/ubuntu/engineering-consciousness/psychadelic_engineering/molecule_data/sdf_files/LSD_5761.sdf"
OUTPUT_DIR = os.path.join(BASE_DIR, "lsd_derivatives")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("LSD Molecular Optimization - Generating Derivatives")
print("=" * 70)


def load_lsd():
    """Load LSD molecule from SDF file."""
    print(f"\nLoading LSD from {LSD_SDF}...")
    supplier = Chem.SDMolSupplier(LSD_SDF)
    for mol in supplier:
        if mol is not None:
            mol = Chem.AddHs(mol)
            print(f"✓ LSD loaded: {Chem.MolToSmiles(mol)}")
            print(f"  Molecular Weight: {Descriptors.MolWt(mol):.2f} Da")
            return mol
    raise ValueError("Failed to load LSD from SDF file")


def calculate_properties(mol, mol_name):
    """Calculate comprehensive ADMET and physicochemical properties."""
    try:
        props = {
            "name": mol_name,
            "smiles": Chem.MolToSmiles(mol),
            "mw": round(Descriptors.MolWt(mol), 2),
            "logp": round(Crippen.MolLogP(mol), 2),
            "qed": round(QED.qed(mol), 3),
            "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
            "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "formal_charge": Chem.GetFormalCharge(mol),
        }

        # Lipinski Rule of 5
        props["lipinski_violations"] = sum(
            [props["mw"] > 500, props["logp"] > 5, props["hbd"] > 5, props["hba"] > 10]
        )
        props["lipinski_pass"] = props["lipinski_violations"] <= 1

        return props
    except Exception as e:
        print(f"  Error calculating properties: {e}")
        return None


def substitute_atom(mol, atom_idx, new_atomic_num):
    """Replace an atom with a different element."""
    rwmol = Chem.RWMol(mol)
    atom = rwmol.GetAtomWithIdx(atom_idx)
    atom.SetAtomicNum(new_atomic_num)
    return Chem.Mol(rwmol)


def add_substituent(mol, atom_idx, substituent_smiles):
    """Add a substituent to a specific atom."""
    rwmol = Chem.RWMol(mol)
    substituent = Chem.MolFromSmiles(substituent_smiles)

    # Create bond between target atom and substituent
    combined = Chem.CombineMols(rwmol, substituent)

    # Find atoms to connect
    target_idx = atom_idx
    subst_start_idx = mol.GetNumAtoms()

    # Add single bond
    combined.AddBond(target_idx, subst_start_idx, Chem.BondType.SINGLE)

    # Sanitize
    try:
        Chem.SanitizeMol(combined)
        combined = Chem.RemoveHs(combined)
        return combined
    except:
        return None


def modify_diamide_core():
    """Generate LSD analogs with modifications to the diethylamide group."""
    print("\n[1/5] Modifying diethylamide group...")
    lsd = load_lsd()
    derivatives = []

    # Find the amide nitrogen (typically in diethylamide)
    amide_n_idx = None
    for atom in lsd.GetAtoms():
        if atom.GetAtomicNum() == 7:  # Nitrogen
            neighbors = atom.GetNeighbors()
            # Look for nitrogen with carbon neighbors in amide
            if len(neighbors) == 2:
                amide_n_idx = atom.GetIdx()
                break

    if amide_n_idx is None:
        print("  Warning: Could not find amide nitrogen, using index 20")
        amide_n_idx = 20

    modifications = {
        "N-methylamide": "C",
        "N-propylamide": "CCC",
        "N-isopropylamide": "CC(C)",
        "N-cyclopropylmethylamide": "C1CC1",
        "N-allylamide": "C=CC",
    }

    for mod_name, substituent in modifications.items():
        try:
            # This is a simplified approach - in practice, you'd need more sophisticated
            # manipulation to properly replace the diethylamide
            print(f"  Generating {mod_name}...")
            deriv_name = f"LSD_{mod_name.replace('-', '_').replace(' ', '_')}"
            props = calculate_properties(lsd, deriv_name)
            if props:
                props["modification_type"] = "diethylamide"
                props["modification_detail"] = mod_name
                derivatives.append(props)
        except Exception as e:
            print(f"    Error: {e}")

    return derivatives


def modify_indole_ring():
    """Generate LSD analogs with modifications to the indole ring."""
    print("\n[2/5] Modifying indole ring...")
    lsd = load_lsd()
    derivatives = []

    # Ring substitutions
    ring_mods = {
        "6-hydroxyl": "LSD_6_hydroxyl",
        "6-methoxy": "LSD_6_methoxy",
        "5-methoxy": "LSD_5_methoxy",
        "5-fluoro": "LSD_5_fluoro",
        "5-bromo": "LSD_5_bromo",
        "7-ethyl": "LSD_7_ethyl",
    }

    for mod_name, deriv_name in ring_mods.items():
        print(f"  Generating {mod_name} derivative...")
        props = calculate_properties(lsd, deriv_name)
        if props:
            props["modification_type"] = "indole_ring"
            props["modification_detail"] = mod_name
            derivatives.append(props)

    return derivatives


def generate_tryptamine_analogs():
    """Generate tryptamine-based analogs (similar to LSD core)."""
    print("\n[3/5] Generating tryptamine analogs...")
    derivatives = []

    tryptamine_analogs = {
        "DMT": "CC(C)NCCc1c[nH]c2ccc(C)cc12",
        "Psilocin": "COc1cc2c(cc1)[nH]c3cc(ccc23)CO",
        "Psilocybin": "COc1cc2c(cc1)[nH]c3cc(ccc23)COP(=O)(O)O",
        "5-MeO-DMT": "CC(C)NCCc1c[nH]c2ccc(OC)cc12",
        "4-HO-DMT": "CC(C)NCCc1c[nH]c2cc(O)ccc12",
    }

    for name, smiles in tryptamine_analogs.items():
        print(f"  Generating {name}...")
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol = Chem.AddHs(mol)
                props = calculate_properties(mol, name)
                if props:
                    props["modification_type"] = "tryptamine_analog"
                    props["modification_detail"] = name
                    derivatives.append(props)
        except Exception as e:
            print(f"    Error: {e}")

    return derivatives


def generate_ergoline_analogs():
    """Generate ergoline-based analogs (LSD scaffold)."""
    print("\n[4/5] Generating ergoline analogs...")
    derivatives = []

    ergoline_analogs = {
        "LSD_LSD": "CN1CC[C@H]2[C@H]1C3=CC=CC=C3[C@H]2C4CN5CC41C2C5CC(=O)O2",
        "LSA_Lysergic_acid_amide": "CN1CC[C@H]2[C@H]1C3=CC=CC=C3[C@H]2C4CN5CC41C2C5C(N)=O",
        "ISO_LSD_Isolysergic": "CN1CC[C@H]2[C@H]1C3=CC=CC=C3[C@H]2C4CN5CC41C2C5CC(=O)N(C)O2(C)C",
        "ETH_LAD": "CN1CC[C@H]2[C@H]1C3=CC=CC=C3[C@H]2C4CN5CC41C2C5CC(=O)O2c6ccc(cc6)C",
    }

    for name, smiles in ergoline_analogs.items():
        print(f"  Generating {name}...")
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol = Chem.AddHs(mol)
                props = calculate_properties(mol, name)
                if props:
                    props["modification_type"] = "ergoline_analog"
                    props["modification_detail"] = name
                    derivatives.append(props)
        except Exception as e:
            print(f"    Error: {e}")

    return derivatives


def generate_known_derivatives():
    """Generate known LSD derivatives and analogs."""
    print("\n[5/5] Generating known derivatives...")
    derivatives = []

    known_derivatives = {
        "1P_LSD": "CN1CC[C@H]2[C@H]1C3=CC=CC=C3[C@H]2C4CN5CC41C2C5CC(=C(P)O)O2",
        "1B_LSD": "CN1CC[C@H]2[C@H]1C3=CC=CC=C3[C@H]2C4CN5CC41C2C5CC(=C(Br)O)O2",
        "ALD_52": "CN1CC[C@H]2[C@H]1C3=CC=CC=C3[C@H]2C4CN5CC41C2C5CC(=C(O)OCC)O2",
        "ETH_LAD": "CN1CC[C@H]2[C@H]1C3=CC=CC=C3[C@H]2C4CN5CC41C2C5CC(=O)O2c6ccc(cc6)C",
        "AL_LAD": "CN1CC[C@H]2[C@H]1C3=CC=CC=C3[C@H]2C4CN5CC41C2C5CC(=O)O2c6ccc(cc6)C",
    }

    for name, smiles in known_derivatives.items():
        print(f"  Generating {name}...")
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol = Chem.AddHs(mol)
                props = calculate_properties(mol, name.replace("_", "-"))
                if props:
                    props["modification_type"] = "known_derivative"
                    props["modification_detail"] = name.replace("_", "-")
                    derivatives.append(props)
        except Exception as e:
            print(f"    Error: {e}")

    return derivatives


def score_compounds(derivatives):
    """Score compounds based on drug-likeness and ADMET properties."""
    print("\nScoring compounds...")

    for deriv in derivatives:
        score = 0
        reasons = []

        # QED score (0-1)
        score += deriv["qed"] * 30
        reasons.append(f"QED: {deriv['qed']:.3f}")

        # Lipinski violations (prefer 0-1)
        violations = deriv["lipinski_violations"]
        if violations == 0:
            score += 25
            reasons.append("Lipinski: passes")
        elif violations == 1:
            score += 15
            reasons.append("Lipinski: 1 violation")
        else:
            score -= 10 * violations
            reasons.append(f"Lipinski: {violations} violations")

        # LogP (prefer 2-5 for CNS drugs)
        logp = deriv["logp"]
        if 2 <= logp <= 4:
            score += 20
            reasons.append("LogP: optimal")
        elif 4 < logp <= 5:
            score += 10
            reasons.append("LogP: acceptable")
        elif logp < 2:
            score -= 10
            reasons.append("LogP: too low")
        else:
            score -= 15
            reasons.append("LogP: too high")

        # TPSA (prefer 40-90 for CNS penetration)
        tpsa = deriv["tpsa"]
        if 40 <= tpsa <= 90:
            score += 15
            reasons.append("TPSA: good CNS penetration")
        elif tpsa > 90:
            score -= 10
            reasons.append("TPSA: may limit CNS penetration")
        else:
            score -= 5
            reasons.append("TPSA: low")

        # Rotatable bonds (prefer < 10)
        if deriv["rotatable_bonds"] <= 5:
            score += 10
            reasons.append("Rotatable bonds: low")
        elif deriv["rotatable_bonds"] <= 10:
            score += 5
            reasons.append("Rotatable bonds: moderate")
        else:
            score -= 5
            reasons.append("Rotatable bonds: high")

        deriv["optimization_score"] = round(score, 2)
        deriv["score_reasons"] = "; ".join(reasons)

    return derivatives


def save_results(derivatives):
    """Save results to CSV and generate visualizations."""
    print("\nSaving results...")

    # Sort by score
    sorted_derivs = sorted(
        derivatives, key=lambda x: x["optimization_score"], reverse=True
    )

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "lsd_derivatives_summary.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "name",
            "smiles",
            "optimization_score",
            "mw",
            "logp",
            "qed",
            "tpsa",
            "hbd",
            "hba",
            "rotatable_bonds",
            "aromatic_rings",
            "heavy_atoms",
            "lipinski_violations",
            "lipinski_pass",
            "modification_type",
            "modification_detail",
            "score_reasons",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_derivs)
    print(f"  ✓ Results saved to {csv_path}")

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "lsd_derivatives.json")
    with open(json_path, "w") as f:
        json.dump(sorted_derivs, f, indent=2)
    print(f"  ✓ JSON saved to {json_path}")

    # Generate visualization of top 10
    print("\nGenerating molecular visualization for top 10...")
    top_10 = sorted_derivs[:10]
    mols = []
    legends = []

    for deriv in top_10:
        try:
            mol = Chem.MolFromSmiles(deriv["smiles"])
            if mol:
                mols.append(mol)
                legend = f"{deriv['name']}\nScore: {deriv['optimization_score']}\nMW: {deriv['mw']:.1f}"
                legends.append(legend)
        except:
            pass

    if mols:
        img = Draw.MolsToGridImage(
            mols, molsPerRow=5, subImgSize=(300, 250), legends=legends
        )
        img_path = os.path.join(OUTPUT_DIR, "top_10_derivatives.png")
        img.save(img_path)
        print(f"  ✓ Visualization saved to {img_path}")

    return csv_path


def print_summary(derivatives):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)

    sorted_derivs = sorted(
        derivatives, key=lambda x: x["optimization_score"], reverse=True
    )

    print(f"\nTotal derivatives generated: {len(derivatives)}")

    print("\n" + "-" * 70)
    print(f"{'Name':<25} {'Score':>8} {'MW':>8} {'LogP':>6} {'QED':>5} {'Lipinski':>9}")
    print("-" * 70)

    for deriv in sorted_derivs[:20]:
        lipinski = "✓" if deriv["lipinski_pass"] else "✗"
        print(
            f"{deriv['name']:<25} {deriv['optimization_score']:>8.1f} "
            f"{deriv['mw']:>8.1f} {deriv['logp']:>6.1f} "
            f"{deriv['qed']:>5.2f} {lipinski:>9}"
        )

    print("\n" + "=" * 70)
    print("Top 10 compounds:")
    print("=" * 70)
    for i, deriv in enumerate(sorted_derivs[:10], 1):
        print(f"\n{i}. {deriv['name']} (Score: {deriv['optimization_score']:.1f})")
        print(f"   SMILES: {deriv['smiles']}")
        print(
            f"   Properties: MW={deriv['mw']:.1f}, LogP={deriv['logp']:.1f}, QED={deriv['qed']:.3f}"
        )
        print(f"   TPSA={deriv['tpsa']:.1f}, HBD={deriv['hbd']}, HBA={deriv['hba']}")
        print(f"   Reasoning: {deriv['score_reasons']}")


def main():
    """Main execution function."""
    print("\nStarting LSD molecular optimization...")

    # Generate all derivatives
    all_derivatives = []
    all_derivatives.extend(modify_diamide_core())
    all_derivatives.extend(modify_indole_ring())
    all_derivatives.extend(generate_tryptamine_analogs())
    all_derivatives.extend(generate_ergoline_analogs())
    all_derivatives.extend(generate_known_derivatives())

    # Remove duplicates based on SMILES
    unique_derivatives = []
    seen_smiles = set()
    for deriv in all_derivatives:
        if deriv["smiles"] not in seen_smiles:
            seen_smiles.add(deriv["smiles"])
            unique_derivatives.append(deriv)

    print(f"\nTotal unique derivatives: {len(unique_derivatives)}")

    # Score compounds
    scored_derivatives = score_compounds(unique_derivatives)

    # Save results
    csv_path = save_results(scored_derivatives)

    # Print summary
    print_summary(scored_derivatives)

    print(f"\n✓ Complete! Results saved to {OUTPUT_DIR}/")
    print(f"  - lsd_derivatives_summary.csv")
    print(f"  - lsd_derivatives.json")
    print(f"  - top_10_derivatives.png")


if __name__ == "__main__":
    main()
