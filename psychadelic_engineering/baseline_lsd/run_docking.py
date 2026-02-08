#!/usr/bin/env python3
"""
Run AutoDock Vina docking: 8 molecules × 13 receptors = 104 docking runs.

Uses the Vina Python API:
  set_receptor → set_ligand_from_file → compute_vina_maps → dock

Features:
- Checkpoint/resume: skips if output PDBQT already exists
- Saves per-run pose PDBQT files in results/
- Aggregates all best affinities into docking_results.csv and .json
"""

import csv
import json
import sys
import time
from pathlib import Path

from vina import Vina

from config import (
    MOLECULES, RECEPTORS,
    PREPARED_LIGANDS_DIR, PREPARED_RECEPTORS_DIR, RESULTS_DIR,
    BINDING_SITES_JSON, RESULTS_CSV, RESULTS_JSON,
    VINA_SEED, VINA_EXHAUSTIVENESS, VINA_N_POSES, VINA_BOX_SIZE,
)


def run_single_dock(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: tuple[float, float, float],
    output_pdbqt: str,
) -> list[list[float]] | None:
    """Run a single Vina docking and return energy table."""
    v = Vina(sf_name="vina", seed=VINA_SEED, verbosity=0)

    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(
        center=list(center),
        box_size=VINA_BOX_SIZE,
    )

    v.dock(
        exhaustiveness=VINA_EXHAUSTIVENESS,
        n_poses=VINA_N_POSES,
    )

    # Write poses
    v.write_poses(output_pdbqt, n_poses=VINA_N_POSES, overwrite=True)

    # Get energies: list of [affinity, ...]
    energies = v.energies(n_poses=VINA_N_POSES)
    return energies


def main():
    print("=" * 60)
    print("AUTODOCK VINA DOCKING")
    print("=" * 60)
    print(f"Molecules:      {len(MOLECULES)}")
    print(f"Receptors:      {len(RECEPTORS)}")
    print(f"Total runs:     {len(MOLECULES) * len(RECEPTORS)}")
    print(f"Exhaustiveness: {VINA_EXHAUSTIVENESS}")
    print(f"Seed:           {VINA_SEED}")
    print(f"Poses:          {VINA_N_POSES}")
    print(f"Box size:       {VINA_BOX_SIZE}")
    print()

    # Load binding sites
    if not BINDING_SITES_JSON.exists():
        print("[ERROR] Binding sites not found. Run prepare_receptors.py first.")
        sys.exit(1)

    binding_sites = json.loads(BINDING_SITES_JSON.read_text())

    # Verify all input files exist
    missing = []
    for mol_name in MOLECULES:
        lig_path = PREPARED_LIGANDS_DIR / f"{mol_name}.pdbqt"
        if not lig_path.exists():
            missing.append(f"Ligand: {lig_path}")
    for rec_name in RECEPTORS:
        rec_path = PREPARED_RECEPTORS_DIR / f"{rec_name}.pdbqt"
        if not rec_path.exists():
            missing.append(f"Receptor: {rec_path}")
        if rec_name not in binding_sites:
            missing.append(f"Binding site: {rec_name}")

    if missing:
        print("[ERROR] Missing files/data:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun prepare_ligands.py and prepare_receptors.py first.")
        sys.exit(1)

    # Run docking
    results = []
    total = len(MOLECULES) * len(RECEPTORS)
    done = 0
    t_start = time.time()

    for mol_name, mol_info in MOLECULES.items():
        lig_path = str(PREPARED_LIGANDS_DIR / f"{mol_name}.pdbqt")

        for rec_name, rec_info in RECEPTORS.items():
            done += 1
            rec_path = str(PREPARED_RECEPTORS_DIR / f"{rec_name}.pdbqt")
            out_path = RESULTS_DIR / f"{mol_name}__{rec_name}_poses.pdbqt"

            site = binding_sites[rec_name]
            center = (site["center_x"], site["center_y"], site["center_z"])

            # Checkpoint: skip if output already exists
            if out_path.exists():
                # Try to read existing result
                try:
                    existing_lines = out_path.read_text().split("\n")
                    for line in existing_lines:
                        if "VINA RESULT" in line:
                            parts = line.split()
                            best_affinity = float(parts[3])
                            results.append({
                                "molecule": mol_name,
                                "receptor": rec_name,
                                "subtype": rec_info["subtype"],
                                "is_validation": rec_info["is_validation"],
                                "best_affinity": best_affinity,
                                "mw": mol_info["mw"],
                            })
                            break
                    print(f"  [{done}/{total}] [SKIP] {mol_name} × {rec_name}: "
                          f"exists (best={best_affinity:.1f} kcal/mol)")
                    continue
                except Exception:
                    pass  # Re-run if can't parse

            # Run docking
            print(f"  [{done}/{total}] Docking {mol_name} × {rec_name} "
                  f"({rec_info['subtype']})...", end="", flush=True)

            try:
                t0 = time.time()
                energies = run_single_dock(rec_path, lig_path, center, str(out_path))
                dt = time.time() - t0

                if energies is not None and len(energies) > 0:
                    best_affinity = float(energies[0][0])
                    results.append({
                        "molecule": mol_name,
                        "receptor": rec_name,
                        "subtype": rec_info["subtype"],
                        "is_validation": rec_info["is_validation"],
                        "best_affinity": best_affinity,
                        "mw": mol_info["mw"],
                    })
                    print(f" {best_affinity:.1f} kcal/mol ({dt:.0f}s)")
                else:
                    print(f" [FAIL] no results ({dt:.0f}s)")
                    results.append({
                        "molecule": mol_name,
                        "receptor": rec_name,
                        "subtype": rec_info["subtype"],
                        "is_validation": rec_info["is_validation"],
                        "best_affinity": None,
                        "mw": mol_info["mw"],
                    })

            except Exception as e:
                print(f" [ERROR] {e}")
                results.append({
                    "molecule": mol_name,
                    "receptor": rec_name,
                    "subtype": rec_info["subtype"],
                    "is_validation": rec_info["is_validation"],
                    "best_affinity": None,
                    "mw": mol_info["mw"],
                })

    elapsed = time.time() - t_start
    print()
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "molecule", "receptor", "subtype", "is_validation",
            "best_affinity", "mw",
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {RESULTS_CSV}")

    # Save JSON
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {RESULTS_JSON}")

    # Summary table
    print()
    print("=" * 60)
    print("BEST AFFINITY SUMMARY (kcal/mol)")
    print("=" * 60)
    subtypes = []
    for rec_info in RECEPTORS.values():
        if rec_info["subtype"] not in subtypes and not rec_info["is_validation"]:
            subtypes.append(rec_info["subtype"])

    header = f"{'Molecule':<14}" + "".join(f"{s:>9}" for s in subtypes)
    print(header)
    print("-" * len(header))

    for mol_name in MOLECULES:
        row = f"{mol_name:<14}"
        for subtype in subtypes:
            val = None
            for r in results:
                if r["molecule"] == mol_name and r["subtype"] == subtype and not r["is_validation"]:
                    val = r["best_affinity"]
                    break
            if val is not None:
                row += f"{val:>9.1f}"
            else:
                row += f"{'N/A':>9}"
        print(row)

    # Validation check
    print()
    print("VALIDATION: Primary vs validation structure affinities")
    for mol_name in MOLECULES:
        for subtype in ["5-HT2A", "5-HT2B"]:
            primary = None
            validation = None
            for r in results:
                if r["molecule"] == mol_name and r["subtype"] == subtype:
                    if not r["is_validation"]:
                        primary = r["best_affinity"]
                    else:
                        validation = r["best_affinity"]
            if primary is not None and validation is not None:
                diff = abs(primary - validation)
                print(f"  {mol_name} @ {subtype}: primary={primary:.1f}, "
                      f"validation={validation:.1f}, Δ={diff:.1f}")


if __name__ == "__main__":
    main()
