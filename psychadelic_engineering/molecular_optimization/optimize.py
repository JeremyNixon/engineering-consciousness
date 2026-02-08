#!/usr/bin/env python
"""Genetic-algorithm molecular optimizer using AutoDock Vina docking scores.

Uses the Graph-based GA operators (crossover + mutate) from
generative_molecular_design/guacamol_baselines/graph_ga/ to evolve molecules
that maximize binding affinity to a given receptor.

Usage:
    python molecular_optimization/optimize.py \
        --receptor receptors/pdb_files/3PBL.pdb \
        --smiles "NCCc1ccc(O)c(O)c1" \
        --center 0.09 -14.83 10.43 \
        --generations 50 --population_size 50 --offspring_size 100
"""

import argparse
import os
import random
import sys
import time

import numpy as np
from rdkit import Chem, rdBase

rdBase.DisableLog("rdApp.error")

# ---------------------------------------------------------------------------
# Import graph_ga crossover / mutate via sys.path
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_GA_DIR = os.path.join(
    BASE_DIR, "generative_molecular_design", "guacamol_baselines", "graph_ga"
)
sys.path.insert(0, GRAPH_GA_DIR)

import crossover as co  # noqa: E402

# mutate.py uses ``from . import crossover as co`` which fails when loaded
# outside the package.  Work around by making the already-imported crossover
# module available under the dotted name that the relative import resolves to.
import types  # noqa: E402

_fake_pkg = types.ModuleType("__graph_ga_pkg__")
_fake_pkg.crossover = co
sys.modules[_fake_pkg.__name__] = _fake_pkg

# Temporarily patch mutate's package so the relative import finds crossover
import importlib  # noqa: E402

_mutate_spec = importlib.util.spec_from_file_location(
    "__graph_ga_pkg__.mutate",
    os.path.join(GRAPH_GA_DIR, "mutate.py"),
    submodule_search_locations=[],
)
mu = importlib.util.module_from_spec(_mutate_spec)
mu.__package__ = _fake_pkg.__name__
sys.modules[_mutate_spec.name] = mu
# crossover must be findable as a sibling
sys.modules[f"{_fake_pkg.__name__}.crossover"] = co
_mutate_spec.loader.exec_module(mu)

# ---------------------------------------------------------------------------
# GA helpers (adapted from goal_directed_generation.py)
# ---------------------------------------------------------------------------

def make_mating_pool(population_mol, population_scores, offspring_size):
    """Fitness-proportional selection with replacement."""
    sum_scores = sum(population_scores)
    if sum_scores == 0:
        # Uniform selection when all scores are zero
        probs = [1.0 / len(population_scores)] * len(population_scores)
    else:
        probs = [s / sum_scores for s in population_scores]
    return list(
        np.random.choice(population_mol, p=probs, size=offspring_size, replace=True)
    )


def reproduce(mating_pool, mutation_rate):
    """Pick two parents, crossover, then mutate."""
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    child = co.crossover(parent_a, parent_b)
    if child is not None:
        child = mu.mutate(child, mutation_rate)
    return child


def sanitize(population_mol):
    """Deduplicate and remove invalid molecules."""
    seen = set()
    clean = []
    for mol in population_mol:
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(mol)
            if smi is not None and smi not in seen:
                seen.add(smi)
                clean.append(mol)
        except ValueError:
            continue
    return clean


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------

def optimize(
    scorer,
    population_size=50,
    offspring_size=100,
    generations=50,
    mutation_rate=0.01,
    patience=5,
    n_results=5,
    initial_smiles=None,
    smiles_file=None,
    seed=42,
):
    """Run the genetic algorithm and return the top molecules.

    Args:
        scorer: VinaDockingScorer instance
        population_size: number of molecules kept each generation
        offspring_size: number of offspring generated each generation
        generations: maximum number of generations
        mutation_rate: probability of mutation per offspring
        patience: stop after this many generations without improvement
        n_results: number of top molecules to return
        initial_smiles: optional starting SMILES (string or list)
        smiles_file: path to .smi file for seeding population
        seed: random seed

    Returns:
        list of (smiles, raw_affinity_kcal) tuples, best first
    """
    np.random.seed(seed)
    random.seed(seed)

    # --- Build initial SMILES pool ---
    all_smiles = []

    if smiles_file and os.path.isfile(smiles_file):
        with open(smiles_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    smi = parts[0]
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        all_smiles.append(Chem.MolToSmiles(mol))

    if initial_smiles:
        if isinstance(initial_smiles, str):
            initial_smiles = [initial_smiles]
        for smi in initial_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                all_smiles.append(Chem.MolToSmiles(mol))

    if not all_smiles:
        raise ValueError(
            "No valid molecules for initial population. "
            "Provide --smiles and/or a valid --smiles_file."
        )

    # Deduplicate
    all_smiles = list(dict.fromkeys(all_smiles))

    # --- Score initial pool and select top population_size ---
    print(f"Scoring initial pool of {len(all_smiles)} molecules ...")
    t0 = time.time()
    pool_scores = [scorer.score(smi) for smi in all_smiles]
    print(f"  done in {time.time() - t0:.1f}s")

    # Sort descending by score, keep top population_size
    ranked = sorted(zip(pool_scores, all_smiles), key=lambda x: x[0], reverse=True)
    ranked = ranked[:population_size]

    population_scores = [s for s, _ in ranked]
    population_smiles = [smi for _, smi in ranked]
    population_mol = [Chem.MolFromSmiles(smi) for smi in population_smiles]

    # Track best score for early stopping
    best_score = max(population_scores) if population_scores else 0.0
    stale_gens = 0

    print(f"\n{'Gen':>4}  {'Max':>8}  {'Avg':>8}  {'Min':>8}  {'Time':>8}")
    print("-" * 44)

    for gen in range(generations):
        t_gen = time.time()

        # --- Generate offspring ---
        mating_pool = make_mating_pool(
            population_mol, population_scores, offspring_size
        )
        offspring_mol = [reproduce(mating_pool, mutation_rate) for _ in range(offspring_size)]

        # --- Merge, sanitize, and score ---
        combined = population_mol + offspring_mol
        combined = sanitize(combined)

        # Score only new molecules (already have scores for current pop)
        pop_smi_set = set(population_smiles)
        combined_smiles = [Chem.MolToSmiles(m) for m in combined]

        combined_scores = []
        for smi in combined_smiles:
            if smi in pop_smi_set:
                idx = population_smiles.index(smi)
                combined_scores.append(population_scores[idx])
            else:
                combined_scores.append(scorer.score(smi))

        # --- Select top population_size ---
        ranked = sorted(
            zip(combined_scores, combined_smiles, combined),
            key=lambda x: x[0],
            reverse=True,
        )[:population_size]

        population_scores = [s for s, _, _ in ranked]
        population_smiles = [smi for _, smi, _ in ranked]
        population_mol = [m for _, _, m in ranked]

        gen_time = time.time() - t_gen
        gen_max = max(population_scores)
        gen_avg = sum(population_scores) / len(population_scores)
        gen_min = min(population_scores)

        print(f"{gen:4d}  {gen_max:8.4f}  {gen_avg:8.4f}  {gen_min:8.4f}  {gen_time:7.1f}s")

        # --- Early stopping ---
        if gen_max > best_score:
            best_score = gen_max
            stale_gens = 0
        else:
            stale_gens += 1
            if stale_gens >= patience:
                print(f"\nEarly stopping: no improvement for {patience} generations.")
                break

    # --- Report results ---
    print("\n" + "=" * 60)
    print(f"Top {n_results} molecules:")
    print("=" * 60)

    results = []
    for i, (sc, smi) in enumerate(zip(population_scores, population_smiles)):
        if i >= n_results:
            break
        affinity = scorer.dock(smi)
        results.append((smi, affinity))
        print(f"  {i + 1}. {smi}")
        print(f"     Affinity: {affinity:.2f} kcal/mol  (score: {sc:.4f})")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evolve molecules to maximize binding affinity via GA + Vina"
    )
    parser.add_argument(
        "--receptor", required=True, help="Path to receptor PDB file"
    )
    parser.add_argument(
        "--smiles", default=None, help="Starting molecule SMILES"
    )
    parser.add_argument(
        "--smiles_file", default=None,
        help="SMILES file for initial population (default: molecule_data/psychoactive_molecules.smi)"
    )
    parser.add_argument(
        "--center", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Binding site center (auto-detected from PDB if omitted)"
    )
    parser.add_argument(
        "--box_size", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Docking box size in Angstroms (default: 20 20 20)"
    )
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--population_size", type=int, default=50)
    parser.add_argument("--offspring_size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--mutation_rate", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n_results", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Unused — kept for CLI compatibility. Scoring is serial.")

    args = parser.parse_args()

    # Default smiles file
    if args.smiles_file is None and args.smiles is None:
        default_smi = os.path.join(BASE_DIR, "molecule_data", "psychoactive_molecules.smi")
        if os.path.isfile(default_smi):
            args.smiles_file = default_smi

    if args.smiles_file is None and args.smiles is None:
        parser.error("Provide --smiles and/or --smiles_file (or ensure molecule_data/psychoactive_molecules.smi exists)")

    # Build scorer — use direct import since we may be run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scorer import VinaDockingScorer

    print("Initializing Vina scorer ...")
    scorer = VinaDockingScorer(
        receptor_pdb=args.receptor,
        center=args.center,
        box_size=args.box_size,
        exhaustiveness=args.exhaustiveness,
    )
    print(f"  Receptor: {args.receptor}")
    print(f"  Center:   {scorer.center}")
    print(f"  Box:      {scorer.box_size}")
    print()

    try:
        optimize(
            scorer=scorer,
            population_size=args.population_size,
            offspring_size=args.offspring_size,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            patience=args.patience,
            n_results=args.n_results,
            initial_smiles=args.smiles,
            smiles_file=args.smiles_file,
            seed=args.seed,
        )
    finally:
        scorer.cleanup()


if __name__ == "__main__":
    main()
