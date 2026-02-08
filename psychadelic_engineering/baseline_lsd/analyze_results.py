#!/usr/bin/env python3
"""
Analyze docking results and generate 7 visualizations.

1. Heatmap: all molecules × all receptor subtypes
2. Bar chart: LSD binding affinity across all 11 subtypes
3. Grouped bars: all 8 variants compared at 5-HT2A
4. Radar/spider plot: LSD vs Ergine vs ETH-LAD selectivity profiles
5. Scatter plot: molecular weight vs 5-HT2A affinity (SAR) with regression
6. Validation scatter: primary vs validation structure correlation
7. Selectivity heatmap: affinities normalized to each molecule's 5-HT2A value
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

from config import MOLECULES, RECEPTORS, RESULTS_CSV, RESULTS_JSON, VIS_DIR


# Ordered subtypes for consistent axis ordering
SUBTYPE_ORDER = [
    "5-HT1A", "5-HT1B", "5-HT1D", "5-HT1E", "5-HT1F",
    "5-HT2A", "5-HT2B", "5-HT2C",
    "5-HT4", "5-HT6", "5-HT7",
]

MOLECULE_ORDER = list(MOLECULES.keys())


def load_results() -> pd.DataFrame:
    """Load docking results from CSV."""
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
    elif RESULTS_JSON.exists():
        df = pd.DataFrame(json.loads(RESULTS_JSON.read_text()))
    else:
        print("[ERROR] No results file found. Run run_docking.py first.")
        sys.exit(1)
    return df


def build_affinity_matrix(df: pd.DataFrame, validation: bool = False) -> pd.DataFrame:
    """Build molecule × subtype affinity matrix from results."""
    filtered = df[df["is_validation"] == validation].copy()
    pivot = filtered.pivot_table(
        index="molecule", columns="subtype", values="best_affinity", aggfunc="first"
    )
    # Reorder
    mols = [m for m in MOLECULE_ORDER if m in pivot.index]
    subs = [s for s in SUBTYPE_ORDER if s in pivot.columns]
    return pivot.reindex(index=mols, columns=subs)


def plot1_heatmap(df: pd.DataFrame):
    """1. Full heatmap: all molecules × all receptor subtypes."""
    matrix = build_affinity_matrix(df, validation=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    vmin, vmax = matrix.min().min(), matrix.max().max()

    # Use diverging colormap centered at median
    mid = matrix.values[~np.isnan(matrix.values)].mean() if not matrix.empty else -7
    norm = TwoSlopeNorm(vmin=vmin, vcenter=mid, vmax=max(vmax, mid + 0.1))

    im = ax.imshow(matrix.values, cmap="RdYlGn_r", norm=norm, aspect="auto")

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                color = "white" if abs(val - mid) > (vmax - vmin) * 0.3 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Binding Affinity (kcal/mol)", fontsize=11)

    ax.set_title("LSD Variants: Binding Affinity Across Serotonin Receptor Subtypes",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Receptor Subtype", fontsize=12)
    ax.set_ylabel("Molecule", fontsize=12)

    plt.tight_layout()
    path = VIS_DIR / "01_affinity_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1/7] Saved {path.name}")


def plot2_lsd_bar(df: pd.DataFrame):
    """2. Bar chart: LSD binding affinity across all 11 subtypes."""
    lsd = df[(df["molecule"] == "LSD") & (df["is_validation"] == False)].copy()
    lsd = lsd.set_index("subtype").reindex(SUBTYPE_ORDER)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(SUBTYPE_ORDER)))

    bars = ax.bar(range(len(SUBTYPE_ORDER)), lsd["best_affinity"].values,
                  color=colors, edgecolor="black", linewidth=0.5)

    # Annotate bars
    for bar, val in zip(bars, lsd["best_affinity"].values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val - 0.15,
                    f"{val:.1f}", ha="center", va="top", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(SUBTYPE_ORDER)))
    ax.set_xticklabels(SUBTYPE_ORDER, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Binding Affinity (kcal/mol)", fontsize=12)
    ax.set_title("LSD Binding Affinity Across Serotonin Receptor Subtypes",
                 fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.invert_yaxis()  # More negative = stronger binding → at top
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = VIS_DIR / "02_lsd_receptor_profile.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2/7] Saved {path.name}")


def plot3_grouped_5ht2a(df: pd.DataFrame):
    """3. Grouped bars: all 8 variants compared at 5-HT2A."""
    ht2a = df[(df["subtype"] == "5-HT2A") & (df["is_validation"] == False)].copy()
    ht2a = ht2a.set_index("molecule").reindex(MOLECULE_ORDER)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(MOLECULE_ORDER)))

    x = range(len(MOLECULE_ORDER))
    bars = ax.bar(x, ht2a["best_affinity"].values, color=colors,
                  edgecolor="black", linewidth=0.5, width=0.7)

    for bar, val in zip(bars, ht2a["best_affinity"].values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val - 0.1,
                    f"{val:.1f}", ha="center", va="top", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MOLECULE_ORDER, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Binding Affinity (kcal/mol)", fontsize=12)
    ax.set_title("5-HT2A Binding Affinity: LSD vs Structural Variants",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = VIS_DIR / "03_5ht2a_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3/7] Saved {path.name}")


def plot4_radar(df: pd.DataFrame):
    """4. Radar/spider plot: LSD vs Ergine vs ETH-LAD selectivity profiles."""
    matrix = build_affinity_matrix(df, validation=False)
    molecules_to_plot = ["LSD", "Ergine", "ETH-LAD"]
    molecules_to_plot = [m for m in molecules_to_plot if m in matrix.index]

    if not molecules_to_plot:
        print("  [4/7] SKIPPED: no molecules for radar plot")
        return

    subtypes = list(matrix.columns)
    n = len(subtypes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    linestyles = ["-", "--", "-."]

    for i, mol in enumerate(molecules_to_plot):
        values = matrix.loc[mol].values.tolist()
        # Invert so stronger binding (more negative) = larger radius
        values_inv = [-v if not np.isnan(v) else 0 for v in values]
        values_inv += values_inv[:1]

        ax.plot(angles, values_inv, color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2, label=mol)
        ax.fill(angles, values_inv, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(subtypes, fontsize=10)
    ax.set_title("Receptor Selectivity Profile\n(larger radius = stronger binding)",
                 fontsize=14, fontweight="bold", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.set_ylabel("-Affinity (kcal/mol)", fontsize=10, labelpad=30)

    plt.tight_layout()
    path = VIS_DIR / "04_selectivity_radar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4/7] Saved {path.name}")


def plot5_mw_vs_affinity(df: pd.DataFrame):
    """5. Scatter plot: molecular weight vs 5-HT2A binding affinity (SAR)."""
    ht2a = df[(df["subtype"] == "5-HT2A") & (df["is_validation"] == False)].copy()
    ht2a = ht2a.dropna(subset=["best_affinity", "mw"])

    if len(ht2a) < 2:
        print("  [5/7] SKIPPED: insufficient data for SAR plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(ht2a["mw"], ht2a["best_affinity"], s=120, c="#e74c3c",
               edgecolors="black", zorder=5, alpha=0.8)

    # Annotate each point
    for _, row in ht2a.iterrows():
        ax.annotate(row["molecule"],
                    (row["mw"], row["best_affinity"]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=10, fontweight="bold")

    # Linear regression
    slope, intercept, r_value, p_value, _ = stats.linregress(
        ht2a["mw"], ht2a["best_affinity"]
    )
    x_line = np.linspace(ht2a["mw"].min() - 10, ht2a["mw"].max() + 10, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "k--", alpha=0.5, linewidth=1.5,
            label=f"R² = {r_value**2:.3f}, p = {p_value:.3f}")

    ax.set_xlabel("Molecular Weight (Da)", fontsize=12)
    ax.set_ylabel("5-HT2A Binding Affinity (kcal/mol)", fontsize=12)
    ax.set_title("Structure-Activity: Molecular Weight vs 5-HT2A Binding",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    path = VIS_DIR / "05_mw_vs_affinity_sar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5/7] Saved {path.name}")


def plot6_validation(df: pd.DataFrame):
    """6. Validation scatter: primary vs validation structure correlation."""
    primary = df[df["is_validation"] == False].copy()
    validation = df[df["is_validation"] == True].copy()

    if validation.empty:
        print("  [6/7] SKIPPED: no validation data")
        return

    # Merge primary and validation for same molecule-subtype pairs
    merged = pd.merge(
        primary[["molecule", "subtype", "best_affinity"]],
        validation[["molecule", "subtype", "best_affinity"]],
        on=["molecule", "subtype"],
        suffixes=("_primary", "_validation"),
    ).dropna()

    if len(merged) < 2:
        print("  [6/7] SKIPPED: insufficient paired data")
        return

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.scatter(merged["best_affinity_primary"], merged["best_affinity_validation"],
               s=100, c="#2ecc71", edgecolors="black", zorder=5)

    for _, row in merged.iterrows():
        ax.annotate(f"{row['molecule']}\n({row['subtype']})",
                    (row["best_affinity_primary"], row["best_affinity_validation"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)

    # Identity line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.4, label="y = x (perfect agreement)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Correlation
    if len(merged) >= 3:
        r, p = stats.pearsonr(merged["best_affinity_primary"],
                              merged["best_affinity_validation"])
        ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\np = {p:.3f}",
                transform=ax.transAxes, fontsize=12, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Primary Structure Affinity (kcal/mol)", fontsize=12)
    ax.set_ylabel("Validation Structure Affinity (kcal/mol)", fontsize=12)
    ax.set_title("Validation: Primary vs Alternative PDB Structure",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    path = VIS_DIR / "06_validation_correlation.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [6/7] Saved {path.name}")


def plot7_selectivity_heatmap(df: pd.DataFrame):
    """7. Selectivity heatmap: affinities normalized to each molecule's 5-HT2A value."""
    matrix = build_affinity_matrix(df, validation=False)

    if "5-HT2A" not in matrix.columns:
        print("  [7/7] SKIPPED: 5-HT2A column missing")
        return

    # Normalize: ΔΔG = affinity - molecule's 5-HT2A affinity
    # Positive values mean weaker binding than at 5-HT2A
    ht2a_ref = matrix["5-HT2A"]
    selectivity = matrix.subtract(ht2a_ref, axis=0)

    fig, ax = plt.subplots(figsize=(14, 7))

    vmax_abs = max(abs(selectivity.min().min()), abs(selectivity.max().max()), 0.1)
    norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0, vmax=vmax_abs)

    im = ax.imshow(selectivity.values, cmap="RdBu", norm=norm, aspect="auto")

    for i in range(selectivity.shape[0]):
        for j in range(selectivity.shape[1]):
            val = selectivity.iloc[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax_abs * 0.5 else "black"
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(selectivity.columns)))
    ax.set_xticklabels(selectivity.columns, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(selectivity.index)))
    ax.set_yticklabels(selectivity.index, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("ΔAffinity vs 5-HT2A (kcal/mol)\n(blue = more selective, red = less)",
                   fontsize=10)

    ax.set_title("Receptor Selectivity: Affinity Relative to 5-HT2A",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Receptor Subtype", fontsize=12)
    ax.set_ylabel("Molecule", fontsize=12)

    plt.tight_layout()
    path = VIS_DIR / "07_selectivity_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [7/7] Saved {path.name}")


def main():
    print("=" * 60)
    print("ANALYSIS & VISUALIZATION")
    print("=" * 60)

    df = load_results()
    print(f"Loaded {len(df)} results from {RESULTS_CSV}")
    print(f"Molecules: {df['molecule'].nunique()}")
    print(f"Receptors: {df['receptor'].nunique()}")

    n_valid = df["best_affinity"].notna().sum()
    n_fail = df["best_affinity"].isna().sum()
    print(f"Valid results: {n_valid}, Failed: {n_fail}")
    print()

    # Quick summary stats
    primary = df[df["is_validation"] == False]
    if not primary.empty:
        print("Quick stats (primary structures only):")
        best = primary.loc[primary["best_affinity"].idxmin()]
        worst = primary.loc[primary["best_affinity"].idxmax()]
        print(f"  Strongest binding: {best['molecule']} @ {best['subtype']}: "
              f"{best['best_affinity']:.1f} kcal/mol")
        print(f"  Weakest binding:   {worst['molecule']} @ {worst['subtype']}: "
              f"{worst['best_affinity']:.1f} kcal/mol")

        # LSD pharmacology rank check
        lsd_data = primary[primary["molecule"] == "LSD"].set_index("subtype")
        rank_subtypes = ["5-HT2A", "5-HT2B", "5-HT2C", "5-HT1A"]
        available = [s for s in rank_subtypes if s in lsd_data.index]
        if len(available) >= 2:
            print(f"\n  LSD pharmacology ranking (expected: 2A > 2B > 2C > 1A):")
            for s in available:
                val = lsd_data.loc[s, "best_affinity"]
                print(f"    {s}: {val:.1f} kcal/mol")

    print()
    print("Generating visualizations...")

    plot1_heatmap(df)
    plot2_lsd_bar(df)
    plot3_grouped_5ht2a(df)
    plot4_radar(df)
    plot5_mw_vs_affinity(df)
    plot6_validation(df)
    plot7_selectivity_heatmap(df)

    print()
    print(f"All visualizations saved to: {VIS_DIR}")


if __name__ == "__main__":
    main()
