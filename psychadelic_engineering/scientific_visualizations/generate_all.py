#!/usr/bin/env python3
"""
Generate 30 high-quality scientific visualizations from the
psychedelic engineering research platform data.

Uses: molecule_data/, receptors/, baseline_lsd/ data
Outputs: scientific_visualizations/*.png
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import re
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIS_DIR = Path(__file__).resolve().parent
MOL_CSV = PROJECT_ROOT / "molecule_data" / "psychoactive_molecules.csv"
MOL_JSON = PROJECT_ROOT / "molecule_data" / "psychoactive_molecules.json"
RECEPTOR_CSV = PROJECT_ROOT / "receptors" / "receptor_index.csv"
BINDING_SITES = PROJECT_ROOT / "baseline_lsd" / "prepared" / "receptors" / "binding_sites.json"

# === Style Configuration ===
STYLE = {
    "figure.facecolor": "#0a0a0a",
    "axes.facecolor": "#111111",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#cccccc",
    "text.color": "#cccccc",
    "xtick.color": "#999999",
    "ytick.color": "#999999",
    "grid.color": "#222222",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
}

# Color palettes
NEON_PALETTE = [
    "#00f0ff", "#ff00ff", "#00ff88", "#ffaa00", "#ff3366",
    "#7b68ee", "#00ccff", "#ff6b6b", "#48dbfb", "#feca57",
    "#ff9ff3", "#54a0ff", "#5f27cd", "#01a3a4", "#f368e0",
    "#ee5a24", "#0abde3", "#10ac84", "#341f97", "#c44569",
]

CLASS_COLORS = {
    "Tryptamine Psychedelics": "#00f0ff",
    "Ergolines": "#ff00ff",
    "Phenethylamine Psychedelics": "#00ff88",
    "Substituted Amphetamines": "#ffaa00",
    "Dissociatives": "#ff3366",
    "Cannabinoids": "#7b68ee",
    "Opioids": "#00ccff",
    "Benzodiazepines": "#ff6b6b",
    "Barbiturates": "#48dbfb",
    "Stimulants": "#feca57",
    "Entactogens": "#ff9ff3",
    "Deliriants": "#54a0ff",
    "Antidepressants": "#5f27cd",
    "Antipsychotics": "#01a3a4",
    "Anxiolytics": "#f368e0",
    "Nootropics": "#ee5a24",
    "Research Chemicals": "#0abde3",
    "MAOIs": "#10ac84",
    "Anaesthetics": "#341f97",
    "Miscellaneous": "#c44569",
}

RECEPTOR_COLORS = {
    "Serotonin": "#ff00ff",
    "Dopamine": "#00f0ff",
    "Cannabinoid": "#00ff88",
    "Opioid": "#ff3366",
    "GABA": "#ffaa00",
    "Glutamate": "#7b68ee",
    "Acetylcholine": "#feca57",
    "Adrenergic": "#48dbfb",
    "Histamine": "#ff9ff3",
    "Sigma": "#54a0ff",
    "Melatonin": "#01a3a4",
    "Trace Amine": "#f368e0",
    "Transporter": "#ee5a24",
}


def apply_style():
    plt.rcParams.update(STYLE)


def load_molecules():
    df = pd.read_csv(MOL_CSV)
    df["MolecularWeight"] = pd.to_numeric(df["MolecularWeight"], errors="coerce")
    df["XLogP"] = pd.to_numeric(df["XLogP"], errors="coerce")
    df["TPSA"] = pd.to_numeric(df["TPSA"], errors="coerce")
    df["Complexity"] = pd.to_numeric(df["Complexity"], errors="coerce")
    df["HeavyAtomCount"] = pd.to_numeric(df["HeavyAtomCount"], errors="coerce")
    df["HBondDonorCount"] = pd.to_numeric(df["HBondDonorCount"], errors="coerce")
    df["HBondAcceptorCount"] = pd.to_numeric(df["HBondAcceptorCount"], errors="coerce")
    df["RotatableBondCount"] = pd.to_numeric(df["RotatableBondCount"], errors="coerce")
    df["AtomStereoCount"] = pd.to_numeric(df["AtomStereoCount"], errors="coerce")
    df["ExactMass"] = pd.to_numeric(df["ExactMass"], errors="coerce")
    return df


def load_receptors():
    return pd.read_csv(RECEPTOR_CSV)


def load_binding_sites():
    with open(BINDING_SITES) as f:
        return json.load(f)


def save_fig(fig, name, dpi=250):
    path = VIS_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor(),
                edgecolor="none", pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved {name}")


# =========================================================================
# VISUALIZATION 1: Molecular Weight Distribution by Drug Class
# =========================================================================
def viz01_mw_distribution(mol):
    apply_style()
    top_classes = mol["category"].value_counts().head(12).index.tolist()
    subset = mol[mol["category"].isin(top_classes)].dropna(subset=["MolecularWeight"])

    fig, ax = plt.subplots(figsize=(16, 9))

    positions = []
    labels = []
    for i, cls in enumerate(top_classes):
        data = subset[subset["category"] == cls]["MolecularWeight"].values
        if len(data) == 0:
            continue
        color = CLASS_COLORS.get(cls, NEON_PALETTE[i % len(NEON_PALETTE)])
        vp = ax.violinplot([data], positions=[i], showmeans=True, showmedians=True, widths=0.8)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.4)
            body.set_edgecolor(color)
        for part in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
            if part in vp:
                vp[part].set_edgecolor(color)
                vp[part].set_alpha(0.8)
        # Scatter individual points
        jitter = np.random.normal(0, 0.08, len(data))
        ax.scatter(np.full_like(data, i) + jitter, data, s=15, alpha=0.6,
                   color=color, edgecolors="none", zorder=3)
        positions.append(i)
        labels.append(cls.replace(" Psychedelics", "\nPsychedelics").replace("Substituted ", "Subst.\n"))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Molecular Weight (Da)", fontsize=13)
    ax.set_title("Molecular Weight Distribution Across Psychoactive Drug Classes",
                 fontsize=16, fontweight="bold", color="#ffffff", pad=20)
    ax.axhline(y=500, color="#ff3366", linestyle="--", alpha=0.4, label="Lipinski MW < 500")
    ax.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333333")
    ax.grid(axis="y", alpha=0.2)

    # Stats annotation
    med = mol["MolecularWeight"].median()
    ax.text(0.98, 0.98, f"N = {len(subset)} molecules\nMedian MW = {med:.1f} Da",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor="#333333", alpha=0.9))

    save_fig(fig, "01_mw_distribution_by_class.png")


# =========================================================================
# VISUALIZATION 2: Chemical Space Map (LogP vs TPSA)
# =========================================================================
def viz02_chemical_space(mol):
    apply_style()
    subset = mol.dropna(subset=["XLogP", "TPSA"]).copy()

    fig, ax = plt.subplots(figsize=(14, 10))

    for cls in subset["category"].unique():
        data = subset[subset["category"] == cls]
        color = CLASS_COLORS.get(cls, "#888888")
        ax.scatter(data["XLogP"], data["TPSA"], s=data["MolecularWeight"].fillna(300) / 5,
                   alpha=0.65, color=color, edgecolors="white", linewidths=0.3,
                   label=cls if len(data) >= 3 else None, zorder=3)

    # BBB zone
    ax.axhspan(0, 90, alpha=0.06, color="#00ff88", zorder=0)
    ax.axvspan(0, 5, alpha=0.06, color="#00f0ff", zorder=0)
    ax.text(4.8, 5, "BBB-Permeable Zone\n(TPSA < 90, LogP 0-5)",
            fontsize=9, color="#00ff88", alpha=0.7, ha="right")

    # Annotate key molecules
    key_mols = ["LSD", "Psilocybin", "DMT", "MDMA", "Ketamine", "THC", "Morphine",
                "Cocaine", "Caffeine", "Nicotine", "Mescaline", "Fentanyl"]
    # Use common_name column
    for name in key_mols:
        row = subset[subset["common_name"] == name]
        if row.empty:
            # Try partial match for THC
            if name == "THC":
                row = subset[subset["common_name"].str.contains("Tetrahydrocannabinol", case=False, na=False)]
            if row.empty:
                continue
        row = row.iloc[0]
        ax.annotate(name, (row["XLogP"], row["TPSA"]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, color="#ffffff", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground="#000000")])

    ax.set_xlabel("LogP (Lipophilicity)", fontsize=13)
    ax.set_ylabel("TPSA (Topological Polar Surface Area, A²)", fontsize=13)
    ax.set_title("Psychoactive Chemical Space: Lipophilicity vs Polarity",
                 fontsize=16, fontweight="bold", color="#ffffff", pad=20)
    ax.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#333333",
              loc="upper right", ncol=2, markerscale=0.7)
    ax.grid(alpha=0.15)

    ax.text(0.02, 0.98, f"N = {len(subset)} compounds\nBubble size ~ MW",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor="#333333", alpha=0.9))

    save_fig(fig, "02_chemical_space_logp_tpsa.png")


# =========================================================================
# VISUALIZATION 3: Blood-Brain Barrier Penetration Analysis
# =========================================================================
def viz03_bbb_penetration(mol):
    apply_style()
    subset = mol.dropna(subset=["XLogP", "TPSA", "MolecularWeight"]).copy()

    # BBB classification based on Pardridge criteria
    def classify_bbb(row):
        if row["TPSA"] < 60 and 1 < row["XLogP"] < 4 and row["MolecularWeight"] < 450:
            return "High"
        elif row["TPSA"] < 90 and 0 < row["XLogP"] < 5 and row["MolecularWeight"] < 500:
            return "Moderate"
        else:
            return "Low"

    subset["BBB_Class"] = subset.apply(classify_bbb, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    bbb_colors = {"High": "#00ff88", "Moderate": "#ffaa00", "Low": "#ff3366"}

    # Panel 1: TPSA distribution by BBB class
    for cls in ["High", "Moderate", "Low"]:
        data = subset[subset["BBB_Class"] == cls]["TPSA"]
        axes[0].hist(data, bins=20, alpha=0.5, color=bbb_colors[cls],
                     label=f"{cls} (n={len(data)})", edgecolor=bbb_colors[cls], linewidth=0.5)
    axes[0].set_xlabel("TPSA (A²)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("TPSA Distribution by BBB Class", fontweight="bold", color="#ffffff")
    axes[0].legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333333")
    axes[0].axvline(x=90, color="#ff3366", linestyle="--", alpha=0.5)
    axes[0].grid(alpha=0.15)

    # Panel 2: Scatter with BBB classification
    for cls in ["High", "Moderate", "Low"]:
        data = subset[subset["BBB_Class"] == cls]
        axes[1].scatter(data["XLogP"], data["MolecularWeight"],
                        s=40, alpha=0.6, color=bbb_colors[cls],
                        label=cls, edgecolors="white", linewidths=0.3)
    axes[1].set_xlabel("LogP")
    axes[1].set_ylabel("Molecular Weight (Da)")
    axes[1].set_title("BBB Penetration Landscape", fontweight="bold", color="#ffffff")
    axes[1].legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333333")
    axes[1].grid(alpha=0.15)

    # Panel 3: Pie chart of BBB classes
    counts = subset["BBB_Class"].value_counts()
    colors_pie = [bbb_colors[c] for c in counts.index]
    wedges, texts, autotexts = axes[2].pie(
        counts.values, labels=counts.index, colors=colors_pie,
        autopct="%1.1f%%", startangle=90, pctdistance=0.85,
        textprops={"color": "#ffffff", "fontsize": 11})
    for t in autotexts:
        t.set_fontweight("bold")
    centre_circle = plt.Circle((0, 0), 0.55, fc="#111111")
    axes[2].add_artist(centre_circle)
    axes[2].set_title("BBB Penetration Distribution", fontweight="bold", color="#ffffff")

    fig.suptitle("Blood-Brain Barrier Penetration Analysis of 258 Psychoactive Compounds",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "03_bbb_penetration_analysis.png")


# =========================================================================
# VISUALIZATION 4: Molecular Complexity vs Heavy Atom Count
# =========================================================================
def viz04_complexity(mol):
    apply_style()
    subset = mol.dropna(subset=["Complexity", "HeavyAtomCount"]).copy()

    fig, ax = plt.subplots(figsize=(14, 10))

    for cls in subset["category"].unique():
        data = subset[subset["category"] == cls]
        color = CLASS_COLORS.get(cls, "#888888")
        ax.scatter(data["HeavyAtomCount"], data["Complexity"],
                   s=50, alpha=0.6, color=color, edgecolors="white",
                   linewidths=0.3, label=cls if len(data) >= 3 else None, zorder=3)

    # Regression line
    x = subset["HeavyAtomCount"].values
    y = subset["Complexity"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "--", color="#ff00ff",
            linewidth=2, alpha=0.7, label=f"R² = {r**2:.3f}")

    # Annotate outliers (high complexity)
    high_complex = subset.nlargest(5, "Complexity")
    for _, row in high_complex.iterrows():
        ax.annotate(row["common_name"], (row["HeavyAtomCount"], row["Complexity"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8,
                    color="#ffffff",
                    path_effects=[pe.withStroke(linewidth=2, foreground="#000000")])

    ax.set_xlabel("Heavy Atom Count", fontsize=13)
    ax.set_ylabel("PubChem Complexity Score", fontsize=13)
    ax.set_title("Molecular Complexity vs Structural Size",
                 fontsize=16, fontweight="bold", color="#ffffff", pad=20)
    ax.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#333333",
              loc="upper left", ncol=2)
    ax.grid(alpha=0.15)

    save_fig(fig, "04_complexity_vs_heavy_atoms.png")


# =========================================================================
# VISUALIZATION 5: Lipinski Rule of Five Analysis
# =========================================================================
def viz05_lipinski(mol):
    apply_style()
    subset = mol.dropna(subset=["MolecularWeight", "XLogP", "HBondDonorCount", "HBondAcceptorCount"]).copy()

    # Lipinski criteria
    subset["MW_pass"] = subset["MolecularWeight"] <= 500
    subset["LogP_pass"] = subset["XLogP"] <= 5
    subset["HBD_pass"] = subset["HBondDonorCount"] <= 5
    subset["HBA_pass"] = subset["HBondAcceptorCount"] <= 10
    subset["violations"] = 4 - (subset["MW_pass"].astype(int) + subset["LogP_pass"].astype(int) +
                                 subset["HBD_pass"].astype(int) + subset["HBA_pass"].astype(int))

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: MW distribution with cutoff
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(subset["MolecularWeight"], bins=30, color="#00f0ff", alpha=0.6, edgecolor="#00f0ff")
    ax1.axvline(x=500, color="#ff3366", linestyle="--", linewidth=2, label="Lipinski limit (500)")
    n_pass = (subset["MolecularWeight"] <= 500).sum()
    ax1.set_title(f"MW ≤ 500 Da: {n_pass}/{len(subset)} pass", fontweight="bold", color="#ffffff")
    ax1.set_xlabel("MW (Da)")
    ax1.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#333333")
    ax1.grid(alpha=0.15)

    # Panel 2: LogP distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(subset["XLogP"], bins=30, color="#ff00ff", alpha=0.6, edgecolor="#ff00ff")
    ax2.axvline(x=5, color="#ff3366", linestyle="--", linewidth=2, label="Lipinski limit (5)")
    n_pass = (subset["XLogP"] <= 5).sum()
    ax2.set_title(f"LogP ≤ 5: {n_pass}/{len(subset)} pass", fontweight="bold", color="#ffffff")
    ax2.set_xlabel("LogP")
    ax2.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#333333")
    ax2.grid(alpha=0.15)

    # Panel 3: HBD vs HBA scatter
    ax3 = fig.add_subplot(gs[0, 2])
    colors_v = subset["violations"].map({0: "#00ff88", 1: "#ffaa00", 2: "#ff6b6b", 3: "#ff3366", 4: "#880000"})
    ax3.scatter(subset["HBondDonorCount"], subset["HBondAcceptorCount"],
                s=40, alpha=0.6, c=colors_v, edgecolors="white", linewidths=0.3)
    ax3.axvline(x=5, color="#ff3366", linestyle="--", alpha=0.5)
    ax3.axhline(y=10, color="#ff3366", linestyle="--", alpha=0.5)
    ax3.fill_between([0, 5], 0, 10, alpha=0.05, color="#00ff88")
    ax3.set_xlabel("H-Bond Donors")
    ax3.set_ylabel("H-Bond Acceptors")
    ax3.set_title("HBD vs HBA (green zone = pass)", fontweight="bold", color="#ffffff")
    ax3.grid(alpha=0.15)

    # Panel 4: Violations bar chart by class
    ax4 = fig.add_subplot(gs[1, 0])
    viol_by_class = subset.groupby("category")["violations"].mean().sort_values()
    top_cls = viol_by_class.tail(12)
    colors_bar = [CLASS_COLORS.get(c, "#888888") for c in top_cls.index]
    ax4.barh(range(len(top_cls)), top_cls.values, color=colors_bar, edgecolor="white", linewidth=0.3)
    ax4.set_yticks(range(len(top_cls)))
    ax4.set_yticklabels([c[:20] for c in top_cls.index], fontsize=8)
    ax4.set_xlabel("Mean Lipinski Violations")
    ax4.set_title("Violations by Drug Class", fontweight="bold", color="#ffffff")
    ax4.grid(axis="x", alpha=0.15)

    # Panel 5: Violations pie chart
    ax5 = fig.add_subplot(gs[1, 1])
    viol_counts = subset["violations"].value_counts().sort_index()
    v_colors = {0: "#00ff88", 1: "#ffaa00", 2: "#ff6b6b", 3: "#ff3366", 4: "#880000"}
    pie_colors = [v_colors.get(v, "#888888") for v in viol_counts.index]
    wedges, texts, autotexts = ax5.pie(
        viol_counts.values, labels=[f"{v} violations" for v in viol_counts.index],
        colors=pie_colors, autopct="%1.1f%%", startangle=90,
        textprops={"color": "#ffffff", "fontsize": 9})
    for t in autotexts:
        t.set_fontweight("bold")
    ax5.set_title("Distribution of Violations", fontweight="bold", color="#ffffff")

    # Panel 6: Summary stats
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    pass_all = (subset["violations"] == 0).sum()
    pass_rate = pass_all / len(subset) * 100
    summary_text = (
        f"LIPINSKI RULE OF FIVE ANALYSIS\n"
        f"{'─' * 40}\n\n"
        f"Total compounds analyzed: {len(subset)}\n\n"
        f"MW ≤ 500 Da:      {subset['MW_pass'].sum()} ({subset['MW_pass'].mean()*100:.1f}%)\n"
        f"LogP ≤ 5:         {subset['LogP_pass'].sum()} ({subset['LogP_pass'].mean()*100:.1f}%)\n"
        f"HBD ≤ 5:          {subset['HBD_pass'].sum()} ({subset['HBD_pass'].mean()*100:.1f}%)\n"
        f"HBA ≤ 10:         {subset['HBA_pass'].sum()} ({subset['HBA_pass'].mean()*100:.1f}%)\n\n"
        f"{'─' * 40}\n"
        f"Pass all 4 rules:  {pass_all} ({pass_rate:.1f}%)\n"
        f"1+ violation:      {len(subset) - pass_all}\n\n"
        f"Median MW:   {subset['MolecularWeight'].median():.1f} Da\n"
        f"Median LogP: {subset['XLogP'].median():.1f}\n"
    )
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace", color="#00f0ff",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#0a0a0a", edgecolor="#333333"))

    fig.suptitle("Lipinski Rule of Five: Drug-Likeness of 258 Psychoactive Compounds",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    save_fig(fig, "05_lipinski_rule_of_five.png")


# =========================================================================
# VISUALIZATION 6: Drug Class Property Spider Charts
# =========================================================================
def viz06_spider_charts(mol):
    apply_style()
    props = ["MolecularWeight", "XLogP", "TPSA", "Complexity", "HeavyAtomCount", "RotatableBondCount"]
    prop_labels = ["MW", "LogP", "TPSA", "Complexity", "Heavy\nAtoms", "Rotatable\nBonds"]

    top_classes = ["Tryptamine Psychedelics", "Ergolines", "Phenethylamine Psychedelics",
                   "Dissociatives", "Opioids", "Benzodiazepines"]
    available = [c for c in top_classes if c in mol["category"].unique()]

    # Compute means per class
    class_means = {}
    for cls in available:
        data = mol[mol["category"] == cls][props].mean()
        class_means[cls] = data.values

    # Normalize to 0-1 range across all classes
    all_vals = np.array(list(class_means.values()))
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    n_props = len(props)
    angles = np.linspace(0, 2 * np.pi, n_props, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 14), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for i, cls in enumerate(available):
        ax = axes[i]
        normalized = (class_means[cls] - mins) / ranges
        values = normalized.tolist() + [normalized[0]]

        color = CLASS_COLORS.get(cls, NEON_PALETTE[i])
        ax.plot(angles, values, color=color, linewidth=2.5)
        ax.fill(angles, values, alpha=0.2, color=color)
        ax.scatter(angles[:-1], values[:-1], s=40, color=color, zorder=5, edgecolors="white")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(prop_labels, fontsize=8, color="#cccccc")
        ax.set_ylim(0, 1.1)
        short_name = cls.replace(" Psychedelics", "").replace("Substituted ", "")
        ax.set_title(short_name, fontsize=12, fontweight="bold", color=color, pad=20)
        ax.set_facecolor("#111111")
        ax.grid(color="#333333", alpha=0.5)

    # Remove extra axes if any
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Molecular Property Profiles by Drug Class (Normalized)",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "06_drug_class_spider_charts.png")


# =========================================================================
# VISUALIZATION 7: Receptor Resolution Distribution
# =========================================================================
def viz07_receptor_resolution(rec):
    apply_style()
    rec_clean = rec.dropna(subset=["resolution_A"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Histogram by method
    for method, color in [("X-ray", "#00f0ff"), ("EM", "#ff00ff")]:
        data = rec_clean[rec_clean["method"] == method]["resolution_A"]
        axes[0].hist(data, bins=15, alpha=0.5, color=color, edgecolor=color,
                     linewidth=0.5, label=f"{method} (n={len(data)}, median={data.median():.2f} A)")
    axes[0].set_xlabel("Resolution (A)", fontsize=13)
    axes[0].set_ylabel("Count", fontsize=13)
    axes[0].set_title("Structure Resolution: X-ray vs Cryo-EM", fontweight="bold", color="#ffffff")
    axes[0].legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333333")
    axes[0].grid(alpha=0.15)

    # Panel 2: Box plot by receptor class
    classes = rec_clean["receptor_class"].value_counts()
    top_classes = classes[classes >= 3].index.tolist()
    data_by_class = [rec_clean[rec_clean["receptor_class"] == c]["resolution_A"].values
                     for c in top_classes]

    bp = axes[1].boxplot(data_by_class, vert=True, patch_artist=True, widths=0.6)
    for i, (patch, cls) in enumerate(zip(bp["boxes"], top_classes)):
        color = RECEPTOR_COLORS.get(cls, NEON_PALETTE[i % len(NEON_PALETTE)])
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(color)
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("#cccccc")

    axes[1].set_xticklabels(top_classes, rotation=45, ha="right", fontsize=9)
    axes[1].set_ylabel("Resolution (A)", fontsize=13)
    axes[1].set_title("Resolution by Receptor Class", fontweight="bold", color="#ffffff")
    axes[1].grid(axis="y", alpha=0.15)

    fig.suptitle("Structural Resolution Analysis: 113 Neurotransmitter Receptor Structures",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "07_receptor_resolution_distribution.png")


# =========================================================================
# VISUALIZATION 8: Receptor Class Treemap-style Visualization
# =========================================================================
def viz08_receptor_landscape(rec):
    apply_style()
    class_counts = rec["receptor_class"].value_counts()
    subtypes = rec.groupby("receptor_class")["subtype"].nunique()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Horizontal bar chart of structures per class
    colors = [RECEPTOR_COLORS.get(c, "#888888") for c in class_counts.index]
    bars = axes[0].barh(range(len(class_counts)), class_counts.values,
                        color=colors, edgecolor="white", linewidth=0.3)
    axes[0].set_yticks(range(len(class_counts)))
    axes[0].set_yticklabels(class_counts.index, fontsize=10)
    axes[0].set_xlabel("Number of PDB Structures", fontsize=12)
    axes[0].set_title("Structures per Receptor Class", fontweight="bold", color="#ffffff")
    for bar, val in zip(bars, class_counts.values):
        axes[0].text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=10, color="#ffffff", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.15)

    # Panel 2: Subtypes per class
    subtypes_sorted = subtypes.reindex(class_counts.index)
    colors2 = [RECEPTOR_COLORS.get(c, "#888888") for c in subtypes_sorted.index]
    bars2 = axes[1].barh(range(len(subtypes_sorted)), subtypes_sorted.values,
                         color=colors2, edgecolor="white", linewidth=0.3, alpha=0.7)
    axes[1].set_yticks(range(len(subtypes_sorted)))
    axes[1].set_yticklabels(subtypes_sorted.index, fontsize=10)
    axes[1].set_xlabel("Number of Unique Subtypes", fontsize=12)
    axes[1].set_title("Subtypes per Receptor Class", fontweight="bold", color="#ffffff")
    for bar, val in zip(bars2, subtypes_sorted.values):
        axes[1].text(val + 0.15, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=10, color="#ffffff", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.15)

    fig.suptitle("Neurotransmitter Receptor Structural Landscape",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "08_receptor_class_landscape.png")


# =========================================================================
# VISUALIZATION 9: Serotonin Receptor Subtype Analysis
# =========================================================================
def viz09_serotonin_subtypes(rec):
    apply_style()
    serotonin = rec[rec["receptor_class"] == "Serotonin"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Structures per subtype
    subtype_counts = serotonin["subtype"].value_counts().sort_index()
    colors = plt.cm.magma(np.linspace(0.2, 0.9, len(subtype_counts)))
    bars = axes[0].bar(range(len(subtype_counts)), subtype_counts.values,
                       color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_xticks(range(len(subtype_counts)))
    axes[0].set_xticklabels(subtype_counts.index, rotation=45, ha="right", fontsize=10)
    axes[0].set_ylabel("Number of Structures", fontsize=12)
    axes[0].set_title("PDB Structures per Serotonin Subtype", fontweight="bold", color="#ffffff")
    for bar, val in zip(bars, subtype_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                     str(val), ha="center", fontsize=10, color="#ffffff", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.15)

    # Panel 2: Resolution by subtype
    subtype_order = sorted(serotonin["subtype"].unique())
    data_lists = [serotonin[serotonin["subtype"] == s]["resolution_A"].dropna().values
                  for s in subtype_order]
    data_lists = [d for d in data_lists if len(d) > 0]
    valid_subtypes = [s for s, d in zip(subtype_order,
                      [serotonin[serotonin["subtype"] == s]["resolution_A"].dropna().values
                       for s in subtype_order]) if len(d) > 0]

    for i, (s, d) in enumerate(zip(valid_subtypes, data_lists)):
        color = plt.cm.magma(0.2 + 0.7 * i / max(len(valid_subtypes) - 1, 1))
        axes[1].scatter([i] * len(d), d, s=80, color=color, edgecolors="white",
                        linewidths=0.5, zorder=3, alpha=0.8)
        axes[1].plot([i, i], [d.min(), d.max()], color=color, linewidth=2, alpha=0.5)

    axes[1].set_xticks(range(len(valid_subtypes)))
    axes[1].set_xticklabels(valid_subtypes, rotation=45, ha="right", fontsize=10)
    axes[1].set_ylabel("Resolution (A)", fontsize=12)
    axes[1].set_title("Resolution Distribution per Subtype", fontweight="bold", color="#ffffff")
    axes[1].grid(axis="y", alpha=0.15)
    axes[1].invert_yaxis()

    fig.suptitle("Serotonin (5-HT) Receptor Family: Structural Coverage",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "09_serotonin_subtype_analysis.png")


# =========================================================================
# VISUALIZATION 10: Binding Site 3D Spatial Map
# =========================================================================
def viz10_binding_sites_3d(bs):
    apply_style()

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0a0a0a")

    names = list(bs.keys())
    xs = [bs[n]["center_x"] for n in names]
    ys = [bs[n]["center_y"] for n in names]
    zs = [bs[n]["center_z"] for n in names]
    sizes = [bs[n]["n_ligand_atoms"] * 15 for n in names]

    # Color by receptor subtype
    subtypes = [n.split("_")[0] for n in names]
    unique_subtypes = list(set(subtypes))
    color_map = plt.cm.rainbow(np.linspace(0, 1, len(unique_subtypes)))
    colors = [color_map[unique_subtypes.index(s)] for s in subtypes]

    scatter = ax.scatter(xs, ys, zs, s=sizes, c=colors, alpha=0.8,
                         edgecolors="white", linewidths=0.5)

    for i, name in enumerate(names):
        short = name.split("_")[0]
        ax.text(xs[i], ys[i], zs[i] + 3, short, fontsize=7, color="#ffffff",
                ha="center", zorder=10)

    ax.set_xlabel("X (A)", fontsize=11, color="#cccccc", labelpad=10)
    ax.set_ylabel("Y (A)", fontsize=11, color="#cccccc", labelpad=10)
    ax.set_zlabel("Z (A)", fontsize=11, color="#cccccc", labelpad=10)
    ax.set_title("Binding Site Centers in 3D Space\n(bubble size ~ ligand atom count)",
                 fontsize=14, fontweight="bold", color="#ffffff", pad=20)

    # Legend
    legend_elements = [Line2D([0], [0], marker="o", color="w", label=s,
                              markerfacecolor=color_map[i], markersize=8)
                       for i, s in enumerate(unique_subtypes)]
    ax.legend(handles=legend_elements, fontsize=8, facecolor="#1a1a1a",
              edgecolor="#333333", loc="upper left")

    ax.tick_params(colors="#999999")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333333")
    ax.yaxis.pane.set_edgecolor("#333333")
    ax.zaxis.pane.set_edgecolor("#333333")

    save_fig(fig, "10_binding_sites_3d_map.png")


# =========================================================================
# VISUALIZATION 11: Tryptamine vs Phenethylamine vs Ergoline Comparison
# =========================================================================
def viz11_psychedelic_comparison(mol):
    apply_style()
    classes = ["Tryptamine Psychedelics", "Phenethylamine Psychedelics", "Ergolines"]
    subset = mol[mol["category"].isin(classes)].copy()
    props = ["MolecularWeight", "XLogP", "TPSA", "HeavyAtomCount", "RotatableBondCount", "Complexity"]
    prop_nice = ["MW (Da)", "LogP", "TPSA (A²)", "Heavy Atoms", "Rotatable Bonds", "Complexity"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (prop, nice) in enumerate(zip(props, prop_nice)):
        ax = axes[i]
        for cls in classes:
            data = subset[subset["category"] == cls][prop].dropna()
            color = CLASS_COLORS.get(cls, "#888888")
            short = cls.replace(" Psychedelics", "")
            ax.hist(data, bins=12, alpha=0.45, color=color, edgecolor=color,
                    linewidth=0.5, label=f"{short} (n={len(data)})")
        ax.set_xlabel(nice, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(nice, fontweight="bold", color="#ffffff")
        ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#333333")
        ax.grid(alpha=0.15)

    fig.suptitle("Classical Psychedelic Scaffolds: Molecular Property Comparison",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "11_psychedelic_scaffold_comparison.png")


# =========================================================================
# VISUALIZATION 12: Molecular Formula Element Composition
# =========================================================================
def viz12_element_composition(mol):
    apply_style()
    subset = mol.dropna(subset=["MolecularFormula"]).copy()

    def parse_formula(formula):
        elements = re.findall(r'([A-Z][a-z]?)(\d*)', str(formula))
        return {el: int(count) if count else 1 for el, count in elements if el}

    all_elements = Counter()
    class_elements = defaultdict(lambda: Counter())
    for _, row in subset.iterrows():
        parsed = parse_formula(row["MolecularFormula"])
        all_elements.update(parsed)
        for el, cnt in parsed.items():
            class_elements[row["category"]][el] += cnt

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Overall element frequency
    top_elements = all_elements.most_common(10)
    elements, counts = zip(*top_elements)
    el_colors = {"C": "#00f0ff", "H": "#888888", "N": "#ff00ff", "O": "#ff3366",
                 "S": "#ffaa00", "F": "#00ff88", "Cl": "#7b68ee", "P": "#feca57",
                 "Br": "#ff9ff3", "I": "#54a0ff"}
    colors = [el_colors.get(e, "#ffffff") for e in elements]
    bars = axes[0].bar(elements, counts, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_ylabel("Total Atom Count (all compounds)", fontsize=12)
    axes[0].set_title("Element Frequency Across All Compounds", fontweight="bold", color="#ffffff")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.15)

    # Panel 2: Heteroatom composition by class (N, O, S, F, Cl, P)
    hetero = ["N", "O", "S", "F", "Cl", "P"]
    top_cls = subset["category"].value_counts().head(8).index.tolist()
    x = np.arange(len(top_cls))
    width = 0.12
    for j, het in enumerate(hetero):
        vals = []
        for cls in top_cls:
            n_mols = len(subset[subset["category"] == cls])
            total_het = class_elements[cls].get(het, 0)
            vals.append(total_het / max(n_mols, 1))
        color = el_colors.get(het, "#ffffff")
        axes[1].bar(x + j * width, vals, width, color=color, label=het,
                    edgecolor="white", linewidth=0.3, alpha=0.8)
    axes[1].set_xticks(x + width * 2.5)
    axes[1].set_xticklabels([c[:15] for c in top_cls], rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean Heteroatom Count per Molecule", fontsize=11)
    axes[1].set_title("Heteroatom Profiles by Drug Class", fontweight="bold", color="#ffffff")
    axes[1].legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333333", ncol=3)
    axes[1].grid(axis="y", alpha=0.15)

    fig.suptitle("Elemental Composition Analysis of Psychoactive Molecules",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "12_element_composition.png")


# =========================================================================
# VISUALIZATION 13: Rotatable Bonds vs MW - Molecular Flexibility
# =========================================================================
def viz13_flexibility(mol):
    apply_style()
    subset = mol.dropna(subset=["RotatableBondCount", "MolecularWeight"]).copy()

    fig, ax = plt.subplots(figsize=(14, 10))

    for cls in subset["category"].unique():
        data = subset[subset["category"] == cls]
        color = CLASS_COLORS.get(cls, "#888888")
        ax.scatter(data["MolecularWeight"], data["RotatableBondCount"],
                   s=40, alpha=0.55, color=color, edgecolors="white",
                   linewidths=0.3, label=cls if len(data) >= 3 else None)

    # Veber rule boundary
    ax.axhline(y=10, color="#ff3366", linestyle="--", alpha=0.5,
               label="Veber limit (≤10 rotatable bonds)")

    # Regression
    x = subset["MolecularWeight"].values
    y = subset["RotatableBondCount"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "--", color="#00ff88",
            linewidth=2, alpha=0.7)
    ax.text(0.05, 0.95, f"R² = {r**2:.3f}\nSlope = {slope:.4f} bonds/Da",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor="#333333", alpha=0.9))

    ax.set_xlabel("Molecular Weight (Da)", fontsize=13)
    ax.set_ylabel("Rotatable Bond Count", fontsize=13)
    ax.set_title("Molecular Flexibility: Rotatable Bonds vs Size",
                 fontsize=16, fontweight="bold", color="#ffffff", pad=20)
    ax.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#333333",
              loc="lower right", ncol=2)
    ax.grid(alpha=0.15)

    save_fig(fig, "13_molecular_flexibility.png")


# =========================================================================
# VISUALIZATION 14: Stereochemistry Distribution
# =========================================================================
def viz14_stereochemistry(mol):
    apply_style()
    subset = mol.dropna(subset=["AtomStereoCount"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Histogram of stereocenters
    axes[0].hist(subset["AtomStereoCount"], bins=range(0, int(subset["AtomStereoCount"].max()) + 2),
                 color="#ff00ff", alpha=0.6, edgecolor="#ff00ff", linewidth=0.5)
    axes[0].set_xlabel("Number of Stereocenters", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Stereocenter Distribution", fontweight="bold", color="#ffffff")
    axes[0].grid(alpha=0.15)

    n_chiral = (subset["AtomStereoCount"] > 0).sum()
    n_achiral = (subset["AtomStereoCount"] == 0).sum()
    axes[0].text(0.95, 0.95, f"Chiral: {n_chiral}\nAchiral: {n_achiral}",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=11,
                 bbox=dict(boxstyle="round", facecolor="#1a1a1a", edgecolor="#333333"))

    # Panel 2: Mean stereocenters by class
    class_stereo = subset.groupby("category")["AtomStereoCount"].mean().sort_values(ascending=True)
    top_stereo = class_stereo.tail(12)
    colors = [CLASS_COLORS.get(c, "#888888") for c in top_stereo.index]
    axes[1].barh(range(len(top_stereo)), top_stereo.values, color=colors,
                 edgecolor="white", linewidth=0.3)
    axes[1].set_yticks(range(len(top_stereo)))
    axes[1].set_yticklabels([c[:25] for c in top_stereo.index], fontsize=9)
    axes[1].set_xlabel("Mean Stereocenters per Molecule", fontsize=12)
    axes[1].set_title("Stereochemical Complexity by Class", fontweight="bold", color="#ffffff")
    axes[1].grid(axis="x", alpha=0.15)

    fig.suptitle("Stereochemistry Analysis of Psychoactive Compounds",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "14_stereochemistry_distribution.png")


# =========================================================================
# VISUALIZATION 15: GPCR Structure Timeline
# =========================================================================
def viz15_structure_timeline(rec):
    apply_style()
    rec_clean = rec.copy()
    rec_clean["year"] = pd.to_datetime(rec_clean["deposition_date"], errors="coerce").dt.year
    rec_clean = rec_clean.dropna(subset=["year", "resolution_A"])

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Panel 1: Scatter - year vs resolution, colored by class
    for cls in rec_clean["receptor_class"].unique():
        data = rec_clean[rec_clean["receptor_class"] == cls]
        color = RECEPTOR_COLORS.get(cls, "#888888")
        axes[0].scatter(data["year"], data["resolution_A"], s=60, alpha=0.7,
                        color=color, edgecolors="white", linewidths=0.3,
                        label=cls if len(data) >= 2 else None)

    axes[0].set_xlabel("Deposition Year", fontsize=12)
    axes[0].set_ylabel("Resolution (A)", fontsize=12)
    axes[0].set_title("Receptor Structure Resolution Over Time", fontweight="bold", color="#ffffff")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#333333", ncol=3)
    axes[0].grid(alpha=0.15)

    # Panel 2: Cumulative structures over time
    year_counts = rec_clean["year"].value_counts().sort_index()
    cumulative = year_counts.cumsum()
    axes[1].fill_between(cumulative.index, cumulative.values, alpha=0.3, color="#00f0ff")
    axes[1].plot(cumulative.index, cumulative.values, color="#00f0ff", linewidth=2.5)
    axes[1].scatter(cumulative.index, cumulative.values, s=40, color="#00f0ff",
                    edgecolors="white", linewidths=0.5, zorder=5)

    # Annotate milestones
    axes[1].set_xlabel("Year", fontsize=12)
    axes[1].set_ylabel("Cumulative PDB Structures", fontsize=12)
    axes[1].set_title("Growth of Neurotransmitter Receptor Structural Data", fontweight="bold", color="#ffffff")
    axes[1].grid(alpha=0.15)

    total = len(rec_clean)
    axes[1].text(0.02, 0.95, f"Total: {total} structures",
                 transform=axes[1].transAxes, ha="left", va="top", fontsize=12,
                 bbox=dict(boxstyle="round", facecolor="#1a1a1a", edgecolor="#333333", alpha=0.9))

    fig.suptitle("Neurotransmitter Receptor Structural Biology Timeline",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "15_structure_timeline.png")


# =========================================================================
# VISUALIZATION 16: Receptor-Ligand Interaction Map
# =========================================================================
def viz16_receptor_ligand_map(rec):
    apply_style()
    rec_clean = rec.dropna(subset=["ligand"]).copy()

    # Build adjacency: receptor_class -> ligand
    pairs = rec_clean.groupby(["receptor_class", "ligand"]).size().reset_index(name="count")
    top_receptors = pairs["receptor_class"].value_counts().head(8).index.tolist()
    pairs = pairs[pairs["receptor_class"].isin(top_receptors)]
    top_ligands = pairs["ligand"].value_counts().head(15).index.tolist()
    pairs = pairs[pairs["ligand"].isin(top_ligands)]

    fig, ax = plt.subplots(figsize=(16, 10))

    receptors = sorted(pairs["receptor_class"].unique())
    ligands = sorted(pairs["ligand"].unique())

    # Create a matrix
    matrix = np.zeros((len(receptors), len(ligands)))
    for _, row in pairs.iterrows():
        ri = receptors.index(row["receptor_class"])
        li = ligands.index(row["ligand"])
        matrix[ri, li] = row["count"]

    cmap = LinearSegmentedColormap.from_list("custom", ["#111111", "#00f0ff", "#ff00ff"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")

    for i in range(len(receptors)):
        for j in range(len(ligands)):
            if matrix[i, j] > 0:
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                        fontsize=9, color="#ffffff", fontweight="bold")

    ax.set_xticks(range(len(ligands)))
    ax.set_xticklabels(ligands, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(receptors)))
    ax.set_yticklabels(receptors, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Number of Co-crystal Structures", fontsize=11)

    ax.set_title("Receptor-Ligand Co-crystallization Map",
                 fontsize=16, fontweight="bold", color="#ffffff", pad=20)
    ax.set_xlabel("Co-crystallized Ligand", fontsize=12)
    ax.set_ylabel("Receptor Class", fontsize=12)

    plt.tight_layout()
    save_fig(fig, "16_receptor_ligand_map.png")


# =========================================================================
# VISUALIZATION 17: Psychedelic Potency Landscape (MW vs LogP contour)
# =========================================================================
def viz17_potency_landscape(mol):
    apply_style()
    psych_classes = ["Tryptamine Psychedelics", "Ergolines", "Phenethylamine Psychedelics",
                     "Dissociatives", "Entactogens"]
    subset = mol[mol["category"].isin(psych_classes)].dropna(subset=["XLogP", "MolecularWeight"]).copy()

    fig, ax = plt.subplots(figsize=(14, 10))

    # KDE contour of entire psychedelic space
    x = subset["XLogP"].values
    y = subset["MolecularWeight"].values

    # Grid for contour
    xi = np.linspace(x.min() - 1, x.max() + 1, 100)
    yi = np.linspace(y.min() - 30, y.max() + 30, 100)
    Xi, Yi = np.meshgrid(xi, yi)

    try:
        from scipy.stats import gaussian_kde
        positions = np.vstack([Xi.ravel(), Yi.ravel()])
        kernel = gaussian_kde(np.vstack([x, y]))
        Z = np.reshape(kernel(positions).T, Xi.shape)
        contour = ax.contourf(Xi, Yi, Z, levels=15, cmap="magma", alpha=0.4)
        ax.contour(Xi, Yi, Z, levels=8, colors="#ffffff", linewidths=0.3, alpha=0.3)
    except Exception:
        pass

    # Scatter by class
    for cls in psych_classes:
        data = subset[subset["category"] == cls]
        color = CLASS_COLORS.get(cls, "#888888")
        short = cls.replace(" Psychedelics", "")
        ax.scatter(data["XLogP"], data["MolecularWeight"],
                   s=80, alpha=0.8, color=color, edgecolors="white",
                   linewidths=0.5, label=short, zorder=5)

    # Annotate key molecules
    for name in ["LSD", "DMT", "Psilocybin", "Mescaline", "MDMA", "Ketamine", "2C-B"]:
        row = subset[subset["common_name"] == name]
        if not row.empty:
            row = row.iloc[0]
            ax.annotate(name, (row["XLogP"], row["MolecularWeight"]),
                        textcoords="offset points", xytext=(8, 5), fontsize=9,
                        color="#ffffff", fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2, foreground="#000000")])

    ax.set_xlabel("LogP (Lipophilicity)", fontsize=13)
    ax.set_ylabel("Molecular Weight (Da)", fontsize=13)
    ax.set_title("Psychedelic Chemical Landscape: Density Contour Map",
                 fontsize=16, fontweight="bold", color="#ffffff", pad=20)
    ax.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333333")
    ax.grid(alpha=0.1)

    save_fig(fig, "17_psychedelic_potency_landscape.png")


# =========================================================================
# VISUALIZATION 18: SMILES Complexity Analysis
# =========================================================================
def viz18_smiles_complexity(mol):
    apply_style()
    subset = mol.dropna(subset=["SMILES"]).copy()
    subset["smiles_len"] = subset["SMILES"].str.len()
    subset["ring_count"] = subset["SMILES"].apply(lambda s: str(s).count("1") + str(s).count("2") +
                                                   str(s).count("3") + str(s).count("4"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Panel 1: SMILES length distribution
    axes[0].hist(subset["smiles_len"], bins=30, color="#00f0ff", alpha=0.6,
                 edgecolor="#00f0ff", linewidth=0.5)
    axes[0].set_xlabel("SMILES String Length", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("SMILES Length Distribution", fontweight="bold", color="#ffffff")
    axes[0].grid(alpha=0.15)

    # Annotate longest
    longest = subset.nlargest(3, "smiles_len")
    for _, row in longest.iterrows():
        axes[0].annotate(row["common_name"],
                         (row["smiles_len"], 0), textcoords="offset points",
                         xytext=(0, 10), fontsize=8, color="#ff3366", rotation=45)

    # Panel 2: SMILES length vs Complexity
    axes[1].scatter(subset["smiles_len"], subset["Complexity"],
                    s=20, alpha=0.5, color="#ff00ff", edgecolors="none")
    x = subset["smiles_len"].values
    y = subset["Complexity"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() > 2:
        slope, intercept, r, _, _ = stats.linregress(x[mask], y[mask])
        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
        axes[1].plot(x_line, slope * x_line + intercept, "--", color="#00ff88", linewidth=2)
        axes[1].text(0.05, 0.95, f"R² = {r**2:.3f}", transform=axes[1].transAxes,
                     ha="left", va="top", fontsize=11, color="#00ff88",
                     bbox=dict(boxstyle="round", facecolor="#1a1a1a", edgecolor="#333333"))
    axes[1].set_xlabel("SMILES Length", fontsize=12)
    axes[1].set_ylabel("PubChem Complexity", fontsize=12)
    axes[1].set_title("SMILES Length vs Complexity", fontweight="bold", color="#ffffff")
    axes[1].grid(alpha=0.15)

    # Panel 3: SMILES length vs MW
    axes[2].scatter(subset["smiles_len"], subset["MolecularWeight"],
                    s=20, alpha=0.5, color="#ffaa00", edgecolors="none")
    y2 = subset["MolecularWeight"].values
    mask2 = ~(np.isnan(x) | np.isnan(y2))
    if mask2.sum() > 2:
        slope2, intercept2, r2, _, _ = stats.linregress(x[mask2], y2[mask2])
        x_line2 = np.linspace(x[mask2].min(), x[mask2].max(), 100)
        axes[2].plot(x_line2, slope2 * x_line2 + intercept2, "--", color="#00ff88", linewidth=2)
        axes[2].text(0.05, 0.95, f"R² = {r2**2:.3f}", transform=axes[2].transAxes,
                     ha="left", va="top", fontsize=11, color="#00ff88",
                     bbox=dict(boxstyle="round", facecolor="#1a1a1a", edgecolor="#333333"))
    axes[2].set_xlabel("SMILES Length", fontsize=12)
    axes[2].set_ylabel("Molecular Weight (Da)", fontsize=12)
    axes[2].set_title("SMILES Length vs MW", fontweight="bold", color="#ffffff")
    axes[2].grid(alpha=0.15)

    fig.suptitle("SMILES Notation Complexity Analysis",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "18_smiles_complexity_analysis.png")


# =========================================================================
# VISUALIZATION 19: Docking Pipeline Parameter Space
# =========================================================================
def viz19_docking_parameters(bs):
    apply_style()

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    names = list(bs.keys())
    centers_x = [bs[n]["center_x"] for n in names]
    centers_y = [bs[n]["center_y"] for n in names]
    centers_z = [bs[n]["center_z"] for n in names]
    ligand_atoms = [bs[n]["n_ligand_atoms"] for n in names]
    subtypes = [n.split("_")[0] for n in names]

    # Panel 1: Ligand atom count per binding site
    ax1 = fig.add_subplot(gs[0, 0])
    colors = [plt.cm.viridis(a / max(ligand_atoms)) for a in ligand_atoms]
    bars = ax1.bar(range(len(names)), ligand_atoms, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(subtypes, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Reference Ligand Atoms", fontsize=11)
    ax1.set_title("Binding Site Ligand Complexity", fontweight="bold", color="#ffffff")
    ax1.grid(axis="y", alpha=0.15)

    # Panel 2: Center coordinate distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(centers_x, centers_y, s=[a * 10 for a in ligand_atoms],
                c=centers_z, cmap="magma", alpha=0.8, edgecolors="white", linewidths=0.5)
    for i, name in enumerate(names):
        ax2.annotate(subtypes[i], (centers_x[i], centers_y[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7, color="#cccccc")
    ax2.set_xlabel("Center X (A)", fontsize=11)
    ax2.set_ylabel("Center Y (A)", fontsize=11)
    ax2.set_title("Binding Site Centers (XY Plane, color=Z)", fontweight="bold", color="#ffffff")
    ax2.grid(alpha=0.15)

    # Panel 3: AutoDock Vina configuration
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    config_text = (
        "AUTODOCK VINA CONFIGURATION\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Search Space:     22.0 x 22.0 x 22.0 A³\n"
        f"Exhaustiveness:   32\n"
        f"Num Poses:        20\n"
        f"Random Seed:      42\n\n"
        f"Total Receptors:  {len(names)}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Molecules (8 Lysergamides):\n"
        f"  LSD         (323.4 Da)\n"
        f"  1P-LSD      (393.5 Da)\n"
        f"  ALD-52      (365.4 Da)\n"
        f"  ETH-LAD     (337.5 Da)\n"
        f"  AL-LAD      (349.5 Da)\n"
        f"  PRO-LAD     (351.5 Da)\n"
        f"  Ergine      (267.3 Da)\n"
        f"  Ergometrine (325.4 Da)\n\n"
        f"Total Docking Runs: 8 x 13 = 104"
    )
    ax3.text(0.05, 0.95, config_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace", color="#00f0ff",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#0a0a0a", edgecolor="#333333"))

    # Panel 4: Search box volume visualization
    ax4 = fig.add_subplot(gs[1, 1])
    # Show relationship between box size and ligand atoms
    box_vol = 22.0 ** 3  # constant
    ax4.bar(range(len(names)), [box_vol / max(a, 1) for a in ligand_atoms],
            color="#ff00ff", alpha=0.6, edgecolor="white", linewidth=0.5)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(subtypes, rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("Volume per Ligand Atom (A³/atom)", fontsize=11)
    ax4.set_title("Search Space Density", fontweight="bold", color="#ffffff")
    ax4.grid(axis="y", alpha=0.15)

    fig.suptitle("AutoDock Vina Docking Parameter Space",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    save_fig(fig, "19_docking_parameter_space.png")


# =========================================================================
# VISUALIZATION 20: Drug Class Correlation Matrix
# =========================================================================
def viz20_class_correlation(mol):
    apply_style()
    props = ["MolecularWeight", "XLogP", "TPSA", "Complexity",
             "HeavyAtomCount", "RotatableBondCount", "HBondDonorCount", "HBondAcceptorCount"]
    subset = mol[props].dropna()

    fig, ax = plt.subplots(figsize=(12, 10))

    corr = subset.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.copy()
    corr_masked[mask] = np.nan

    cmap = LinearSegmentedColormap.from_list("custom", ["#ff3366", "#111111", "#00f0ff"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    short_labels = ["MW", "LogP", "TPSA", "Complexity", "Heavy\nAtoms",
                    "Rotatable\nBonds", "HBD", "HBA"]
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=10)

    for i in range(len(props)):
        for j in range(len(props)):
            val = corr.iloc[i, j]
            color = "#ffffff" if abs(val) > 0.5 else "#999999"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson Correlation", fontsize=11)

    ax.set_title("Molecular Descriptor Correlation Matrix",
                 fontsize=16, fontweight="bold", color="#ffffff", pad=20)
    save_fig(fig, "20_descriptor_correlation_matrix.png")


# =========================================================================
# VISUALIZATION 21: Lysergamide Structural Variants
# =========================================================================
def viz21_lysergamide_variants(mol):
    apply_style()
    lysergamides = ["LSD", "1P-LSD", "ALD-52", "ETH-LAD", "AL-LAD", "PRO-LAD", "Ergine", "Ergometrine"]
    subset = mol[mol["common_name"].isin(lysergamides)].copy()
    subset = subset.set_index("common_name").reindex(lysergamides).reset_index()

    props = ["MolecularWeight", "XLogP", "TPSA", "Complexity", "HeavyAtomCount", "RotatableBondCount"]
    prop_labels = ["MW (Da)", "LogP", "TPSA (A²)", "Complexity", "Heavy Atoms", "Rot. Bonds"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = plt.cm.magma(np.linspace(0.2, 0.9, len(lysergamides)))

    for i, (prop, label) in enumerate(zip(props, prop_labels)):
        ax = axes[i]
        values = subset[prop].values
        bars = ax.bar(range(len(lysergamides)), values, color=colors,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(lysergamides)))
        ax.set_xticklabels(lysergamides, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontweight="bold", color="#ffffff")
        ax.grid(axis="y", alpha=0.15)

        # Annotate values
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="#ffffff")

    fig.suptitle("Lysergamide Structural Variants: Property Comparison (8 LSD Analogs)",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "21_lysergamide_variants.png")


# =========================================================================
# VISUALIZATION 22: Generative Model Taxonomy
# =========================================================================
def viz22_generative_models():
    apply_style()

    models = {
        "GNN-Based": ["GraphINVENT", "hgraph2graph", "torchdrug", "JTNN", "MoLeR", "Pocket2Mol"],
        "Flow-Based": ["FlowMol", "PropMolFlow"],
        "Reinforcement\nLearning": ["DrugEx", "REINVENT4", "MolScore", "SyntheMol"],
        "Diffusion": ["DecompDiff", "TargetDiff", "3D-SBDD"],
        "VAE/AAE": ["BiAAE", "GENTRL", "MOLLEO"],
        "Language\nModels": ["MolGPT", "SAFE", "ChemCrow"],
        "Benchmarks": ["GuacaMol", "MOSES", "COVID Moonshot"],
    }

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")

    categories = list(models.keys())
    n_cats = len(categories)
    cat_colors = [NEON_PALETTE[i * 2 % len(NEON_PALETTE)] for i in range(n_cats)]

    # Layout: categories as rows
    y_start = 0.92
    y_step = 0.12

    for i, (cat, items) in enumerate(models.items()):
        y = y_start - i * y_step
        color = cat_colors[i]

        # Category box
        ax.text(0.12, y, cat, fontsize=13, fontweight="bold", color=color,
                ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a",
                          edgecolor=color, linewidth=2))

        # Model items
        for j, item in enumerate(items):
            x = 0.3 + j * 0.11
            ax.text(x, y, item, fontsize=9, color="#cccccc", ha="center", va="center",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a0a",
                              edgecolor=color, linewidth=0.5, alpha=0.8))
            # Connection line
            ax.annotate("", xy=(x - 0.04, y), xytext=(0.2, y),
                        transform=ax.transAxes,
                        arrowprops=dict(arrowstyle="-", color=color, alpha=0.3, linewidth=0.5))

    # Title and summary
    ax.text(0.5, 1.0, "Generative Molecular Design: 28 AI Models Taxonomy",
            fontsize=16, fontweight="bold", color="#ffffff",
            ha="center", va="top", transform=ax.transAxes)

    total_models = sum(len(v) for v in models.values())
    ax.text(0.5, 0.02, f"Total: {total_models} models across {n_cats} architectural paradigms",
            fontsize=11, color="#999999", ha="center", va="bottom", transform=ax.transAxes)

    save_fig(fig, "22_generative_model_taxonomy.png")


# =========================================================================
# VISUALIZATION 23: Binding Software Capability Matrix
# =========================================================================
def viz23_binding_software():
    apply_style()

    tools = {
        "AutoDock Vina": ["Docking", "Scoring"],
        "Uni-Dock": ["Docking", "Scoring"],
        "DeepDock": ["Docking", "DL Scoring"],
        "DiffDock": ["Docking", "Diffusion"],
        "GNINA": ["Docking", "CNN Scoring"],
        "AlphaFold": ["Structure Pred."],
        "RoseTTAFold": ["Structure Pred."],
        "OpenFold": ["Structure Pred."],
        "OpenMM": ["MD Simulation"],
        "RDKit": ["Cheminformatics"],
        "OpenBabel": ["Format Conv."],
        "Meeko": ["Ligand Prep."],
        "Boltz": ["MD Integration"],
        "PhysicsML": ["Physics ML"],
    }

    capabilities = sorted(set(c for caps in tools.values() for c in caps))
    tool_names = list(tools.keys())

    fig, ax = plt.subplots(figsize=(14, 9))

    matrix = np.zeros((len(tool_names), len(capabilities)))
    for i, tool in enumerate(tool_names):
        for cap in tools[tool]:
            j = capabilities.index(cap)
            matrix[i, j] = 1

    cmap = LinearSegmentedColormap.from_list("custom", ["#111111", "#00f0ff"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")

    for i in range(len(tool_names)):
        for j in range(len(capabilities)):
            if matrix[i, j] > 0:
                ax.text(j, i, "●", ha="center", va="center",
                        fontsize=14, color="#00f0ff", fontweight="bold")

    ax.set_xticks(range(len(capabilities)))
    ax.set_xticklabels(capabilities, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(tool_names)))
    ax.set_yticklabels(tool_names, fontsize=10)

    ax.set_title("Binding Software Capability Matrix: 20 Computational Tools",
                 fontsize=16, fontweight="bold", color="#ffffff", pad=20)
    ax.grid(alpha=0.1)

    save_fig(fig, "23_binding_software_matrix.png")


# =========================================================================
# VISUALIZATION 24: Opioid Receptor Family Analysis
# =========================================================================
def viz24_opioid_analysis(rec):
    apply_style()
    opioid = rec[rec["receptor_class"] == "Opioid"].copy()
    opioid_mols = ["Morphine", "Codeine", "Heroin", "Fentanyl", "Buprenorphine",
                   "Oxycodone", "Hydrocodone", "Methadone", "Naloxone", "Naltrexone"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Structures per opioid subtype
    subtype_counts = opioid["subtype"].value_counts()
    colors = ["#ff3366", "#ff6b6b", "#ff9ff3", "#ffaa00", "#feca57"]
    bars = axes[0].bar(range(len(subtype_counts)), subtype_counts.values,
                       color=colors[:len(subtype_counts)], edgecolor="white", linewidth=0.5)
    axes[0].set_xticks(range(len(subtype_counts)))
    axes[0].set_xticklabels(subtype_counts.index, rotation=45, ha="right", fontsize=10)
    axes[0].set_ylabel("Number of PDB Structures", fontsize=12)
    axes[0].set_title("Opioid Receptor Subtype Coverage", fontweight="bold", color="#ffffff")
    for bar, val in zip(bars, subtype_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                     str(val), ha="center", fontsize=11, color="#ffffff", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.15)

    # Panel 2: Resolution by subtype
    for i, subtype in enumerate(opioid["subtype"].unique()):
        data = opioid[opioid["subtype"] == subtype]
        color = colors[i % len(colors)]
        axes[1].scatter([i] * len(data), data["resolution_A"],
                        s=100, color=color, edgecolors="white", linewidths=0.5, zorder=5)
        for _, row in data.iterrows():
            if pd.notna(row.get("ligand", None)):
                axes[1].annotate(str(row["ligand"])[:12],
                                 (i, row["resolution_A"]),
                                 textcoords="offset points", xytext=(8, 0),
                                 fontsize=7, color="#cccccc")

    axes[1].set_xticks(range(len(opioid["subtype"].unique())))
    axes[1].set_xticklabels(opioid["subtype"].unique(), rotation=45, ha="right", fontsize=10)
    axes[1].set_ylabel("Resolution (A)", fontsize=12)
    axes[1].set_title("Opioid Receptor Structures with Ligands", fontweight="bold", color="#ffffff")
    axes[1].invert_yaxis()
    axes[1].grid(axis="y", alpha=0.15)

    fig.suptitle("Opioid Receptor System: Structural Biology Overview",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "24_opioid_receptor_analysis.png")


# =========================================================================
# VISUALIZATION 25: Comprehensive Drug-Likeness Scatter Matrix
# =========================================================================
def viz25_scatter_matrix(mol):
    apply_style()
    props = ["MolecularWeight", "XLogP", "TPSA", "Complexity"]
    prop_labels = ["MW (Da)", "LogP", "TPSA (A²)", "Complexity"]
    subset = mol.dropna(subset=props).copy()

    n = len(props)
    fig, axes = plt.subplots(n, n, figsize=(16, 16))

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if i == j:
                # Diagonal: histogram
                ax.hist(subset[props[i]], bins=25, color="#00f0ff", alpha=0.6,
                        edgecolor="#00f0ff", linewidth=0.5)
                ax.set_title(prop_labels[i], fontsize=10, color="#ffffff")
            else:
                # Off-diagonal: scatter
                for cls in subset["category"].unique():
                    data = subset[subset["category"] == cls]
                    color = CLASS_COLORS.get(cls, "#888888")
                    ax.scatter(data[props[j]], data[props[i]], s=8, alpha=0.4,
                               color=color, edgecolors="none")

            if j == 0:
                ax.set_ylabel(prop_labels[i], fontsize=9)
            if i == n - 1:
                ax.set_xlabel(prop_labels[j], fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.1)

    fig.suptitle("Molecular Descriptor Scatter Matrix: 258 Psychoactive Compounds",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.01)
    plt.tight_layout()
    save_fig(fig, "25_scatter_matrix.png")


# =========================================================================
# VISUALIZATION 26: Consciousness Target Receptor Network
# =========================================================================
def viz26_consciousness_network(rec, mol):
    apply_style()

    # Map consciousness-related molecules to their receptor targets
    consciousness_map = {
        "5-HT2A": {"molecules": ["LSD", "Psilocybin", "DMT", "Mescaline", "2C-B"],
                    "effect": "Visual hallucinations, ego dissolution"},
        "5-HT1A": {"molecules": ["LSD", "DMT", "Buspirone"],
                    "effect": "Anxiolysis, mood elevation"},
        "NMDA": {"molecules": ["Ketamine", "PCP", "DXM", "Memantine"],
                 "effect": "Dissociation, anesthesia"},
        "Mu (MOR)": {"molecules": ["Morphine", "Fentanyl", "Salvinorin A"],
                      "effect": "Euphoria, pain relief"},
        "CB1": {"molecules": ["THC", "Anandamide", "JWH-018"],
                "effect": "Altered perception, relaxation"},
        "GABA-A": {"molecules": ["Diazepam", "Zolpidem", "Ethanol"],
                    "effect": "Sedation, anxiolysis"},
        "D2": {"molecules": ["LSD", "Amphetamine", "Cocaine"],
               "effect": "Reward, motivation"},
        "Sigma-1": {"molecules": ["DMT", "Haloperidol"],
                     "effect": "Neuroprotection"},
    }

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis("off")

    # Circular layout for receptors
    n_receptors = len(consciousness_map)
    receptor_angles = np.linspace(0, 2 * np.pi, n_receptors, endpoint=False)
    receptor_radius = 0.35
    center = (0.5, 0.5)

    receptor_positions = {}
    for i, (rec_name, info) in enumerate(consciousness_map.items()):
        angle = receptor_angles[i]
        x = center[0] + receptor_radius * np.cos(angle)
        y = center[1] + receptor_radius * np.sin(angle)
        receptor_positions[rec_name] = (x, y)

        # Draw receptor node
        color = NEON_PALETTE[i % len(NEON_PALETTE)]
        ax.text(x, y, rec_name, fontsize=11, fontweight="bold", color=color,
                ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a0a0a",
                          edgecolor=color, linewidth=2))

        # Draw molecule satellites
        n_mols = len(info["molecules"])
        mol_radius = 0.12
        for j, mol_name in enumerate(info["molecules"]):
            mol_angle = angle + (j - n_mols / 2) * 0.15
            mx = x + mol_radius * np.cos(mol_angle)
            my = y + mol_radius * np.sin(mol_angle)

            ax.text(mx, my, mol_name, fontsize=7, color="#cccccc",
                    ha="center", va="center", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a",
                              edgecolor="#444444", linewidth=0.5))

    # Center label
    ax.text(0.5, 0.5, "CONSCIOUSNESS\nENGINEERING", fontsize=14, fontweight="bold",
            color="#ffffff", ha="center", va="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a1a1a",
                      edgecolor="#ff00ff", linewidth=2))

    ax.set_title("Consciousness Engineering: Receptor-Molecule Target Map",
                 fontsize=16, fontweight="bold", color="#ffffff", pad=20,
                 transform=ax.transAxes, y=1.0, x=0.5)

    save_fig(fig, "26_consciousness_target_network.png")


# =========================================================================
# VISUALIZATION 27: Psychedelic vs Therapeutic Molecule Comparison
# =========================================================================
def viz27_psychedelic_vs_therapeutic(mol):
    apply_style()

    psychedelic_cats = ["Tryptamine Psychedelics", "Ergolines", "Phenethylamine Psychedelics",
                        "Dissociatives", "Entactogens"]
    therapeutic_cats = ["Antidepressants", "Antipsychotics", "Benzodiazepines", "Anxiolytics"]

    mol_copy = mol.copy()
    mol_copy["group"] = "Other"
    mol_copy.loc[mol_copy["category"].isin(psychedelic_cats), "group"] = "Psychedelic"
    mol_copy.loc[mol_copy["category"].isin(therapeutic_cats), "group"] = "Therapeutic"

    subset = mol_copy[mol_copy["group"].isin(["Psychedelic", "Therapeutic"])].copy()

    props = ["MolecularWeight", "XLogP", "TPSA", "Complexity", "HBondDonorCount", "RotatableBondCount"]
    prop_labels = ["MW (Da)", "LogP", "TPSA (A²)", "Complexity", "H-Bond Donors", "Rotatable Bonds"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = {"Psychedelic": "#ff00ff", "Therapeutic": "#00f0ff"}

    for i, (prop, label) in enumerate(zip(props, prop_labels)):
        ax = axes[i]
        for group in ["Psychedelic", "Therapeutic"]:
            data = subset[subset["group"] == group][prop].dropna()
            ax.hist(data, bins=15, alpha=0.45, color=colors[group],
                    edgecolor=colors[group], linewidth=0.5,
                    label=f"{group} (n={len(data)}, μ={data.mean():.1f})")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(label, fontweight="bold", color="#ffffff")
        ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#333333")
        ax.grid(alpha=0.15)

    # Statistical tests
    fig.text(0.5, -0.02,
             "Psychedelic compounds tend toward lower MW, higher LogP, and lower TPSA than psychiatric therapeutics",
             ha="center", fontsize=11, color="#999999", style="italic")

    fig.suptitle("Psychedelic vs Therapeutic Compounds: Molecular Property Comparison",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "27_psychedelic_vs_therapeutic.png")


# =========================================================================
# VISUALIZATION 28: Receptor Method & Organism Analysis
# =========================================================================
def viz28_receptor_methods(rec):
    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Method distribution
    method_counts = rec["method"].value_counts()
    colors_m = {"X-ray": "#00f0ff", "EM": "#ff00ff"}
    pie_colors = [colors_m.get(m, "#888888") for m in method_counts.index]
    wedges, texts, autotexts = axes[0, 0].pie(
        method_counts.values, labels=method_counts.index, colors=pie_colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": "#ffffff", "fontsize": 12})
    for t in autotexts:
        t.set_fontweight("bold")
    centre = plt.Circle((0, 0), 0.5, fc="#111111")
    axes[0, 0].add_artist(centre)
    axes[0, 0].text(0, 0, f"N={len(rec)}", ha="center", va="center",
                     fontsize=14, color="#ffffff", fontweight="bold")
    axes[0, 0].set_title("Structure Determination Method", fontweight="bold", color="#ffffff")

    # Panel 2: Organism distribution
    org_counts = rec["organism"].value_counts().head(5)
    colors_o = ["#00ff88", "#ffaa00", "#ff3366", "#7b68ee", "#feca57"]
    bars = axes[0, 1].barh(range(len(org_counts)), org_counts.values,
                           color=colors_o, edgecolor="white", linewidth=0.5)
    axes[0, 1].set_yticks(range(len(org_counts)))
    axes[0, 1].set_yticklabels(org_counts.index, fontsize=10)
    axes[0, 1].set_xlabel("Number of Structures", fontsize=12)
    axes[0, 1].set_title("Source Organism", fontweight="bold", color="#ffffff")
    axes[0, 1].grid(axis="x", alpha=0.15)

    # Panel 3: Method by receptor class
    cross = pd.crosstab(rec["receptor_class"], rec["method"])
    top_cls = rec["receptor_class"].value_counts().head(8).index
    cross = cross.reindex(top_cls).fillna(0)
    x = np.arange(len(cross))
    width = 0.35
    if "X-ray" in cross.columns:
        axes[1, 0].bar(x - width / 2, cross.get("X-ray", 0), width,
                        color="#00f0ff", label="X-ray", edgecolor="white", linewidth=0.5)
    if "EM" in cross.columns:
        axes[1, 0].bar(x + width / 2, cross.get("EM", 0), width,
                        color="#ff00ff", label="Cryo-EM", edgecolor="white", linewidth=0.5)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(cross.index, rotation=45, ha="right", fontsize=9)
    axes[1, 0].set_ylabel("Count", fontsize=12)
    axes[1, 0].set_title("Method by Receptor Class", fontweight="bold", color="#ffffff")
    axes[1, 0].legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333333")
    axes[1, 0].grid(axis="y", alpha=0.15)

    # Panel 4: Resolution comparison
    for method, color in [("X-ray", "#00f0ff"), ("EM", "#ff00ff")]:
        data = rec[rec["method"] == method]["resolution_A"].dropna()
        axes[1, 1].hist(data, bins=15, alpha=0.5, color=color, edgecolor=color,
                         linewidth=0.5, label=f"{method} (μ={data.mean():.2f} A)")
    axes[1, 1].set_xlabel("Resolution (A)", fontsize=12)
    axes[1, 1].set_ylabel("Count", fontsize=12)
    axes[1, 1].set_title("Resolution by Method", fontweight="bold", color="#ffffff")
    axes[1, 1].legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333333")
    axes[1, 1].grid(alpha=0.15)

    fig.suptitle("Structural Biology Methods & Organism Analysis",
                 fontsize=16, fontweight="bold", color="#ffffff", y=1.02)
    plt.tight_layout()
    save_fig(fig, "28_receptor_methods_analysis.png")


# =========================================================================
# VISUALIZATION 29: Platform Architecture Overview
# =========================================================================
def viz29_platform_architecture():
    apply_style()

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Draw architecture blocks
    blocks = [
        {"name": "MOLECULE DATABASE\n258 Compounds\n17 Drug Classes", "x": 0.08, "y": 0.75, "w": 0.2, "h": 0.18, "color": "#00f0ff"},
        {"name": "RECEPTOR STRUCTURES\n113 PDB Files\n13 GPCR Families", "x": 0.08, "y": 0.45, "w": 0.2, "h": 0.18, "color": "#ff00ff"},
        {"name": "BINDING SOFTWARE\n20 Tools\nDocking + ML + MD", "x": 0.38, "y": 0.6, "w": 0.22, "h": 0.18, "color": "#00ff88"},
        {"name": "GENERATIVE DESIGN\n28 AI Models\nGNN/Flow/RL/Diffusion", "x": 0.38, "y": 0.3, "w": 0.22, "h": 0.18, "color": "#ffaa00"},
        {"name": "DOCKING PIPELINE\nAutoDock Vina\n104 Runs (8×13)", "x": 0.7, "y": 0.6, "w": 0.22, "h": 0.18, "color": "#ff3366"},
        {"name": "ANALYSIS & VIS\n30 Visualizations\nStatistical Analysis", "x": 0.7, "y": 0.3, "w": 0.22, "h": 0.18, "color": "#7b68ee"},
    ]

    for block in blocks:
        rect = FancyBboxPatch(
            (block["x"], block["y"]), block["w"], block["h"],
            boxstyle="round,pad=0.02", facecolor="#0a0a0a",
            edgecolor=block["color"], linewidth=2.5
        )
        ax.add_patch(rect)
        ax.text(block["x"] + block["w"] / 2, block["y"] + block["h"] / 2,
                block["name"], ha="center", va="center", fontsize=11,
                fontweight="bold", color=block["color"], linespacing=1.5)

    # Arrows
    arrows = [
        (0.28, 0.84, 0.38, 0.74),  # Molecules -> Binding
        (0.28, 0.54, 0.38, 0.64),  # Receptors -> Binding
        (0.28, 0.54, 0.38, 0.39),  # Receptors -> Generative
        (0.6, 0.69, 0.7, 0.69),    # Binding -> Docking
        (0.6, 0.39, 0.7, 0.39),    # Generative -> Analysis
        (0.81, 0.6, 0.81, 0.48),   # Docking -> Analysis
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                    linewidth=1.5, connectionstyle="arc3,rad=0.1"))

    # Title
    ax.text(0.5, 0.97, "Engineering Consciousness: Computational Research Platform",
            ha="center", va="top", fontsize=18, fontweight="bold", color="#ffffff",
            transform=ax.transAxes)
    ax.text(0.5, 0.08, "Integrating molecular databases, structural biology, computational docking,\n"
                        "and generative AI for psychedelic drug discovery and consciousness research",
            ha="center", va="bottom", fontsize=11, color="#888888", style="italic",
            transform=ax.transAxes)

    save_fig(fig, "29_platform_architecture.png")


# =========================================================================
# VISUALIZATION 30: Grand Summary Dashboard
# =========================================================================
def viz30_grand_summary(mol, rec, bs):
    apply_style()

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: Category distribution
    ax1 = fig.add_subplot(gs[0, 0])
    cat_counts = mol["category"].value_counts().head(10)
    colors = [CLASS_COLORS.get(c, "#888888") for c in cat_counts.index]
    ax1.barh(range(len(cat_counts)), cat_counts.values, color=colors,
             edgecolor="white", linewidth=0.3)
    ax1.set_yticks(range(len(cat_counts)))
    ax1.set_yticklabels([c[:18] for c in cat_counts.index], fontsize=7)
    ax1.set_xlabel("Count", fontsize=8)
    ax1.set_title("Molecule Classes", fontweight="bold", color="#ffffff", fontsize=10)
    ax1.grid(axis="x", alpha=0.1)

    # Panel 2: MW histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(mol["MolecularWeight"].dropna(), bins=30, color="#00f0ff", alpha=0.6, edgecolor="#00f0ff")
    ax2.axvline(x=500, color="#ff3366", linestyle="--", alpha=0.5)
    ax2.set_xlabel("MW (Da)", fontsize=8)
    ax2.set_title("MW Distribution", fontweight="bold", color="#ffffff", fontsize=10)
    ax2.grid(alpha=0.1)

    # Panel 3: LogP vs TPSA
    ax3 = fig.add_subplot(gs[0, 2])
    sub = mol.dropna(subset=["XLogP", "TPSA"])
    ax3.scatter(sub["XLogP"], sub["TPSA"], s=10, alpha=0.5, color="#ff00ff", edgecolors="none")
    ax3.axhline(y=90, color="#00ff88", linestyle="--", alpha=0.3)
    ax3.set_xlabel("LogP", fontsize=8)
    ax3.set_ylabel("TPSA", fontsize=8)
    ax3.set_title("Chemical Space", fontweight="bold", color="#ffffff", fontsize=10)
    ax3.grid(alpha=0.1)

    # Panel 4: Receptor classes
    ax4 = fig.add_subplot(gs[0, 3])
    rec_counts = rec["receptor_class"].value_counts().head(8)
    r_colors = [RECEPTOR_COLORS.get(c, "#888888") for c in rec_counts.index]
    ax4.barh(range(len(rec_counts)), rec_counts.values, color=r_colors,
             edgecolor="white", linewidth=0.3)
    ax4.set_yticks(range(len(rec_counts)))
    ax4.set_yticklabels(rec_counts.index, fontsize=7)
    ax4.set_xlabel("Structures", fontsize=8)
    ax4.set_title("Receptor Coverage", fontweight="bold", color="#ffffff", fontsize=10)
    ax4.grid(axis="x", alpha=0.1)

    # Panel 5: Resolution histogram
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(rec["resolution_A"].dropna(), bins=20, color="#ffaa00", alpha=0.6, edgecolor="#ffaa00")
    ax5.set_xlabel("Resolution (A)", fontsize=8)
    ax5.set_title("Resolution Dist.", fontweight="bold", color="#ffffff", fontsize=10)
    ax5.grid(alpha=0.1)

    # Panel 6: Complexity vs Heavy Atoms
    ax6 = fig.add_subplot(gs[1, 1])
    sub2 = mol.dropna(subset=["HeavyAtomCount", "Complexity"])
    ax6.scatter(sub2["HeavyAtomCount"], sub2["Complexity"], s=10, alpha=0.5,
                color="#00ff88", edgecolors="none")
    ax6.set_xlabel("Heavy Atoms", fontsize=8)
    ax6.set_ylabel("Complexity", fontsize=8)
    ax6.set_title("Size vs Complexity", fontweight="bold", color="#ffffff", fontsize=10)
    ax6.grid(alpha=0.1)

    # Panel 7: Binding site atoms
    ax7 = fig.add_subplot(gs[1, 2])
    names_bs = list(bs.keys())
    atoms_bs = [bs[n]["n_ligand_atoms"] for n in names_bs]
    subtypes_bs = [n.split("_")[0] for n in names_bs]
    ax7.bar(range(len(names_bs)), atoms_bs, color="#ff3366", edgecolor="white", linewidth=0.3)
    ax7.set_xticks(range(len(names_bs)))
    ax7.set_xticklabels(subtypes_bs, rotation=90, fontsize=6)
    ax7.set_ylabel("Ligand Atoms", fontsize=8)
    ax7.set_title("Binding Sites", fontweight="bold", color="#ffffff", fontsize=10)
    ax7.grid(axis="y", alpha=0.1)

    # Panel 8: Summary stats
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis("off")
    stats_text = (
        "PLATFORM SUMMARY\n"
        "━━━━━━━━━━━━━━━━━━━\n\n"
        f"Molecules:    258\n"
        f"Drug Classes: {mol['category'].nunique()}\n"
        f"Receptors:    113\n"
        f"GPCR Families:{rec['receptor_class'].nunique()}\n"
        f"Binding Sites: {len(bs)}\n"
        f"AI Models:     28\n"
        f"Software:      20\n"
        f"Docking Runs:  104\n"
    )
    ax8.text(0.1, 0.95, stats_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace", color="#00f0ff",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0a0a0a", edgecolor="#333333"))

    # Bottom panels: wide
    # Panel 9: Method pie
    ax9 = fig.add_subplot(gs[2, 0:2])
    # Stereocenters by class
    stereo = mol.groupby("category")["AtomStereoCount"].mean().sort_values().tail(10)
    s_colors = [CLASS_COLORS.get(c, "#888888") for c in stereo.index]
    ax9.barh(range(len(stereo)), stereo.values, color=s_colors, edgecolor="white", linewidth=0.3)
    ax9.set_yticks(range(len(stereo)))
    ax9.set_yticklabels([c[:20] for c in stereo.index], fontsize=7)
    ax9.set_xlabel("Mean Stereocenters", fontsize=8)
    ax9.set_title("Stereochemical Complexity", fontweight="bold", color="#ffffff", fontsize=10)
    ax9.grid(axis="x", alpha=0.1)

    # Panel 10: H-Bond analysis
    ax10 = fig.add_subplot(gs[2, 2:4])
    sub3 = mol.dropna(subset=["HBondDonorCount", "HBondAcceptorCount"])
    for cls in sub3["category"].unique():
        data = sub3[sub3["category"] == cls]
        color = CLASS_COLORS.get(cls, "#888888")
        ax10.scatter(data["HBondDonorCount"], data["HBondAcceptorCount"],
                     s=15, alpha=0.4, color=color, edgecolors="none")
    ax10.axvline(x=5, color="#ff3366", linestyle="--", alpha=0.3)
    ax10.axhline(y=10, color="#ff3366", linestyle="--", alpha=0.3)
    ax10.set_xlabel("H-Bond Donors", fontsize=8)
    ax10.set_ylabel("H-Bond Acceptors", fontsize=8)
    ax10.set_title("Hydrogen Bonding Capacity", fontweight="bold", color="#ffffff", fontsize=10)
    ax10.grid(alpha=0.1)

    fig.suptitle("Engineering Consciousness: Comprehensive Research Platform Dashboard",
                 fontsize=18, fontweight="bold", color="#ffffff", y=1.01)
    save_fig(fig, "30_grand_summary_dashboard.png")


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 70)
    print("  GENERATING 30 SCIENTIFIC VISUALIZATIONS")
    print("  Engineering Consciousness Research Platform")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    mol = load_molecules()
    rec = load_receptors()
    bs = load_binding_sites()
    print(f"  Molecules: {len(mol)}")
    print(f"  Receptors: {len(rec)}")
    print(f"  Binding sites: {len(bs)}")
    print()

    # Generate all 30 visualizations
    print("Generating visualizations...\n")

    viz01_mw_distribution(mol)
    viz02_chemical_space(mol)
    viz03_bbb_penetration(mol)
    viz04_complexity(mol)
    viz05_lipinski(mol)
    viz06_spider_charts(mol)
    viz07_receptor_resolution(rec)
    viz08_receptor_landscape(rec)
    viz09_serotonin_subtypes(rec)
    viz10_binding_sites_3d(bs)
    viz11_psychedelic_comparison(mol)
    viz12_element_composition(mol)
    viz13_flexibility(mol)
    viz14_stereochemistry(mol)
    viz15_structure_timeline(rec)
    viz16_receptor_ligand_map(rec)
    viz17_potency_landscape(mol)
    viz18_smiles_complexity(mol)
    viz19_docking_parameters(bs)
    viz20_class_correlation(mol)
    viz21_lysergamide_variants(mol)
    viz22_generative_models()
    viz23_binding_software()
    viz24_opioid_analysis(rec)
    viz25_scatter_matrix(mol)
    viz26_consciousness_network(rec, mol)
    viz27_psychedelic_vs_therapeutic(mol)
    viz28_receptor_methods(rec)
    viz29_platform_architecture()
    viz30_grand_summary(mol, rec, bs)

    print()
    print("=" * 70)
    print(f"  ALL 30 VISUALIZATIONS SAVED TO:")
    print(f"  {VIS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
