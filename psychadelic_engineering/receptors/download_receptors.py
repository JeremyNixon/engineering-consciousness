"""
Download high-resolution neuronal receptor structures from RCSB PDB.

Covers serotonin (5-HT), dopamine, cannabinoid, opioid, GABA, glutamate,
acetylcholine, sigma, adrenergic, and histamine receptors relevant to
psychoactive compound binding studies.
"""

import requests
import json
import os
import time
import csv

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PDB_DIR = os.path.join(OUTPUT_DIR, "pdb_files")
RCSB_DOWNLOAD = "https://files.rcsb.org/download"
RCSB_API = "https://data.rcsb.org/rest/v1/core/entry"

os.makedirs(PDB_DIR, exist_ok=True)

# Comprehensive receptor list with curated PDB IDs
# Format: (PDB_ID, receptor_name, subtype, description, organism, ligand)
RECEPTORS = [
    # =============================================
    # SEROTONIN (5-HT) RECEPTORS
    # =============================================
    # 5-HT1A
    ("7E2Y", "Serotonin", "5-HT1A", "Active state bound to aripiprazole, Gi protein", "Homo sapiens", "Aripiprazole"),
    ("7E2Z", "Serotonin", "5-HT1A", "Active state bound to serotonin, Gi protein", "Homo sapiens", "Serotonin"),

    # 5-HT1B
    ("4IAR", "Serotonin", "5-HT1B", "Bound to ergotamine, active state", "Homo sapiens", "Ergotamine"),
    ("4IAQ", "Serotonin", "5-HT1B", "Bound to dihydroergotamine", "Homo sapiens", "Dihydroergotamine"),
    ("5V54", "Serotonin", "5-HT1B", "Bound to donitriptan agonist", "Homo sapiens", "Donitriptan"),

    # 5-HT1D
    ("7E32", "Serotonin", "5-HT1D", "Active state with Gi protein", "Homo sapiens", ""),

    # 5-HT1E
    ("7E33", "Serotonin", "5-HT1E", "Active state with Gi protein", "Homo sapiens", ""),

    # 5-HT1F
    ("7EXD", "Serotonin", "5-HT1F", "Bound to lasmiditan, Gi complex", "Homo sapiens", "Lasmiditan"),

    # 5-HT2A (key target for classical psychedelics)
    ("6WHA", "Serotonin", "5-HT2A", "Bound to LSD, active state", "Homo sapiens", "LSD"),
    ("6WH4", "Serotonin", "5-HT2A", "Bound to 25-CN-NBOH, Gq complex", "Homo sapiens", "25-CN-NBOH"),
    ("7WC4", "Serotonin", "5-HT2A", "Bound to psilocin", "Homo sapiens", "Psilocin"),
    ("7WC5", "Serotonin", "5-HT2A", "Bound to serotonin", "Homo sapiens", "Serotonin"),
    ("7WC6", "Serotonin", "5-HT2A", "Bound to lisuride (non-hallucinogenic)", "Homo sapiens", "Lisuride"),
    ("7WC7", "Serotonin", "5-HT2A", "Bound to methiothepin (antagonist)", "Homo sapiens", "Methiothepin"),
    ("8DPG", "Serotonin", "5-HT2A", "Bound to DMT", "Homo sapiens", "DMT"),
    ("7VOD", "Serotonin", "5-HT2A", "Inactive state with risperidone", "Homo sapiens", "Risperidone"),
    ("7VOE", "Serotonin", "5-HT2A", "Inactive state with zotepine", "Homo sapiens", "Zotepine"),

    # 5-HT2B
    ("4IB4", "Serotonin", "5-HT2B", "Bound to ergotamine", "Homo sapiens", "Ergotamine"),
    ("5TUD", "Serotonin", "5-HT2B", "Bound to LSD", "Homo sapiens", "LSD"),
    ("4NC3", "Serotonin", "5-HT2B", "Bound to ergotamine, different crystal form", "Homo sapiens", "Ergotamine"),

    # 5-HT2C
    ("6BQG", "Serotonin", "5-HT2C", "Bound to ergotamine, active state", "Homo sapiens", "Ergotamine"),
    ("6BQH", "Serotonin", "5-HT2C", "Bound to ritanserin, inactive state", "Homo sapiens", "Ritanserin"),

    # 5-HT3 (ionotropic)
    ("6DG8", "Serotonin", "5-HT3A", "Apo state ion channel", "Mus musculus", ""),
    ("6NP0", "Serotonin", "5-HT3A", "Bound to serotonin, open state", "Mus musculus", "Serotonin"),
    ("6HIN", "Serotonin", "5-HT3A", "Bound to granisetron antagonist", "Mus musculus", "Granisetron"),

    # 5-HT4
    ("7XT8", "Serotonin", "5-HT4", "Active state with Gs protein", "Homo sapiens", "Serotonin"),

    # 5-HT6
    ("7XTC", "Serotonin", "5-HT6", "Active state with Gs protein", "Homo sapiens", ""),

    # 5-HT7
    ("7XTA", "Serotonin", "5-HT7", "Active state with Gs protein", "Homo sapiens", "5-CT"),

    # =============================================
    # DOPAMINE RECEPTORS
    # =============================================
    # D1
    ("7CKW", "Dopamine", "D1", "Active state with Gs protein, SKF-81297 agonist", "Homo sapiens", "SKF-81297"),
    ("7CKZ", "Dopamine", "D1", "Active state with Gs protein, fenoldopam", "Homo sapiens", "Fenoldopam"),
    ("7LJC", "Dopamine", "D1", "Bound to non-catechol agonist", "Homo sapiens", ""),
    ("7F0T", "Dopamine", "D1", "Bound to tavapadon, biased agonist", "Homo sapiens", "Tavapadon"),

    # D2
    ("6CM4", "Dopamine", "D2", "Bound to risperidone (antipsychotic)", "Homo sapiens", "Risperidone"),
    ("6LUQ", "Dopamine", "D2", "Bound to haloperidol", "Homo sapiens", "Haloperidol"),
    ("7DFP", "Dopamine", "D2", "Active state with Gi protein, bromocriptine", "Homo sapiens", "Bromocriptine"),
    ("7JVR", "Dopamine", "D2", "Active state with Gi, rotigotine", "Homo sapiens", "Rotigotine"),

    # D3
    ("3PBL", "Dopamine", "D3", "Bound to eticlopride antagonist", "Homo sapiens", "Eticlopride"),
    ("7CMU", "Dopamine", "D3", "Active state with Gi protein, pramipexole", "Homo sapiens", "Pramipexole"),

    # D4
    ("5WIU", "Dopamine", "D4", "Bound to nemonapride antagonist", "Homo sapiens", "Nemonapride"),
    ("5WIV", "Dopamine", "D4", "Bound to L-745870 antagonist", "Homo sapiens", "L-745870"),

    # D5
    ("7LJD", "Dopamine", "D5", "Active state with Gs protein", "Homo sapiens", ""),

    # =============================================
    # CANNABINOID RECEPTORS
    # =============================================
    # CB1
    ("5TGZ", "Cannabinoid", "CB1", "Bound to AM6538 antagonist", "Homo sapiens", "AM6538"),
    ("5XRA", "Cannabinoid", "CB1", "Bound to AM841 agonist", "Homo sapiens", "AM841"),
    ("6N4B", "Cannabinoid", "CB1", "Bound to MDMB-Fubinaca, Gi complex", "Homo sapiens", "MDMB-Fubinaca"),
    ("5U09", "Cannabinoid", "CB1", "Bound to taranabant inverse agonist", "Homo sapiens", "Taranabant"),
    ("6KPG", "Cannabinoid", "CB1", "Bound to CP55940 agonist, Gi", "Homo sapiens", "CP55940"),

    # CB2
    ("5ZTY", "Cannabinoid", "CB2", "Bound to AM10257 antagonist", "Homo sapiens", "AM10257"),
    ("6KPC", "Cannabinoid", "CB2", "Bound to WIN 55212-2, Gi complex", "Homo sapiens", "WIN 55212-2"),
    ("6PT0", "Cannabinoid", "CB2", "Bound to AM12033 agonist, Gi", "Homo sapiens", "AM12033"),

    # =============================================
    # OPIOID RECEPTORS
    # =============================================
    # Mu (MOR)
    ("4DKL", "Opioid", "Mu (MOR)", "Bound to beta-funaltrexamine, inactive", "Mus musculus", "BFN"),
    ("5C1M", "Opioid", "Mu (MOR)", "Active state with BU72 agonist, Gi", "Mus musculus", "BU72"),
    ("6DDE", "Opioid", "Mu (MOR)", "Bound to DAMGO agonist, Gi protein", "Mus musculus", "DAMGO"),
    ("6DDF", "Opioid", "Mu (MOR)", "Active state agonist", "Mus musculus", ""),
    ("7T2G", "Opioid", "Mu (MOR)", "Bound to fentanyl, Gi", "Homo sapiens", "Fentanyl"),
    ("8EF5", "Opioid", "Mu (MOR)", "Bound to morphine, Gi", "Homo sapiens", "Morphine"),

    # Delta (DOR)
    ("4N6H", "Opioid", "Delta (DOR)", "Bound to naltrindole antagonist", "Homo sapiens", "Naltrindole"),
    ("4RWD", "Opioid", "Delta (DOR)", "Bound to DIPP-NH2 peptide agonist", "Homo sapiens", "DIPP-NH2"),
    ("6PT2", "Opioid", "Delta (DOR)", "Active state with DPI-287, Gi", "Homo sapiens", "DPI-287"),

    # Kappa (KOR)
    ("4DJH", "Opioid", "Kappa (KOR)", "Bound to JDTic antagonist", "Homo sapiens", "JDTic"),
    ("6B73", "Opioid", "Kappa (KOR)", "Active state with MP1104 agonist, Gi", "Homo sapiens", "MP1104"),
    ("6VI4", "Opioid", "Kappa (KOR)", "Bound to salvinorin A (from S. divinorum)", "Homo sapiens", "Salvinorin A"),

    # Nociceptin (NOP / ORL1)
    ("5DHH", "Opioid", "NOP/ORL1", "Bound to C-24 antagonist", "Homo sapiens", "C-24"),
    ("5DHG", "Opioid", "NOP/ORL1", "Bound to SB-612111", "Homo sapiens", "SB-612111"),

    # =============================================
    # GABA RECEPTORS
    # =============================================
    # GABA-A (ionotropic - benzodiazepine target)
    ("6HUG", "GABA", "GABA-A (alpha1/beta2/gamma2)", "Cryo-EM, bound to GABA + flumazenil", "Homo sapiens", "GABA + Flumazenil"),
    ("6X3T", "GABA", "GABA-A (alpha1/beta2/gamma2)", "Bound to diazepam (Valium)", "Homo sapiens", "Diazepam"),
    ("6X3X", "GABA", "GABA-A (alpha1/beta3/gamma2)", "Bound to alprazolam (Xanax)", "Homo sapiens", "Alprazolam"),
    ("6X3S", "GABA", "GABA-A (alpha1/beta2/gamma2)", "Bound to zolpidem (Ambien)", "Homo sapiens", "Zolpidem"),
    ("7QNB", "GABA", "GABA-A (alpha5/beta3/gamma2)", "Bound to bretazenil", "Homo sapiens", "Bretazenil"),
    ("6D6U", "GABA", "GABA-A (alpha1/beta3)", "Bound to phenobarbital", "Homo sapiens", "Phenobarbital"),
    ("6HUP", "GABA", "GABA-A (alpha1/beta2/gamma2)", "Bound to Ro15-4513 inverse agonist", "Homo sapiens", "Ro15-4513"),

    # GABA-B (metabotropic)
    ("7EB2", "GABA", "GABA-B (B1/B2)", "Active state heterodimer with Gi", "Homo sapiens", "Baclofen"),

    # =============================================
    # GLUTAMATE RECEPTORS
    # =============================================
    # NMDA (ketamine/PCP target)
    ("4PE5", "Glutamate", "NMDA (GluN1/GluN2B)", "Bound to ifenprodil, allosteric", "Rattus norvegicus", "Ifenprodil"),
    ("4TLM", "Glutamate", "NMDA (GluN1/GluN2B)", "Full receptor cryo-EM", "Xenopus/Rattus", ""),
    ("7EU7", "Glutamate", "NMDA (GluN1/GluN2A)", "Bound to ketamine", "Rattus norvegicus", "Ketamine"),
    ("7SAD", "Glutamate", "NMDA (GluN1/GluN2A)", "Bound to PCP (phencyclidine)", "Rattus norvegicus", "PCP"),

    # AMPA
    ("5WEO", "Glutamate", "AMPA (GluA2)", "Tetrameric receptor, desensitized", "Rattus norvegicus", ""),
    ("6QKC", "Glutamate", "AMPA (GluA1/2)", "Heteromeric receptor cryo-EM", "Rattus norvegicus", ""),

    # Kainate
    ("5KUF", "Glutamate", "Kainate (GluK2)", "Full receptor structure", "Rattus norvegicus", ""),

    # mGluR5 (psychedelic modulator)
    ("6FFH", "Glutamate", "mGluR5", "Bound to mavoglurant (NAM)", "Homo sapiens", "Mavoglurant"),
    ("7FD8", "Glutamate", "mGluR5", "Active state TM domain", "Homo sapiens", ""),

    # =============================================
    # NICOTINIC ACETYLCHOLINE RECEPTORS
    # =============================================
    ("5KXI", "Acetylcholine", "nAChR (alpha4/beta2)", "Bound to nicotine", "Homo sapiens", "Nicotine"),
    ("6CNJ", "Acetylcholine", "nAChR (alpha4/beta2)", "Bound to NS9283 positive allosteric modulator", "Homo sapiens", "NS9283"),
    ("7KOO", "Acetylcholine", "nAChR (alpha7)", "Bound to epibatidine agonist", "Homo sapiens", "Epibatidine"),
    ("7EKI", "Acetylcholine", "nAChR (alpha3/beta4)", "Bound to nicotine", "Homo sapiens", "Nicotine"),

    # Muscarinic
    ("6OIJ", "Acetylcholine", "M1 mAChR", "Active state with Gi protein", "Homo sapiens", "Iperoxo"),
    ("3UON", "Acetylcholine", "M2 mAChR", "Bound to QNB antagonist", "Homo sapiens", "QNB"),
    ("4DAJ", "Acetylcholine", "M3 mAChR", "Bound to tiotropium", "Rattus norvegicus", "Tiotropium"),

    # =============================================
    # SIGMA RECEPTORS
    # =============================================
    ("5HK1", "Sigma", "Sigma-1", "Bound to PD144418 antagonist", "Homo sapiens", "PD144418"),
    ("5HK2", "Sigma", "Sigma-1", "Bound to 4-IBP agonist", "Homo sapiens", "4-IBP"),
    ("6DK1", "Sigma", "Sigma-1", "Bound to haloperidol", "Homo sapiens", "Haloperidol"),
    ("7W2O", "Sigma", "Sigma-1", "Bound to DMT", "Homo sapiens", "DMT"),

    # =============================================
    # ADRENERGIC RECEPTORS
    # =============================================
    ("3SN6", "Adrenergic", "Beta-2", "Active state with Gs, BI-167107 agonist (Nobel Prize structure)", "Homo sapiens", "BI-167107"),
    ("6KUX", "Adrenergic", "Alpha-2A", "Bound to brimonidine, Gi complex", "Homo sapiens", "Brimonidine"),
    ("7B6W", "Adrenergic", "Alpha-2A", "Active state with dexmedetomidine", "Homo sapiens", "Dexmedetomidine"),
    ("7YMI", "Adrenergic", "Alpha-1A", "Bound to oxymetazoline", "Homo sapiens", "Oxymetazoline"),

    # =============================================
    # HISTAMINE RECEPTORS
    # =============================================
    ("7UL3", "Histamine", "H1", "Bound to doxepin (antihistamine/antidepressant)", "Homo sapiens", "Doxepin"),
    ("7YFC", "Histamine", "H2", "Active state with Gs", "Homo sapiens", "Histamine"),
    ("7F61", "Histamine", "H3", "Bound to PF-03654746 antagonist", "Homo sapiens", "PF-03654746"),

    # =============================================
    # TRACE AMINE-ASSOCIATED RECEPTORS (TAAR)
    # =============================================
    ("8JLN", "Trace Amine", "TAAR1", "Bound to ulotaront, Gs complex", "Homo sapiens", "Ulotaront"),

    # =============================================
    # MELATONIN RECEPTORS
    # =============================================
    ("6ME2", "Melatonin", "MT1", "Bound to 2-phenylmelatonin agonist", "Homo sapiens", "2-PMT"),
    ("6ME3", "Melatonin", "MT1", "Bound to 2-iodomelatonin", "Homo sapiens", "2-Iodomelatonin"),
    ("6ME6", "Melatonin", "MT2", "Bound to ramelteon", "Homo sapiens", "Ramelteon"),

    # =============================================
    # SEROTONIN TRANSPORTER (SERT) - SSRI target
    # =============================================
    ("5I6X", "Transporter", "SERT", "Bound to paroxetine (Paxil)", "Homo sapiens", "Paroxetine"),
    ("5I71", "Transporter", "SERT", "Bound to citalopram (Celexa)", "Homo sapiens", "Citalopram"),
    ("6DZZ", "Transporter", "SERT", "Bound to S-citalopram (Lexapro)", "Homo sapiens", "Escitalopram"),
    ("7LIA", "Transporter", "SERT", "Bound to MDMA (ecstasy)", "Homo sapiens", "MDMA"),
    ("7LI9", "Transporter", "SERT", "Bound to serotonin + ibogaine", "Homo sapiens", "Ibogaine"),

    # =============================================
    # DOPAMINE TRANSPORTER (DAT) - stimulant target
    # =============================================
    ("4XP1", "Transporter", "DAT", "Bound to dopamine, drosophila", "Drosophila melanogaster", "Dopamine"),
    ("4XP4", "Transporter", "DAT", "Bound to D-amphetamine", "Drosophila melanogaster", "Amphetamine"),
    ("4XP6", "Transporter", "DAT", "Bound to cocaine analog", "Drosophila melanogaster", "Cocaine analog"),

    # =============================================
    # NOREPINEPHRINE TRANSPORTER (NET)
    # =============================================
    ("7ZWK", "Transporter", "NET", "Bound to venlafaxine (Effexor)", "Homo sapiens", "Venlafaxine"),

    # =============================================
    # VESICULAR MONOAMINE TRANSPORTER
    # =============================================
    ("8JQG", "Transporter", "VMAT2", "Bound to tetrabenazine", "Homo sapiens", "Tetrabenazine"),
]


def download_pdb(pdb_id, output_path):
    """Download PDB file from RCSB."""
    url = f"{RCSB_DOWNLOAD}/{pdb_id}.pdb"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.text) > 100:
            with open(output_path, "w") as f:
                f.write(resp.text)
            return True
    except Exception as e:
        print(f"  PDB download error: {e}")

    # Try mmCIF format as fallback (many cryo-EM structures only have cif)
    url = f"{RCSB_DOWNLOAD}/{pdb_id}.cif"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.text) > 100:
            cif_path = output_path.replace(".pdb", ".cif")
            with open(cif_path, "w") as f:
                f.write(resp.text)
            return True
    except Exception as e:
        print(f"  CIF download error: {e}")
    return False


def get_pdb_metadata(pdb_id):
    """Get entry metadata from RCSB API."""
    url = f"{RCSB_API}/{pdb_id}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            resolution = None
            method = None
            # Extract resolution
            diffraction = data.get("rcsb_entry_info", {})
            resolution = diffraction.get("resolution_combined", [None])[0]
            method = diffraction.get("experimental_method")

            title = data.get("struct", {}).get("title", "")
            deposition_date = data.get("rcsb_accession_info", {}).get("deposit_date", "")

            return {
                "resolution_A": resolution,
                "method": method,
                "title": title,
                "deposition_date": deposition_date,
            }
    except Exception as e:
        print(f"  Metadata error for {pdb_id}: {e}")
    return {}


def main():
    total = len(RECEPTORS)
    print(f"Downloading {total} receptor structures from RCSB PDB...\n")

    results = []
    failed = []

    for i, (pdb_id, receptor_class, subtype, description, organism, ligand) in enumerate(RECEPTORS):
        print(f"[{i+1}/{total}] {pdb_id} - {receptor_class} {subtype}: {description}")

        # Download structure file
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
        ok = download_pdb(pdb_id, pdb_path)

        if not ok:
            print(f"  FAILED to download {pdb_id}")
            failed.append(pdb_id)
            time.sleep(0.3)
            continue

        # Get metadata
        meta = get_pdb_metadata(pdb_id)
        res = meta.get("resolution_A", "N/A")
        method = meta.get("method", "N/A")
        print(f"  OK - Resolution: {res} A, Method: {method}")

        record = {
            "pdb_id": pdb_id,
            "receptor_class": receptor_class,
            "subtype": subtype,
            "description": description,
            "organism": organism,
            "ligand": ligand,
            "resolution_A": meta.get("resolution_A"),
            "method": meta.get("method"),
            "title": meta.get("title"),
            "deposition_date": meta.get("deposition_date"),
        }

        # Check which format was saved
        if os.path.exists(pdb_path):
            record["file"] = f"{pdb_id}.pdb"
        else:
            cif_path = pdb_path.replace(".pdb", ".cif")
            if os.path.exists(cif_path):
                record["file"] = f"{pdb_id}.cif"
            else:
                record["file"] = None

        results.append(record)
        time.sleep(0.25)

    # Save index JSON
    json_path = os.path.join(OUTPUT_DIR, "receptor_index.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save index CSV
    csv_path = os.path.join(OUTPUT_DIR, "receptor_index.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    # Summary by receptor class
    from collections import Counter
    class_counts = Counter(r["receptor_class"] for r in results)

    print(f"\n{'='*60}")
    print(f"RECEPTOR DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total structures: {len(results)} / {total}")
    if failed:
        print(f"Failed downloads: {failed}")
    print(f"\nBy receptor class:")
    for cls, n in class_counts.most_common():
        print(f"  {n:3d}  {cls}")
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"  receptor_index.json  - Full metadata")
    print(f"  receptor_index.csv   - Spreadsheet format")
    print(f"  pdb_files/           - PDB/CIF structure files")


if __name__ == "__main__":
    main()
