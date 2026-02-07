"""
Download molecular structure data for psychoactive molecules from PubChem.

Retrieves SMILES, InChI, molecular formula, molecular weight, 2D/3D coordinate files,
and other properties for a comprehensive list of psychoactive compounds.
"""

import requests
import json
import csv
import time
import os
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "molecule_data")
SDF_DIR = os.path.join(OUTPUT_DIR, "sdf_files")
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Comprehensive list of psychoactive molecules organized by class
PSYCHOACTIVE_MOLECULES = {
    # === Classical Psychedelics (Tryptamines) ===
    "Tryptamine Psychedelics": [
        "Psilocybin", "Psilocin", "DMT", "5-MeO-DMT", "Bufotenin",
        "4-AcO-DMT", "4-HO-MET", "4-HO-MiPT", "4-HO-DiPT",
        "5-MeO-MiPT", "5-MeO-DiPT", "DET", "DiPT", "MiPT",
        "DPT", "Alpha-methyltryptamine", "5-MeO-AMT",
        "Ibogaine", "Noribogaine",
    ],
    # === Ergolines ===
    "Ergolines": [
        "LSD", "LSA", "1P-LSD", "ALD-52", "ETH-LAD", "AL-LAD",
        "PRO-LAD", "Ergine", "Ergometrine",
    ],
    # === Phenethylamines ===
    "Phenethylamine Psychedelics": [
        "Mescaline", "2C-B", "2C-I", "2C-E", "2C-T-7", "2C-T-2",
        "2C-C", "2C-D", "2C-P", "2C-T-4",
        "DOB", "DOI", "DOM", "DOC",
        "25I-NBOMe", "25C-NBOMe", "25B-NBOMe",
        "TMA-2", "Escaline", "Allylescaline",
    ],
    # === Amphetamines ===
    "Substituted Amphetamines": [
        "Amphetamine", "Methamphetamine", "MDMA", "MDA", "MDEA",
        "MDE", "MBDB", "BDB",
        "4-Fluoroamphetamine", "2-Fluoroamphetamine",
        "Fenfluramine", "Phentermine",
    ],
    # === Dissociatives ===
    "Dissociatives": [
        "Ketamine", "Phencyclidine", "Dextromethorphan",
        "Methoxetamine", "3-MeO-PCP", "3-HO-PCP",
        "Dizocilpine", "Memantine", "Nitrous oxide",
        "Tiletamine", "Deschloroketamine",
    ],
    # === Cannabinoids ===
    "Cannabinoids": [
        "Tetrahydrocannabinol", "Cannabidiol", "Cannabinol",
        "Cannabigerol", "Cannabichromene",
        "Tetrahydrocannabivarin", "Cannabidivarin",
        "Anandamide", "2-Arachidonoylglycerol",
        "JWH-018", "JWH-073", "HU-210", "CP 55940", "WIN 55212-2",
        "Nabilone", "Dronabinol",
    ],
    # === Opioids ===
    "Opioids": [
        "Morphine", "Codeine", "Heroin", "Oxycodone", "Hydrocodone",
        "Fentanyl", "Methadone", "Buprenorphine", "Tramadol",
        "Naloxone", "Naltrexone", "Thebaine", "Papaverine",
        "Meperidine", "Hydromorphone", "Oxymorphone",
        "Loperamide", "Tapentadol",
        "Salvinorin A",
        "Mitragynine", "7-Hydroxymitragynine",
    ],
    # === Benzodiazepines ===
    "Benzodiazepines": [
        "Diazepam", "Alprazolam", "Clonazepam", "Lorazepam",
        "Midazolam", "Triazolam", "Temazepam", "Oxazepam",
        "Chlordiazepoxide", "Flurazepam", "Nitrazepam",
        "Flunitrazepam", "Bromazepam", "Clobazam",
    ],
    # === Barbiturates ===
    "Barbiturates": [
        "Phenobarbital", "Pentobarbital", "Secobarbital",
        "Amobarbital", "Thiopental", "Methohexital",
        "Butalbital",
    ],
    # === Stimulants ===
    "Stimulants": [
        "Cocaine", "Caffeine", "Nicotine", "Methylphenidate",
        "Modafinil", "Armodafinil", "Adrafinil",
        "Cathinone", "Methcathinone", "Mephedrone", "Methylone",
        "MDPV", "Alpha-PVP",
        "Ephedrine", "Pseudoephedrine",
        "Phenylpiracetam", "Theophylline", "Theobromine",
    ],
    # === Entactogens / Empathogens ===
    "Entactogens": [
        "6-APB", "5-APB", "5-MAPB", "6-MAPB",
        "4-FA", "Mephedrone", "Methylone",
        "AMT",
    ],
    # === Deliriants / Anticholinergics ===
    "Deliriants": [
        "Scopolamine", "Atropine", "Diphenhydramine",
        "Dimenhydrinate", "Datura stramonium",
        "Muscimol", "Ibotenic acid",
    ],
    # === Antidepressants (SSRIs, SNRIs, TCAs, MAOIs) ===
    "Antidepressants": [
        "Fluoxetine", "Sertraline", "Paroxetine", "Citalopram",
        "Escitalopram", "Fluvoxamine",
        "Venlafaxine", "Duloxetine", "Desvenlafaxine",
        "Amitriptyline", "Nortriptyline", "Imipramine",
        "Desipramine", "Clomipramine", "Doxepin",
        "Phenelzine", "Tranylcypromine", "Isocarboxazid",
        "Moclobemide", "Selegiline",
        "Bupropion", "Mirtazapine", "Trazodone", "Nefazodone",
        "Vortioxetine", "Vilazodone", "Agomelatine",
        "Tianeptine", "Ketamine",
    ],
    # === Antipsychotics ===
    "Antipsychotics": [
        "Haloperidol", "Chlorpromazine", "Fluphenazine",
        "Risperidone", "Olanzapine", "Quetiapine",
        "Aripiprazole", "Clozapine", "Ziprasidone",
        "Paliperidone", "Lurasidone", "Cariprazine",
        "Brexpiprazole", "Pimozide",
    ],
    # === Anxiolytics / Sedatives (non-benzo) ===
    "Anxiolytics": [
        "Buspirone", "Hydroxyzine", "Gabapentin", "Pregabalin",
        "Zolpidem", "Zaleplon", "Eszopiclone",
        "Melatonin", "Ramelteon", "Suvorexant",
        "Ethanol", "GHB", "1,4-Butanediol", "GBL",
    ],
    # === Nootropics ===
    "Nootropics": [
        "Piracetam", "Aniracetam", "Oxiracetam", "Pramiracetam",
        "Noopept", "Sulbutiamine", "Alpha-GPC",
        "Citicoline", "L-Theanine", "Bacopa monnieri",
        "Lion's mane", "Phosphatidylserine",
    ],
    # === Psychedelic Research Chemicals ===
    "Research Chemicals": [
        "4-AcO-DMT", "4-HO-MET", "1P-LSD",
        "3-MMC", "4-CMC", "NEP", "Hexen",
        "Clonazolam", "Flualprazolam", "Flubromazolam",
        "O-Desmethyltramadol", "Etizolam",
        "Phenibut", "F-Phenibut",
    ],
    # === MAOIs (used with ayahuasca) ===
    "MAOIs": [
        "Harmine", "Harmaline", "Tetrahydroharmine",
        "Syrian rue", "Banisteriopsis caapi",
    ],
    # === Anaesthetics ===
    "Anaesthetics": [
        "Propofol", "Sevoflurane", "Isoflurane", "Desflurane",
        "Halothane", "Xenon", "Lidocaine", "Procaine",
    ],
    # === Miscellaneous ===
    "Miscellaneous": [
        "Kava", "Kratom", "Epibatidine", "Arecoline",
        "Yohimbine", "Apomorphine", "Bromocriptine",
        "Cabergoline", "Pergolide", "Pramipexole",
        "Ropinirole", "Levodopa", "Carbidopa",
    ],
}


def get_all_molecule_names():
    """Flatten the dictionary into a deduplicated list."""
    seen = set()
    molecules = []
    for category, names in PSYCHOACTIVE_MOLECULES.items():
        for name in names:
            if name.lower() not in seen:
                seen.add(name.lower())
                molecules.append((category, name))
    return molecules


def search_pubchem_cid(name):
    """Search PubChem for a compound by name, return CID."""
    url = f"{PUBCHEM_BASE}/compound/name/{requests.utils.quote(name)}/cids/JSON"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            if cids:
                return cids[0]
    except Exception as e:
        print(f"  Error searching for {name}: {e}")
    return None


def get_compound_properties(cid):
    """Get molecular properties from PubChem for a given CID."""
    properties = [
        "MolecularFormula", "MolecularWeight", "CanonicalSMILES",
        "IsomericSMILES", "InChI", "InChIKey",
        "IUPACName", "XLogP", "ExactMass", "MonoisotopicMass",
        "TPSA", "Complexity", "Charge",
        "HBondDonorCount", "HBondAcceptorCount",
        "RotatableBondCount", "HeavyAtomCount",
        "AtomStereoCount", "BondStereoCount",
        "CovalentUnitCount",
    ]
    prop_str = ",".join(properties)
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/{prop_str}/JSON"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0]
    except Exception as e:
        print(f"  Error getting properties for CID {cid}: {e}")
    return None


def download_sdf(cid, output_path, record_type="3d"):
    """Download SDF (structure-data file) for a compound."""
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/SDF?record_type={record_type}"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            with open(output_path, "w") as f:
                f.write(resp.text)
            return True
        elif record_type == "3d":
            # Fall back to 2D if 3D not available
            return download_sdf(cid, output_path, record_type="2d")
    except Exception as e:
        print(f"  Error downloading SDF for CID {cid}: {e}")
    return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SDF_DIR, exist_ok=True)

    molecules = get_all_molecule_names()
    total = len(molecules)
    print(f"Attempting to download data for {total} unique psychoactive molecules...\n")

    results = []
    failed = []

    for i, (category, name) in enumerate(molecules):
        print(f"[{i+1}/{total}] {name} ({category})")

        # Search for CID
        cid = search_pubchem_cid(name)
        if cid is None:
            print(f"  NOT FOUND on PubChem, skipping.")
            failed.append({"name": name, "category": category, "reason": "Not found"})
            time.sleep(0.3)
            continue

        print(f"  CID: {cid}")

        # Get properties
        props = get_compound_properties(cid)
        if props is None:
            print(f"  Could not retrieve properties.")
            failed.append({"name": name, "category": category, "cid": cid, "reason": "No properties"})
            time.sleep(0.3)
            continue

        # Download SDF file
        safe_name = name.replace("/", "_").replace(" ", "_")
        sdf_path = os.path.join(SDF_DIR, f"{safe_name}_{cid}.sdf")
        sdf_ok = download_sdf(cid, sdf_path)
        print(f"  SDF: {'OK' if sdf_ok else 'FAILED'}")

        # Build record
        record = {
            "common_name": name,
            "category": category,
            "CID": cid,
            "sdf_file": os.path.basename(sdf_path) if sdf_ok else None,
        }
        record.update(props)
        results.append(record)

        # Print key info
        smiles = props.get("CanonicalSMILES", "N/A")
        formula = props.get("MolecularFormula", "N/A")
        mw = props.get("MolecularWeight", "N/A")
        print(f"  Formula: {formula} | MW: {mw} | SMILES: {smiles[:60]}{'...' if len(str(smiles))>60 else ''}")

        # Rate limit: PubChem allows 5 requests/second
        time.sleep(0.4)

    # Save results as JSON
    json_path = os.path.join(OUTPUT_DIR, "psychoactive_molecules.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} molecule records to {json_path}")

    # Save results as CSV
    csv_path = os.path.join(OUTPUT_DIR, "psychoactive_molecules.csv")
    if results:
        fieldnames = list(results[0].keys())
        # Gather all possible keys
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)
        # Put common_name, category, CID first
        for key in ["CID", "category", "common_name"]:
            if key in fieldnames:
                fieldnames.remove(key)
                fieldnames.insert(0, key)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Saved CSV to {csv_path}")

    # Save SMILES file (useful for cheminformatics tools)
    smiles_path = os.path.join(OUTPUT_DIR, "psychoactive_molecules.smi")
    with open(smiles_path, "w") as f:
        for r in results:
            smi = r.get("CanonicalSMILES", "")
            if smi:
                f.write(f"{smi}\t{r['common_name']}\n")
    print(f"Saved SMILES file to {smiles_path}")

    # Save failed lookups
    if failed:
        failed_path = os.path.join(OUTPUT_DIR, "failed_lookups.json")
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"\n{len(failed)} molecules could not be found. See {failed_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total attempted:  {total}")
    print(f"Successfully downloaded: {len(results)}")
    print(f"Failed/not found: {len(failed)}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Files:")
    print(f"  psychoactive_molecules.json  - Full property data")
    print(f"  psychoactive_molecules.csv   - Spreadsheet format")
    print(f"  psychoactive_molecules.smi   - SMILES strings")
    print(f"  sdf_files/                   - 3D/2D structure files")


if __name__ == "__main__":
    main()
