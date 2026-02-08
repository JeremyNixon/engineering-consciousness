"""General-purpose Vina docking scorer for any molecule + receptor pair."""

import os
import tempfile
import logging

from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

logger = logging.getLogger(__name__)

# Residues to exclude when auto-detecting binding site from HETATM records
EXCLUDE_HETATM = {
    "HOH", "WAT", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU",
    "CO", "NI", "CD", "SO4", "PO4", "GOL", "EDO", "ACT", "DMS", "BME",
    "MPD", "PEG", "PGE", "IOD",
}


def _detect_binding_site(pdb_path):
    """Auto-detect binding site center from HETATM records in a PDB file.

    Finds all HETATM atoms whose residue name is not a common solvent/ion,
    then returns their centroid as the binding site center.

    Returns:
        list of [x, y, z] or None if no suitable HETATM found
    """
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM"):
                resname = line[17:20].strip()
                if resname not in EXCLUDE_HETATM:
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append((x, y, z))
                    except ValueError:
                        continue
    if not coords:
        return None
    xs, ys, zs = zip(*coords)
    return [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)]


class VinaDockingScorer:
    """Wraps AutoDock Vina docking for use as a scoring function.

    Converts receptor PDB to PDBQT once, then scores arbitrary SMILES strings
    by generating 3D conformers and docking them.

    The ``score`` method returns a normalized value in [0, 1] (higher = better
    binding), while ``dock`` returns the raw Vina affinity in kcal/mol.
    """

    def __init__(self, receptor_pdb, center=None, box_size=None,
                 exhaustiveness=8):
        self.receptor_pdb = os.path.abspath(receptor_pdb)
        self.exhaustiveness = exhaustiveness
        self.box_size = box_size or [20, 20, 20]

        # Auto-detect binding site if center not provided
        if center is not None:
            self.center = list(center)
        else:
            self.center = _detect_binding_site(self.receptor_pdb)
            if self.center is None:
                raise ValueError(
                    f"Could not auto-detect binding site from {receptor_pdb}. "
                    "No non-solvent HETATM records found. "
                    "Please provide --center manually."
                )
            logger.info("Auto-detected binding site center: %s", self.center)

        # Prepare receptor PDBQT once
        self.receptor_pdbqt = self._prepare_receptor()

    def _prepare_receptor(self):
        conv = openbabel.OBConversion()
        conv.SetInAndOutFormats("pdb", "pdbqt")
        conv.AddOption("r", openbabel.OBConversion.OUTOPTIONS)
        obmol = openbabel.OBMol()
        conv.ReadFile(obmol, self.receptor_pdb)
        rec_file = tempfile.NamedTemporaryFile(
            suffix=".pdbqt", delete=False, prefix="receptor_"
        )
        conv.WriteFile(obmol, rec_file.name)
        rec_file.close()
        logger.info("Prepared receptor PDBQT: %s", rec_file.name)
        return rec_file.name

    def _smiles_to_pdbqt(self, smiles):
        """Convert a SMILES string to a Meeko PDBQT string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            result = AllChem.EmbedMolecule(mol, params)
            if result == -1:
                return None
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        try:
            mol_setup = MoleculePreparation().prepare(mol)[0]
            pdbqt_str, _, _ = PDBQTWriterLegacy.write_string(mol_setup)
            return pdbqt_str
        except Exception:
            return None

    def dock(self, smiles):
        """Dock a SMILES string and return raw Vina affinity (kcal/mol).

        Returns 0.0 if the molecule cannot be processed.
        """
        try:
            pdbqt_str = self._smiles_to_pdbqt(smiles)
            if pdbqt_str is None:
                return 0.0
            v = Vina(sf_name="vina", cpu=0, verbosity=0)
            v.set_receptor(rigid_pdbqt_filename=self.receptor_pdbqt)
            v.set_ligand_from_string(pdbqt_str)
            v.compute_vina_maps(center=self.center, box_size=self.box_size)
            v.dock(exhaustiveness=self.exhaustiveness, n_poses=1)
            return float(v.energies()[0][0])
        except Exception as e:
            logger.warning("Docking failed for %s: %s", smiles, e)
            return 0.0

    def score(self, smiles):
        """Return a normalized score in [0, 1] (higher = better binding).

        Maps Vina affinity from [-12, 0] kcal/mol to [1, 0].
        """
        affinity = self.dock(smiles)
        if affinity >= 0.0:
            return 0.0
        if affinity <= -12.0:
            return 1.0
        return -affinity / 12.0

    def cleanup(self):
        """Remove temporary receptor PDBQT file."""
        if hasattr(self, "receptor_pdbqt") and os.path.exists(self.receptor_pdbqt):
            os.unlink(self.receptor_pdbqt)
