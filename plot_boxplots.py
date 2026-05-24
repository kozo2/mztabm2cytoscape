"""Per-metabolite abundance boxplots from a pickled mzTab-M model.

For each unique non-null InChI in small_molecule_summary, draw a boxplot of
abundance_assay values grouped by metadata.study_variable. Titles show the
SmallMoleculeSummary.chemical_name above the InChIKey derived from the InChI.
Output files are written to output/boxplots/<inchikey>.png.
"""

import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).parent
PKL = ROOT / "data" / "MTBLS40.pkl"
OUT_DIR = ROOT / "output" / "boxplots"


def inchi_to_key(inchi: str) -> str | None:
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    key = Chem.MolToInchiKey(mol)
    return key or None


def main() -> None:
    with PKL.open("rb") as fh:
        mztab = pickle.load(fh)

    sv_groups = [
        (sv.name or f"SV{sv.id}", list(sv.assay_refs))
        for sv in mztab.metadata.study_variable
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    written = 0
    skipped_invalid = 0

    for sm in mztab.small_molecule_summary:
        if not sm.inchi or not sm.abundance_assay:
            continue
        for inchi in sm.inchi:
            if not inchi:
                continue
            key = inchi_to_key(inchi)
            if not key:
                skipped_invalid += 1
                continue
            if key in seen:
                continue
            seen.add(key)

            abundances = sm.abundance_assay
            data = []
            labels = []
            for sv_name, refs in sv_groups:
                values = [
                    abundances[r - 1]
                    for r in refs
                    if 0 < r <= len(abundances) and abundances[r - 1] is not None
                ]
                if values:
                    data.append(values)
                    labels.append(sv_name)

            if not data:
                continue

            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.boxplot(data, tick_labels=labels)
            chemical_name = sm.chemical_name[0] if sm.chemical_name else None
            title = f"{chemical_name}\n{key}" if chemical_name else key
            ax.set_title(title)
            ax.set_xlabel("study_variable")
            ax.set_ylabel("abundance_assay")
            fig.tight_layout()

            first_layer = key.split("-", 1)[0]
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", first_layer)
            fig.savefig(OUT_DIR / f"{safe}.png", dpi=120)
            plt.close(fig)
            written += 1

    print(f"Wrote {written} boxplots to {OUT_DIR}")
    if skipped_invalid:
        print(f"Skipped {skipped_invalid} InChIs that RDKit could not parse")


if __name__ == "__main__":
    main()
