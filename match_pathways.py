"""Match mzTab-M small-molecule InChIs against WikiPathways InChIKey table.

For each pathway in data/wikipathways2inchikey.csv, compute the ratio of its
InChIKey first-layer values that also appear among the first-layer InChIKeys
derived from the small_molecule_summary InChIs of data/MTBLS40.pkl.
"""

import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

DATA = Path(__file__).parent / "data"
PKL = DATA / "MTBLS40.pkl"
WP_CSV = DATA / "wikipathways2inchikey.csv"


def inchi_to_first_layer(inchi: str) -> str | None:
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    key = Chem.MolToInchiKey(mol)
    if not key:
        return None
    return key.split("-", 1)[0]


def main() -> None:
    with PKL.open("rb") as fh:
        mztab = pickle.load(fh)

    sample_first_layers: set[str] = set()
    for sm in mztab.small_molecule_summary:
        if not sm.inchi:
            continue
        for inchi in sm.inchi:
            if not inchi:
                continue
            first = inchi_to_first_layer(inchi)
            if first:
                sample_first_layers.add(first)

    print(f"Distinct sample InChIKey first-layers: {len(sample_first_layers)}")

    wp = pd.read_csv(WP_CSV)
    wp["first_layer"] = wp["inchikey"].str.split("-").str[0]

    rows = []
    for pathway, group in wp.groupby("pathway"):
        layers = set(group["first_layer"].dropna())
        if not layers:
            continue
        matches = layers & sample_first_layers
        rows.append(
            {
                "pathway": pathway,
                "pathwayTitle": group["pathwayTitle"].iloc[0],
                "total_inchikeys": len(layers),
                "matched": len(matches),
                "ratio": len(matches) / len(layers),
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values(["ratio", "matched"], ascending=[False, False])
        .reset_index(drop=True)
    )

    out = DATA / "pathway_match_ratios.csv"
    result.to_csv(out, index=False)
    print(f"Wrote {out} ({len(result)} pathways)")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
