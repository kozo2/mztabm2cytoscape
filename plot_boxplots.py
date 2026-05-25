"""Per-metabolite abundance boxplots from a pickled mzTab-M model.

For each unique non-null InChI in small_molecule_summary, draw a boxplot of
abundance_assay values grouped by metadata.study_variable. Boxes are colored
by study_variable and individual points are overlaid as a jittered scatter.
Titles show the SmallMoleculeSummary.chemical_name (no InChIKey). Output
files are named by the first layer of the InChIKey and written to
output/boxplots/<first_layer>.png.
"""

import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
            box_spacing = 0.55
            box_width = 0.4
            positions = [1 + i * box_spacing for i in range(len(data))]
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i % cmap.N) for i in range(len(data))]

            bp = ax.boxplot(
                data,
                tick_labels=labels,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
            )
            ax.set_xlim(positions[0] - box_spacing / 2, positions[-1] + box_spacing / 2)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
                patch.set_edgecolor("black")
            for median in bp["medians"]:
                median.set_color("black")

            rng = np.random.default_rng(0)
            for pos, values, color in zip(positions, data, colors):
                jitter = rng.uniform(-box_width * 0.3, box_width * 0.3, size=len(values))
                ax.scatter(
                    np.full(len(values), pos) + jitter,
                    values,
                    color=color,
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.85,
                    s=24,
                    zorder=3,
                )

            chemical_name = sm.chemical_name[0] if sm.chemical_name else None
            default_title_fs = plt.rcParams["axes.titlesize"]
            if isinstance(default_title_fs, str):
                default_title_fs = plt.rcParams["font.size"]
            title_fontsize = default_title_fs * 3
            ax.set_title(chemical_name or "", fontsize=title_fontsize)
            ax.set_xlabel("study_variable")
            ax.set_ylabel("abundance_assay")
            tick_fontsize = plt.rcParams["font.size"] * 2
            ax.tick_params(axis="both", labelsize=tick_fontsize)
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
