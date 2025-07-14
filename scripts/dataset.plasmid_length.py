
from utils.cli import DatasetPlasmidLengthParser
from utils.report import create_report
from matiss import plots  
import pandas as pd 
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
plots.set_style()

parser = DatasetPlasmidLengthParser()
args = parser.parse()

outdir = Path(args.output)
outdir.mkdir(exist_ok=True, parents=True)

report = create_report(
    outdir,
    args,
    "dataset.plasmid_length.py"
)

fig, axs = plots.get_figure(
    1, 2, (9,5)
)

taxa = ["Enterobacterales", "Enterococcus"]

# Load predictions
X_all = pd.read_excel(
    args.input,
    index_col=[0,1],
    sheet_name = "Ground-truth"
)

X_info = pd.read_excel(
    args.input,
    index_col=[0,1],
    sheet_name = "Plasmid Characteristics"
)

X_all = X_all.join(X_info)

for ax, taxon in zip(
    axs.flatten(),
    taxa
):

    # Subset taxon
    X = X_all[X_all.Taxon.eq(taxon)]

    # Explode hybrid contigs
    X = X.assign(
        hybrid_contig = X["Hybrid contig ID"].str.split(";"),
        hybrid_length = X["Hybrid contig length"].astype(str).str.split(";")
    ).explode(
        [
            "hybrid_contig",
            "hybrid_length"
        ]
    )

    X = X[X["Ground-truth class"].eq("plasmid")]

    X = X.reset_index() \
        .drop_duplicates(["Sample ID", "hybrid_contig"])

    X = X.assign(
        hybrid_length_kbp = X.hybrid_length.astype(float)/1000
    )

    X = X[ 
        X.hybrid_length_kbp.lt(300)
    ]

    sns.histplot(
        X,
        x = "hybrid_length_kbp",
        color = ".3",
        stat = "density",
        binwidth = 10,
        ax = ax
    )

    sns.kdeplot(
        X,
        x = "hybrid_length_kbp",
        color = ".3",
        bw_adjust = .75,
        fill = False,
        cut = 0,
        ax = ax
    )

    ax.vlines(
        x = (30 if taxon == "Enterobacterales" else 28),
        ls = "--",
        color = ".3",
        ymin = 0,
        ymax = ax.get_ylim()[1]
    )

    plots.config_axes(
        ax,
        xlabel = "Plasmid length (kbp)",
        ylabel = "PDF",
        title = taxon,
        xlim = (0, 300)
    )

filepath = str(outdir.joinpath("dataset.plasmid_length"))
plots.save(filepath, ["eps", "png"])


print(f"Rendered {filepath}.", file = report)

report.close()

