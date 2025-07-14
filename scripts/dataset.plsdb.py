
from utils.cli import DatasetPLSDBParser
from utils.report import create_report
import numpy as np
import pandas as pd
from pathlib import Path 
from matiss import plots, stats
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sps
plots.set_style()

parser = DatasetPLSDBParser()
args = parser.parse()

outdir = Path(args.output)
outdir.mkdir(
    exist_ok = True,
    parents = True
)

report = create_report(outdir, args, "dataset.plsdb.py")

# Load identity to closest PLSDB hit
X = pd.read_excel(
    "data/predictions.xlsx",
    sheet_name = "Plasmid Characteristics",
    index_col = [0, 1]
)

# Merge with hybrid contig IDs
ids = pd.read_excel(
    "data/predictions.xlsx",
    sheet_name = "Ground-truth",
    index_col = [0, 1]
)

X = ids.join(X)

# Remove chromosomes
X = X.dropna(subset="Mash identity to closest PLSDB plasmid")

# Explode multiple hybrid contigs per row
X = X.assign(
        hybrid_contig = X["Hybrid contig ID"].str.split(";"),
        identity = X["Mash identity to closest PLSDB plasmid"].astype(str).str.split(";")
    ).explode(["hybrid_contig", "identity"]) \
    .reset_index() \
    .drop_duplicates(["Sample ID", "hybrid_contig"])

X = X.assign(identity = X.identity.astype(float))

# Test if the distribution of identities is similar across taxa
test_result = stats.Test(
    X, 
    y = "identity", 
    x = "Taxon", 
    test = sps.ttest_ind, 
    pairwise = True
).result

p_values = pd.DataFrame(
    [["Enterobacterales", "Enterococcus", test_result.pvalue]], 
    columns = ["group_1", "group_2", "p"]
)

# Plot boxplots
fig, ax = plots.get_figure(1, 1, figsize=(5,4))
sns.boxplot(
    X,
    x = "Taxon",
    y = "identity",
    ax = ax, 
    linecolor = "0", 
    linewidth = 1,
    flierprops = {
        "marker": "o", 
        "markerfacecolor": ".8",
        "markeredgecolor": "0", 
        "alpha": .3
    },
    medianprops = {
        "linewidth": 2
    },
    color=".8",
    fill=True
)
plots.config_axes(
    ax,
    ylabel = "Identity to best hit in PLSDB",
    ypercent = True
)
plots.add_significance(
    ax, 
    p_table = p_values,
    pad = 0.05
)

p_value = np.round(
    p_values \
        .iloc[0] \
        .p,
    5
)
if p_value < 1e-5: 
    p_value_fmt = "$p < 10^{-5}$"
else:
    p_value_fmt = f"$p = {p_value}$"

if p_value <= 0.05:
    ax.text(
        .5, 
        ax.get_ylim()[1], 
        p_value_fmt, 
        ha="center", 
        fontsize=15
    )

filepath = str(outdir.joinpath("plsdb.identity"))
plots.save(filepath, ["eps", "png"])
print(f"Rendered {filepath}.", file = report)


report.close()

