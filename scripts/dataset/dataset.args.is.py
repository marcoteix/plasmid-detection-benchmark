#%%
from scripts.utils.cli import DatasetArgsISParser
from scripts.utils.report import create_report
import numpy as np
import pandas as pd
from pathlib import Path
from matiss import plots, stats
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sps
plots.set_style()

parser = DatasetArgsISParser()
args = parser.parse()

# Create an output directory
outdir = Path(args.output)
outdir.mkdir(exist_ok=True, parents=True)

# Create report 
report = create_report(
    outdir,
    args,
    "dataset.args.is.py"
)

# Load the number of IS per plasmid
n_is = pd.read_excel(
    args.input,
    index_col=[0,1],
    sheet_name = "Plasmid Characteristics"
)

# Load sample info and join
X = pd.read_excel(
    args.input,
    index_col=[0,1],
    sheet_name = "Ground-truth"
)

X = X.join(n_is)

# Subset samples from the target taxon
X = X[X.Taxon.eq(args.taxon)]

# Explode rows with multiple hybrid contigs and keep one row per plasmid.
# Then join with the number of IS per plasmid
X = X.assign(
        hybrid_contig = X["Hybrid contig ID"].str.split(";")
    ).explode("hybrid_contig") \
    .reset_index() \
    .drop_duplicates(["Sample ID", "hybrid_contig"])
 
# Exclude chromosomes
X = X[X["Ground-truth class"].eq("plasmid")]

# Test if the number of IS is similar between ARG+ and ARG- plasmids
test_result = stats.Test(
    X, 
    y = "Number of insertion sequences (IS) in hybrid contig", 
    x = "Hybrid contig has ARGs", 
    test = sps.ttest_ind, 
    pairwise = True
).result

print(
    "T-test comparing the number of IS in plasmids with and without ARGs:",
    f"\tstatistic: {test_result.statistic}",
    f"\tp-value: {test_result.pvalue}",
    sep = "\n",
    file = report
)

# Plot a boxplot of the number of IS per ARG content
fig, ax = plots.get_figure(1, 1, figsize=(5,5))

sns.boxplot(
    X,
    x = "Hybrid contig has ARGs",
    y = "Number of insertion sequences (IS) in hybrid contig",
    ax = ax, 
    linecolor = "0", 
    linewidth = 1,
    flierprops = {
        "marker": "o", 
        "markerfacecolor": ".8",
        "markeredgecolor": "0", 
        "alpha": 0
    },
    color=".8",
    fill=True,
    order = [False, True]
)

sns.stripplot(
    X,
    x = "Hybrid contig has ARGs",
    y = "Number of insertion sequences (IS) in hybrid contig",
    alpha = .25,
    color = ".3",
    ax = ax        
)

plots.config_axes(
    ax,
    ylabel = "Number of transposases",
    title = args.taxon
)
ax.set_xticks(
    [0, 1],
    ["Without ARGs", "With ARGs"]
)

# Add p-value to the plot
p_values = pd.DataFrame(
    [
        [
            "Without ARGs", 
            "With ARGs", 
            test_result.pvalue
        ]
    ], 
    columns = [
        "group_1", 
        "group_2", 
        "p"
    ]
)

plots.add_significance(
    ax, 
    p_table = p_values
)

p_value = np.round(p_values.iloc[0].p, 5)

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
        fontsize=10
    )
ax.set_ylim(-.5, ax.get_ylim()[1]*1.05)

filepath = str(outdir.joinpath(f"dataset.arg.n_is.{args.taxon}"))
plots.save(
    filepath,
    ["eps", "png"]
)
plt.show()

print(
    f"Rendered {filepath}.",
    sep = "\n",
    file = report
)

report.close()
