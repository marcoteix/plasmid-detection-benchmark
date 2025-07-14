
from utils.cli import DatasetArgsISParser
from utils.report import create_report
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

args.taxon = "Enterococcus"

# Create an output directory
outdir = Path(args.output)
outdir.mkdir(exist_ok=True, parents=True)

# Create report 
report = create_report(
    outdir,
    args,
    "dataset.args.fragmentation.py"
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

# Remove incomplete plasmids
X = X[X["Has complete hybrid assembly?"]]

# Explode rows with multiple hybrid contigs and keep one row per plasmid.
# Then join with the number of IS per plasmid
X = X.assign(
        hybrid_contig = X["Hybrid contig ID"].str.split(";"),
        **{
            "Hybrid contig length": 
                X["Hybrid contig length"].astype(str).str.split(";"),
            "Number of insertion sequences (IS) in hybrid contig": 
                X["Number of insertion sequences (IS) in hybrid contig"].astype(str).str.split(";")
        }
    ).explode(
        [
            "hybrid_contig", 
            "Hybrid contig length",
            "Number of insertion sequences (IS) in hybrid contig"
        ]
    ).reset_index() \
    .drop_duplicates(["Sample ID", "hybrid_contig"])
 
# Exclude chromosomes
X = X[X["Ground-truth class"].eq("plasmid")]

for col in [
    "Hybrid contig length",
    "Number of insertion sequences (IS) in hybrid contig"
]:
    X = X.assign(
        **{col: X[col].astype(int)}
    )

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


print(
    f"Rendered {filepath}.",
    sep = "\n",
    file = report
)

fig, axs = plots.get_figure(
    1, 2, 
    figsize=(8,5)
)

sns.boxplot(
    X, 
    x="Hybrid contig has ARGs", 
    y="Hybrid contig length", 
    color=".8",
    fill=True, 
    ax=axs[0], 
    linecolor="0", 
    linewidth=1,
    flierprops={
        "marker": "o", 
        "markerfacecolor": ".8",
        "markeredgecolor": "0", 
        "alpha": 0
    }
)

sns.stripplot(
    X, 
    x="Hybrid contig has ARGs", 
    y="Hybrid contig length", 
    color=".3",
    alpha = .25,
    ax=axs[0]
)

test_res = stats.Test(
    X, 
    x="Hybrid contig has ARGs", 
    y="Hybrid contig length",  
    test=sps.ttest_ind, 
    pairwise=True
).result

ps = pd.DataFrame(
    [["False", "True", test_res.pvalue]], 
    columns=["group_1", "group_2", "p"]
)

yticks = [
    int(x/1000) 
    for x in axs[0].get_yticks()
]

plots.config_axes(
    axs[0], 
    move_legend=False, 
    xlabel="", 
    ylabel="Plasmid length (kbp)"
)

axs[0].set_yticks(
    axs[0].get_yticks(), 
    yticks
)

plots.add_significance(
    axs[0], 
    p_table=ps
)

axs[0].set_ylim(
    (-1000, axs[0].set_ylim()[1])
)

xticks = ["Without\nARGs", "With\nARGs"]
axs[0].set_xticks(axs[0].get_xticks(), xticks)

if test_res.pvalue <= 0.05:

    p = test_res.pvalue
    if p < 1e-5: p = "$p < 10^{-5}$"
    else: p = f"$p={test_res.pvalue:.5f}$"

    axs[0].text(
        .5, 
        axs[0].get_ylim()[1], 
        p, 
        ha="center", 
        fontsize=10
    )

# Add the number of SR contigs per plasmid
plasmids = pd.read_excel(
    args.input,
    index_col=[0,1],
    sheet_name = "Ground-truth"
)
plasmids = plasmids[ 
    plasmids["Has complete hybrid assembly?"]
]

n_sr_contigs = plasmids.assign(
        hybrid_contig = plasmids["Hybrid contig ID"].str.split(";")
    ).explode("hybrid_contig") \
    .reset_index() \
    .groupby(["Sample ID", "hybrid_contig"]) \
    .size() \
    .rename("Number of SR contigs")

X = X.join(n_sr_contigs, on = ["Sample ID", "Hybrid contig ID"])

sns.boxplot(
    X, 
    x = "Hybrid contig has ARGs", 
    y = "Number of SR contigs", 
    color = ".8",
    fill = True, 
    ax = axs[1], 
    linecolor = "0", 
    linewidth = 1,
    flierprops = {
        "marker": "o", 
        "markerfacecolor": ".8",
        "markeredgecolor": "0", 
        "alpha": 0
    }
)

sns.stripplot(
    X, 
    x = "Hybrid contig has ARGs", 
    y = "Number of SR contigs", 
    color=".3",
    alpha = .25,
    ax=axs[1]        
)
    
test_res = stats.Test(
    X, 
    x = "Hybrid contig has ARGs", 
    y = "Number of SR contigs",  
    test=sps.ttest_ind, 
    pairwise=True
).result

ps = pd.DataFrame(
    [["False", "True", test_res.pvalue]], 
    columns=["group_1", "group_2", "p"]
)

plots.config_axes(
    axs[1], 
    move_legend=False, 
    xlabel="", 
    ylabel="Number of short-read contigs"
)

plots.add_significance(axs[1], p_table=ps)

axs[1].set_ylim((0, axs[1].set_ylim()[1]))

xticks = ["Without\nARGs", "With\nARGs"]
axs[1].set_xticks(
    axs[1].get_xticks(), 
    xticks
)

if test_res.pvalue <= 0.05:
    p = test_res.pvalue
    if p < 1e-5: p = "$p < 10^{-5}$"
    else: p = f"$p={test_res.pvalue:.5f}$"
    axs[1].text(
        .5, 
        axs[1].get_ylim()[1], 
        p, 
        ha="center",
        fontsize = 10
    )

fig.suptitle(args.taxon, fontsize = 20)
filepath = str(outdir.joinpath(f"dataset.arg.pla_length.{args.taxon}"))
plots.save(
    filepath,
    format = ["eps", "png"]
)

print(f"Rendered {filepath}.", file = report)

report.close()


