
from utils.cli import DetectionRepClusterParser
from utils.plots import BEST_TOOLS, TOOL_CMAP
from utils import metrics
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matiss import plots
import seaborn as sns
from pathlib import Path
from scipy.stats import false_discovery_control
from tqdm import tqdm
plots.set_style()

parser = DetectionRepClusterParser()
args = parser.parse()

outdir = Path(args.output)
outdir.mkdir(exist_ok=True, parents=True)

# Load predictions and ground-truth
X = pd.read_excel(
    args.input,
    index_col = [0, 1],
    sheet_name = "Ground-truth"
).join(
    pd.read_excel(
        args.input,
        index_col = [0, 1],
        sheet_name = "Plasmid Detection"
    )
).join(
    pd.read_excel(
        args.input,
        index_col = [0, 1],
        sheet_name = "Plasmid Characteristics"
    )
)

# Subset target taxon, discard chromosomes and samples without a 
# complete hybrid assembly
X = X[ 
    X.Taxon.eq(args.taxon) & \
    X["Ground-truth class"].eq("plasmid") & \
    X["Has complete hybrid assembly?"] & \
    ~X["Replicon cluster"].isna()
]

# Explode X according to the hybrid contig (plasmid)
X = X.assign(
    hybrid_contig = X["Hybrid contig ID"].str.split(";"),
    rep_cluster = X["Replicon cluster"].astype(str).str.split(";"),
    inc_type = X["Incompatibility group"].astype(str).str.split(";"),
    mash = X["Mash identity to closest PLSDB plasmid"].astype(str).str.split(";")
).explode(["hybrid_contig", "rep_cluster", "inc_type", "mash"])

X = X.assign(
    inc_type = X.inc_type.replace(
        {
            "nan": "Unknown",
            "None": "Unknown"
        }
    )
)

col = (
    "inc_type"
    if args.taxon == "Enterobacterales"
    else "rep_cluster"
)

# Get a count of rep clusters/inc types
counts = X.reset_index() \
    .drop_duplicates(["Sample ID", "hybrid_contig"]) \
    [col].value_counts()
counts.to_csv(outdir.joinpath(f"{col}.counts.tsv"), sep="\t")

results = {}

# Iterate over rep types
for t in X[col].unique():

    # Subset according to the rep type
    X_rep = X[X[col].eq(t)]

    # If there are less than five plasmids, ignore
    if len(
        X_rep.reset_index() \
            .drop_duplicates(["Sample ID", "hybrid_contig"])
        ) < 5: continue

    results[t] = {}

    # Bootstrap
    for m, X_bt in tqdm(
        enumerate(
            metrics.get_bootstrap_samples(X_rep, niter=args.niter)), 
            total = args.niter,
            desc = "Bootstrapping..."
        ):
            results[t][m] = pd.Series(
                {
                    k: metrics.group_metrics(
                        X_bt.copy(), 
                        k, 
                        None, 
                        gt_col="Ground-truth class"
                    ).Recall 
                    for k in TOOL_CMAP
                    if k in X.columns.to_list()
                }
            )
        
    results[t] = pd.concat(
        results[t], 
        names=["iteration"]
    )

results = pd.concat(results, names=[col], axis=1)

# Write results
results.to_csv(
    outdir.joinpath(f"detection.recall.{col}.tsv"), 
    sep="\t"
)

# Also write a summary of results
results.reset_index().drop(columns="iteration") \
    .rename(columns={"level_1": "tool"}) \
    .groupby("tool").describe() \
    .to_csv(
        outdir.joinpath(f"detection.recall.{col}.summary.tsv"), 
        sep="\t"
    )

summary = results.reset_index() \
    .rename(columns={"level_1":"tool"}) \
    .melt(id_vars=["iteration", "tool"]) \
    .groupby(["tool", col])[["value"]] \
    .apply(lambda x: pd.Series(np.percentile(x, [2.5, 97.5]), 
        index=["left", "right"])) \
    .join(results.reset_index() \
    .rename(columns={"level_1":"tool"}) \
    .melt(id_vars=["iteration", "tool"]) \
    .groupby(["tool", col]) \
    .value.median()) \
    .round(3).astype(str)
summary.assign(
    fmt = summary.value.apply(
        lambda x: 
            f"{float(x):.3f}") + " (" + summary \
                .left.apply(
                    lambda x: f"{float(x):.3f}"
                ) + ", " + summary.right.apply(
                    lambda x: f"{float(x):.3f}"
                ) + ")"
    ).reset_index() \
    .pivot(
        index="tool", 
        columns=[col], 
        values="fmt"
    ).to_csv(
        outdir.joinpath(f"detection.recall.{col}.ci.tsv"), 
        sep="\t"
    )

# Compare recall between each pair of rep clusters/inc types using a bootstrapping test
p_values = {}
for tool in results.index.get_level_values(1):

    results_subset = results.xs(tool, axis=0, level=1) \
        .dropna(
            axis=1,
            how="all"
        )
    p_values[tool] = {}

    for g1 in results_subset.columns:
        p_values[tool][g1] = {}
        
        for g2 in results_subset.columns:
            p_values[tool][g1][g2] = metrics.difference_pvalues(
                results_subset[g1],
                results_subset[g2]
            )

        p_values[tool][g1] = pd.Series(
            p_values[tool][g1],
            name = "value"
        )

    p_values[tool] = pd.concat(p_values[tool])

p_values = pd.concat(
    p_values,
    names = ["tool", "group_1", "group_2"]
).rename("pvalue")

p_values = pd.Series(
    false_discovery_control(p_values, axis=None),
    index = p_values.index,
    name = p_values.name 
)

p_values.to_csv(
    outdir.joinpath(f"{col}.pairswise_p_values.tsv"),
    sep="\t"
)

results.index.names = ["iteration", "tool"]

fig, (ax, ax_plsdb) = plt.subplots(
    2, 1,
    figsize = (6, 5),
    height_ratios = [5, 2]
)

width = (
    5
    if args.taxon == "Enterobacterales"
    else 8
)

plot_data = results.reset_index() \
    .drop(
        columns = "iteration"
    ).dropna(
        axis=1, 
        how="all"
    ).melt(id_vars="tool")

best = plot_data[plot_data.tool.isin(BEST_TOOLS)]
other = plot_data[~plot_data.tool.isin(BEST_TOOLS)]

x_order = best.groupby(col) \
    .value \
    .median() \
    .sort_values(ascending=False) \
    .index \
    .to_list()

# Add MASH distance to closest PLSDB plasmid

X = X.assign(distance = 1 - X.mash.astype(float))

sns.pointplot(
    X.reset_index().drop_duplicates(["Sample ID", "hybrid_contig"]),
    x = col,
    y = "distance",
    estimator = "mean",
    order = x_order,
    color = ".3",
    lw = 0,
    markersize = 10,
    errorbar = ("ci", 95), 
    err_kws={
        'linewidth': 2, 
        "alpha": 1
    },
    ax = ax_plsdb
)

sns.pointplot(
    best, 
    x=col, 
    y="value", 
    hue="tool", 
    palette=TOOL_CMAP, 
    ax=ax, 
    dodge=.4, 
    alpha=.8, 
    estimator=lambda x: np.median(x), 
    errorbar=lambda x: np.percentile(x, [2.5, 97.5]), 
    capsize=.1, 
    err_kws={
        'linewidth': 2, 
        "alpha": .5
    }, 
    markersize=10,
    order = x_order, 
    lw=0,
    zorder=10
)

sns.pointplot(
    other, 
    x=col, 
    y="value", 
    hue="tool", 
    palette=TOOL_CMAP, 
    ax=ax, 
    dodge=.2,
    alpha=.3, 
    lw=0, 
    estimator=lambda x: np.median(x), 
    errorbar=None, 
    order=x_order, 
    markersize=5
)

plots.config_axes(
    ax, 
    move_legend=True, 
    ylabel="Recall", 
    ypercent=True,
    ylim=(0,1.05), 
    xrotation=45,
    xlabel = None
)

plots.config_axes(
    ax_plsdb,
    move_legend = False,
    ylabel = "Distance",
    xlabel = (
        "Incompatibility group"
        if args.taxon == "Enterobacterales"
        else "Replicon cluster"
    ),
    ypercent = True,
    ylim = (0, .3)
)

counts.index = [ 
    x if x != "-" else "None"
    for x in counts.index
]

ax_plsdb.set_xticks(
    ax.get_xticks(), 
    [
        f"{x.get_text().replace('_', ' ').replace(',', '/').replace('rep cluster ', '')} ({counts.loc[x.get_text()]})" 
        for x in ax.get_xticklabels()
    ],
    rotation = 90,
    fontsize = 18
)

ax_plsdb.set_yticks(
    np.arange(0, .31, .15),
    [f"{x:.0%}" for x in np.arange(0, .31, .15)],
    fontsize = 18
)

ax.set_yticks(
    ax.get_yticks(),
    ax.get_yticklabels(),
    fontsize = 18
)

ax.set_xticks(
    ax.get_xticks(),
    []
)

ax.set_ylim(0, 1.05)
   
plots.save(
    outdir.joinpath(f"detection.recall.{col}"),
    format = ["png", "eps"]
)



