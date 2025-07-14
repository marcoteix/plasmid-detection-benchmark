
from utils.cli import DetectionPlasmidSizeParser
from utils.plots import TOOL_ORDER
from utils import metrics
from scipy.stats import false_discovery_control
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from matiss import plots
import seaborn as sns
import matplotlib.pyplot as plt
plots.set_style()

parser = DetectionPlasmidSizeParser()
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
        sheet_name = "Plasmid Characteristics"
    )
)

predictions = pd.read_excel(
    args.input,
    index_col = [0, 1],
    sheet_name = "Plasmid Detection"
)

X = X.join(predictions)

# Subset target taxon
X = X[
    X.Taxon.eq(args.taxon) & \
    X["Has complete hybrid assembly?"]
]

results = {"small": {}, "large": {}}

# Explode X according to the hybrid contig (plasmid)
X = X.assign(
    hybrid_contig = X["Hybrid contig ID"].str.split(";"),
    hybrid_length = X["Hybrid contig length"].astype(str).str.split(";")
).explode(["hybrid_contig", "hybrid_length"])

# Classify plasmids as small or large
threshold = 28000 if args.taxon == "Enterococcus" else 30000

X = X.assign(
    pla_type = X.hybrid_length.astype(int).ge(threshold) \
        .replace({True: "large", False: "small"})
)

# Bootstrap
for n, X_bt in tqdm(
    enumerate(
        metrics.get_bootstrap_samples(
            X, 
            niter = args.niter
        )
    ), 
    total = args.niter,
    desc = "Bootstrapping..."
):
    
    for group in ["small", "large"]:
        results[group][n] = pd.Series(
            {
                k: metrics.group_metrics(
                    X_bt[X_bt.pla_type.eq(group)].copy(), 
                    k, 
                    None, 
                    gt_col="Ground-truth class"
                ).Recall 
                for k in TOOL_ORDER
            }
        )
    
for k in results: 
    results[k] = pd.concat(results[k], names=["iteration"])

results = pd.concat(results, names=["plasmid_type"], axis=1)

# Write results and summary
results.to_csv(
    outdir.joinpath("detection.recall.plasmid_size.tsv"), 
    sep="\t"
)

results.reset_index() \
    .drop(columns="iteration") \
    .rename(columns={"level_1": "tool"}) \
    .groupby("tool").describe() \
    .to_csv(
        outdir.joinpath("detection.recall.plasmid_size.summary.tsv"), 
        sep="\t"
    )

summary = results.reset_index() \
    .rename(columns={"level_1":"tool"}) \
    .melt(id_vars=["iteration", "tool"]) \
    .groupby(["tool", "plasmid_type"])[["value"]] \
    .apply(
        lambda x: pd.Series(
            np.percentile(x, [2.5, 97.5]), 
            index=["left", "right"]
        )
    ).join(
        results.reset_index() \
            .rename(columns={"level_1":"tool"}) \
            .melt(id_vars=["iteration", "tool"]) \
            .groupby(["tool", "plasmid_type"]) \
            .value.median()
    ).round(3).astype(str)

summary.assign(
    fmt = summary.value + " (" + summary.left + ", " + summary.right + ")"
).reset_index() \
.pivot(
    index="tool", 
    columns=["plasmid_type"], 
    values="fmt"
).to_csv(
    outdir.joinpath("detection.recall.plasmid_size.ci.tsv"), 
    sep="\t"
)

p_values = {}
for tool in TOOL_ORDER:

    results_subset = results.xs(tool, axis=0, level=1)

    p_values[tool] = metrics.difference_pvalues(
        results_subset.small,
        results_subset.large,
        paired = True
    )
p_values = pd.Series(
    p_values,
    name = "value"
)

# Apply FDR correction
p_values = pd.Series(
    false_discovery_control(p_values),
    index = p_values.index,
    name = "pvalues"
)
p_values.to_csv(
    outdir.joinpath("recall.plasmid_size.pvalues.tsv"), 
    sep="\t"
)

# Plot results 

fig, ax = plots.get_figure(1, 1, figsize=(4,4))

threshold_str = str(threshold/1000) + " kb"

results.index.names = ["iteration", "tool"]
    
melted_results = results.reset_index() \
    .melt(
        id_vars=["iteration", "tool"], 
        value_vars=["small", "large"]
    ).replace(
        {
            "small": f"Small ($\leq${threshold_str})",
            "large": f"Large ($>${threshold_str})"
        }
    )

groups = melted_results.plasmid_type.unique()

cmap = {
    k: v 
    for k, v in zip(
        groups,
        sns.color_palette("Set1", 2)
    )
}

for y, tool in enumerate(TOOL_ORDER[::-1]):

    for y_gap, group in enumerate(groups):

        plot_data = melted_results[ 
            melted_results.tool.eq(tool) & \
            melted_results.plasmid_type.eq(group)
        ]

        median = plot_data.value.median()
        ci = np.percentile(plot_data.value, [2.5, 97.5])
        is_significant = p_values.loc[tool] < .05

        # Plot median as points. Closed if the difference between groups is 
        # significant and open otherwise
        ax.scatter(
            [median], [y + y_gap*.1 - .05],
            edgecolor = cmap[group],
            facecolor = (cmap[group] if is_significant else "1"),
            zorder = y_gap + 2,
            label = group if y == 0 else None
        )

        # Add 95% CI
        ax.plot(
            ci, [y + y_gap*.1 - .05]*2,
            color = cmap[group],
            zorder = y_gap + 1
        )

ax.legend(frameon = False)

plots.config_axes(
    ax = ax,
    move_legend = False,
    xlabel = "Recall",
    ylabel = "Tool",
    legend_title = "Plasmid size",
    title = args.taxon,
    xpercent = True,
    xlim = (0, 1.05)
)

plots.save(
    outdir.joinpath("recall.plasmid_size"),
    ["png", "eps"]
)




