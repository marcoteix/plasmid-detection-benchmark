
from utils.cli import DetectionARGsParser
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

parser = DetectionARGsParser()
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
    X.Taxon.eq(args.taxon)
]

results = {"With ARGs": {}, "Without ARGs": {}}
    
tools = predictions.columns.to_list()

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
    
    results["With ARGs"][n] = pd.concat(
        {
            k: metrics.group_metrics(
                X_bt[X_bt["SR contig has ARGs"]].copy(), 
                k, 
                None, 
                gt_col = "Ground-truth class"
            ) 
            for k in tools
        }, 
        names = ["tool"]
    )

    results["Without ARGs"][n] = pd.concat(
        {
            k: metrics.group_metrics(
                X_bt[~X_bt["SR contig has ARGs"]].copy(), 
                k, 
                None, 
                gt_col = "Ground-truth class"
            ) 
            for k in tools
        }, 
        names = ["tool"]
    )
    
for k in results: 
    results[k] = pd.concat(results[k], names=["iteration"])

results = pd.concat(results, names=["arg"], axis=1)

# Write results and a summary
results.to_csv(
    outdir.joinpath("detection.recall.args.tsv"), 
    sep="\t"
)

results.reset_index() \
    .drop(columns = "iteration") \
    .groupby(["tool", "Metric"]).describe() \
    .to_csv(
        outdir.joinpath("detection.args.summary.tsv"), 
        sep = "\t"
    )

summary = results.reset_index() \
    .melt(
        id_vars=["iteration", "tool", "Metric"]
    ).groupby(
        ["tool", "arg", "Metric"]
    )[["value"]].apply(
        lambda x: pd.Series(
            np.percentile(x, [2.5, 97.5]), 
            index=["left", "right"]
        )
    ).join(
        results.reset_index() \
            .melt(
                id_vars=["iteration", "tool", "Metric"]
            ).groupby(
                ["tool", "arg", "Metric"]
            ).value.median()
    )
summary.sort_index(
        level=2
    ).assign(
        fmt = summary.apply(
            lambda x: f"{x.value:.3f} ({x.left:.3f}, {x.right:.3f})",
            axis=1
        )
    ).reset_index() \
    .pivot(
        index="tool", 
        columns=["Metric", "arg"], 
        values="fmt"
    ).to_csv(
        outdir.joinpath("detection.args.ci.tsv"), 
        sep="\t"
    )

# Get p-values

p_values = {}
for tool in TOOL_ORDER:
    
    p_values[tool] = {}

    for metric in np.unique(results.index.get_level_values(2)):

        results_subset = results.xs(
            (tool, metric),
            axis = 0,
            level = [1,2]
        )

        p_values[tool][metric] = metrics.difference_pvalues(
            results_subset["Without ARGs"],
            results_subset["With ARGs"]
        )

    p_values[tool] = pd.Series(
        p_values[tool],
        name = "value"
    )
p_values = pd.concat(
    p_values,
    names = ["tool", "metric"]
)

p_values = pd.Series(
    false_discovery_control(p_values, axis=None),
    index = p_values.index,
    name = p_values.name 
)

p_values.to_csv(
    outdir.joinpath("args.p_values.tsv"),
    sep="\t"
)

# Plot results

fig, axs = plots.get_figure(1, 3, figsize=(12,4))

groups = ["With ARGs", "Without ARGs"]

for m, metric in enumerate(
    [
        "F1 score",
        "Precision",
        "Recall"
    ]
):
    
    results_subset = results.xs(metric, axis=0, level=2) \
        .reset_index() \
        .melt(
            id_vars=[
                "iteration", 
                "tool" 
            ], 
            value_vars=[
                "With ARGs", 
                "Without ARGs"
            ]
        )
    
    cmap = {
        k: v 
        for k, v in zip(
            groups,
            sns.color_palette("Set1", 2)
        )
    }
    
    for y, tool in enumerate(TOOL_ORDER[::-1]):

        for y_gap, group in enumerate(groups):

            plot_data = results_subset[ 
                results_subset.tool.eq(tool) & \
                results_subset.arg.eq(group)
            ]

            median = plot_data.value.median()
            ci = np.percentile(plot_data.value, [2.5, 97.5])
            is_significant = p_values.loc[(tool, metric)] < .05

            # Plot median as points. Closed if the difference between groups is 
            # significant and open otherwise
            axs[m].scatter(
                [median], [y + y_gap*.1 - .05],
                edgecolor = cmap[group],
                facecolor = (cmap[group] if is_significant else "1"),
                zorder = y_gap + 2,
                label = group if y == 0 else None
            )

            # Add 95% CI
            axs[m].plot(
                ci, [y + y_gap*.1 - .05]*2,
                color = cmap[group],
                zorder = y_gap + 1
            )
    
    plots.config_axes(
        axs[m], 
        move_legend=False, 
        xlabel = metric, 
        title = args.taxon if m==1 else None,
        xpercent = True,
        xlim = (0, 1.05)
    )

    axs[m].legend(frameon = False)
    
    if m > 0:
        axs[m].get_legend().remove()
        axs[m].set_yticks(
            axs[0].get_yticks(),
            []
        )
    else:
        axs[m].set_yticks(
            np.arange(len(TOOL_ORDER)),
            TOOL_ORDER[::-1]
        )

plots.save(
    outdir.joinpath("detection.args"),
    ["eps", "png"]
)


