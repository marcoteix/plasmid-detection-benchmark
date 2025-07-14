
from utils.cli import DetectionMetricsParser
from utils.report import create_report
from utils import metrics
import numpy as np
import pandas as pd
from matiss import plots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
plots.set_style()

parser = DetectionMetricsParser()
args = parser.parse()

outdir = Path(args.output)
outdir.mkdir(exist_ok=True, parents=True)

report = create_report(outdir, args, "detection.metrics.py")

# Load predictions
X = pd.read_excel(
    args.input,
    sheet_name = "Plasmid Detection",
    index_col = [0, 1]
)

# List of tools
tools = X.columns.to_list()

# Load ground-truth and join with predictions
X = X.join(
        pd.read_excel(
        args.input,
        sheet_name = "Ground-truth",
        index_col = [0, 1]
    )
)

# Number of plasmids for which all contigs were correctly classified
n_perfect = {}
# Percentage of bp correctly identified
pct_bp = {}
# Number of plasmids in the dataset
n_pla = {}
# Detection metrics
class_metrics = {}

for taxon in ["Enterobacterales", "Enterococcus"]:

    class_metrics[taxon] = {}
    n_perfect[taxon] = {}
    pct_bp[taxon] = {}

    X_taxon = X[X.Taxon.eq(taxon)]
    
    # Bootstrap; compute detection metrics
    for n, X_samp in tqdm(
        enumerate(
            metrics.get_bootstrap_samples(X_taxon, niter=args.niter)
        ), 
        total = args.niter,
        desc = f"Computing {taxon} plasmid detection metrics..."
    ):
        class_metrics[taxon][n] = pd.concat(
            {
                k: metrics.group_metrics(
                    X_samp.copy(), 
                    k, 
                    None, 
                    gt_col = "Ground-truth class"
                ) 
                for k in tools
            }
        )
    class_metrics[taxon] = pd.concat(class_metrics[taxon])

    # Check how many plasmids were perfectly detected 

    # Keep complete assemblies
    X_complete = X_taxon[ 
        X_taxon["Has complete hybrid assembly?"]
    ]

    # Explode X according to the hybrid contig (plasmid)
    X_complete = X_complete["Hybrid contig ID"] \
        .str.split(";") \
        .explode() \
        .to_frame() \
        .join(X_complete.drop(columns="Hybrid contig ID"))

    for tool in tools:
        # Get the total length classified as plasmid/chromosomal and the total length
        # of the plasmid
        len_class = X_complete.reset_index() \
            .groupby(["Sample ID", "Hybrid contig ID", tool]) \
            ["SR contig length"].sum() \
            .unstack(fill_value=0).stack() \
            .rename("frag_length").to_frame() \
            .reset_index() \
            .join(X.groupby(["Sample ID", "Hybrid contig ID"]) \
            ["SR contig length"].sum() \
            .rename("total"), on=["Sample ID", "Hybrid contig ID"]) \
            .convert_dtypes(convert_floating=False)
        
        n_perfect[taxon][tool] = (
            len_class[tool].eq("plasmid") & \
            len_class.frag_length.eq(len_class.total)
        ).sum()
        
        pct_bp[taxon][tool] = len_class[
            len_class[tool].eq("plasmid")
        ].frag_length / len_class[
            len_class[tool].eq("plasmid")
        ].total

        n_pla[taxon] = len(
            len_class[["Sample ID", "Hybrid contig ID"]] \
                .drop_duplicates()
        )

    n_perfect[taxon] = pd.Series(n_perfect[taxon], name=taxon)
    pct_bp[taxon] = pd.concat(pct_bp[taxon], names=["tool"])

n_perfect = pd.concat(n_perfect, names=["dataset"])
pct_bp = pd.concat(pct_bp, names=["dataset"])
n_pla = pd.Series(n_pla, name="n_plasmids")

tool_order = class_metrics["Enterobacterales"] \
    .xs("F1 score", level=2) \
    .sort_values(ascending=False) \
    .index.to_list()

class_metrics = pd.concat(class_metrics) \
    .reset_index() \
    .rename(
        columns={
            "level_0": "dataset",
            "level_1": "iteration", 
            "level_2": "Tool", 
            0: "value"
        }
    )

filepath = str(outdir.joinpath(f"metrics.bootstrap.tsv"))
class_metrics.to_csv(filepath, sep="\t")
print(
    f"Saved plasmid detection metrics to {filepath}.",
    file = report
)

n_perfect = n_perfect.reset_index() \
    .rename(
        columns = {
            "level_1": "tool",
            0: "value"
        }
    )

filepath = str(outdir.joinpath(f"n_perfect.tsv"))
n_perfect.to_csv(filepath, sep="\t")
print(
    f"Saved the number of perfectly detected plasmids to {filepath}.",
    file = report
)

filepath = str(outdir.joinpath(f"pct_bp.tsv"))
pct_bp.to_csv(filepath, sep="\t")
print(
    f"Saved the percentage of detected plasmid bp to {filepath}.",
    file = report
)

print(
    "Number of plasmids per taxon:",
    n_pla,
    sep = "\n",
    file = report
)

# Compute the median and the 95% CI
median = class_metrics \
    .groupby(["dataset", "Tool", "Metric"]) \
    .value.median() \
    .rename("median")

ci = pd.concat(
    [
        class_metrics \
        .groupby(["dataset", "Tool", "Metric"]) \
        .value.apply(
            lambda x: np.percentile(x, l)
        ).rename(l) 
        for l in [2.5, 97.5]
    ], 
    axis=1
)
agg_metrics = pd.concat([median, ci], axis=1)

filepath = str(outdir.joinpath(f"metrics.agg.tsv"))
agg_metrics.to_csv(filepath, sep="\t")
print(
    f"Saved median and 95% CI of plasmid detection metrics to {filepath}.",
    file = report
)

# Check for significant differences between taxa
significant = pd.concat(
    {
        metric: metrics.compare_groups(
            class_metrics[class_metrics.Metric.eq(metric)], 
            values="value", 
            groups="dataset", 
            index="Tool"
        ) 
        for metric in class_metrics.Metric.unique()
    }
).rename("significant")

for k, v in n_pla.items():
    n_perfect.loc[
        n_perfect.dataset.eq(k), 
        "value"
    ] = n_perfect.loc[
        n_perfect.dataset.eq(k), 
        "value"
    ]/v 

metrics_ = [
    "F1 score", 
    "Precision", 
    "Recall"
]

# Order of the tools along the axes
tool_order = [
    'Plasmer',
    'PlaScope',
    'PlasmidEC',
    'gplas2',
    'RFPlasmid',
    'Platon',
    'PLASMe',
    'MOB-recon',
    'HyAsP',
    'plASgraph2',
    'geNomad',
    'PlasmidFinder'
]

fig, axs = plots.get_figure(1, 4, (14,4))
taxa = ["Enterobacterales", "Enterococcus"]
cmap = {
    k: v 
    for k, v in zip(
        taxa,
        sns.color_palette("Set1", 2)
    )
}

for n, (metric, ax) in enumerate(
    zip(metrics_, axs.flatten())
):
    
    for y, tool in enumerate(tool_order[::-1]):

        for y_gap, taxon in enumerate(taxa):

            plot_data = class_metrics[ 
                class_metrics.Tool.eq(tool) & \
                class_metrics.Metric.eq(metric) & \
                class_metrics.dataset.eq(taxon)
            ]

            median = plot_data.value.median()
            ci = np.percentile(plot_data.value, [2.5, 97.5])
            is_significant = significant.loc[(metric, tool)]

            # Plot median as points. Closed if the difference between groups is 
            # significant and open otherwise
            ax.scatter(
                [median], [y + y_gap*.1 - .05],
                edgecolor = cmap[taxon],
                facecolor = (cmap[taxon] if is_significant else "1"),
                zorder = y_gap + 2,
                label = taxon if y == 0 else None
            )

            # Add 95% CI
            ax.plot(
                ci, [y + y_gap*.1 - .05]*2,
                color = cmap[taxon],
                zorder = y_gap + 1
            )

    plots.config_axes(
        ax = ax,
        move_legend = False,
        xlabel = metric,
        ylabel = (
            "Tool"
            if n == 0
            else None
        ),
        xlim = (0, 1.05),
        xpercent = True
    )
        
    if n > 0:
        ax.legend().remove()
        ax.set_yticks(
            ax.get_yticks(),
            [""] * len(ax.get_yticks())
        )
    else:
        ax.legend(frameon = False)

for y, tool in enumerate(tool_order[::-1]):

    for y_gap, taxon in enumerate(taxa):

        value = n_perfect.loc[ 
            n_perfect.tool.eq(tool) & \
            n_perfect.dataset.eq(taxon),
            "value"
        ].iloc[0]

        # Plot median as points. Closed if the difference between groups is 
        # significant and open otherwise
        axs[3].scatter(
            [value], [y + y_gap*.1 - .05],
            color = cmap[taxon],
            zorder = y_gap + 2,
        )

plots.config_axes(
    ax = axs[3],
    xlabel = "Proportion of fully\ndetected plasmids",
    ylabel = None,
    xlim = (0, 1.05),
    xpercent = True
)

axs[3].legend().remove()
axs[3].set_yticks(
    ax.get_yticks(),
    [""] * len(ax.get_yticks())
)

axs[0].set_yticks(
    np.arange(len(tool_order)),
    tool_order[::-1]
)

for ax in axs:
    ax.set_ylim(*axs[0].get_ylim())

filepath = str(outdir.joinpath("detection"))
plots.save(
    filepath,
    format = ["png", "eps"]
)
print(f"Rendered {filepath}.", file = report)


report.close()
