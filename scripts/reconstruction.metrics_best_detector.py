
from utils.cli import ReconstructionMetricsParser
from utils.report import create_report
from utils.plots import TOOL_ORDER_RECONSTRUCTION
from utils import metrics
from scipy.stats import false_discovery_control
from sklearn import metrics as skm
import numpy as np
import pandas as pd
from matiss import plots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
plots.set_style()

parser = ReconstructionMetricsParser()
args = parser.parse()

outdir = Path(args.output)
outdir.mkdir(exist_ok=True, parents=True)

report = create_report(outdir, args, "reconstruction.metrics_best_detector.py")

# Load predictions
X = pd.read_excel(
    args.input,
    sheet_name = "Plasmid Reconst. (Best Det.)",
    index_col = [0, 1]
).fillna("chromosome")

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

# Discard incomplete hybrid assemblies
X = X[X["Has complete hybrid assembly?"]]

# Plasmid reconstruction metrics
metrics_f = {
    "NMI": skm.normalized_mutual_info_score,
    "Homogeneity": skm.homogeneity_score, 
    "Completeness": skm.completeness_score
}
metric_names = list(metrics_f.keys())

results = {}
ci = {}
sd = {}
mean = {}

print(
    "Calculating plasmid reconstruction using predictions obtained with initial plasmid \
contig classifications from the best plasmid detection tools.",
    sep = "\n",
    file = report
)

taxa = ["Enterobacterales", "Enterococcus"]

for taxon in taxa:

    # Add a tag to each hybrid contig name and bin name with the sample name
    Xb = X.copy()
    Xb = Xb[Xb.Taxon.eq(taxon)]

    Xb = Xb.assign(
        tmp_sample = Xb.index.get_level_values("Sample ID")
    )

    for idx, row in Xb.iterrows():
        Xb.loc[idx, "Hybrid contig ID"] = ";".join(
            [x+"_"+row.tmp_sample for x in row["Hybrid contig ID"].split(";")]
        )
        for tool in tools:
            Xb.loc[idx, tool] = ";".join(
                [x+"_"+row.tmp_sample for x in str(row[tool]).split(";")]
            )
    Xb = Xb.drop(columns="tmp_sample")

    # Estimate metrics through bootstrapping
    results[taxon] = {}
    
    for n, Xbg in tqdm(
        enumerate(
            metrics.get_bootstrap_samples(Xb, niter=args.niter)
        ),
        total = args.niter,
        desc = "Bootstrapping..."
    ):
        
        results[taxon][n] = {}
        
        for metric, mfunc in metrics_f.items():

            # For SR contigs mapping to multiple hybrid contigs, expand
            # into cross product
            results[taxon][n][metric] = pd.Series(
                {
                    tool: mfunc(
                        *metrics.multilabel_to_single_label(
                            Xbg["Hybrid contig ID"], 
                            Xbg[tool]
                        )
                    ) 
                    for tool in tools
                }
            )

        results[taxon][n] = pd.concat(results[taxon][n])
    results[taxon] = pd.concat(results[taxon])

results = pd.concat(results) \
    .reset_index() \
    .rename(
        columns={
            "level_0": "dataset",
            "level_1": "iteration", 
            "level_2": "Metric", 
            "level_3": "Tool", 
            0: "value"
        }
    )

filepath = str(outdir.joinpath("reconstruction.metrics.tsv"))
results.to_csv(filepath, sep = "\t")
print(
    f"Saved plasmid reconstruction metrics (for each bootstrapping iteration) as {filepath}.",
    file = report
)

# Compute the median and the 95% CI
median = results.groupby(["dataset", "Tool", "Metric"]) \
    .value.median() \
    .rename("median")

ci = pd.concat(
    [
        results.groupby(["dataset", "Tool", "Metric"]) \
            .value.apply(
                lambda x: np.percentile(x, l)
            ).rename(l) 
        for l in [2.5, 97.5]
    ], 
    axis = 1
)

agg_metrics = pd.concat([median, ci], axis=1)

filepath = str(outdir.joinpath("reconstruction.metrics.agg.tsv"))
agg_metrics.to_csv(filepath, sep = "\t")
print(
    f"Saved aggregate plasmid reconstruction metrics as {filepath}.",
    file = report
)

# Check for significant differences between taxa
significant = pd.concat(
    {
        metric: metrics.compare_groups(
            results[
                results.Metric.eq(metric) & \
                ~results.Tool.eq("PlasBin-Flow")
            ], 
            values = "value", 
            groups = "dataset", 
            index = "Tool"
        ) 
        for metric in results.Metric.unique()
    }
).rename("significant")

significant.loc["PlasBin-Flow"] = pd.NA

# Test if the tools have similar performance for both taxa. Get p-values from
# bootstrapping samples

p_values = {}
    
for metric in results.Metric.unique():

    p_values[metric] = {}

    for tool in results.Tool.unique():
        
        results_subset = results[
            results.Tool.eq(tool) &
            results.Metric.eq(metric)
        ]

        p_values[metric][tool] = metrics.difference_pvalues(
            results_subset[results_subset.dataset.eq("Enterobacterales")] \
                .set_index("iteration") \
                .value,
            results_subset[results_subset.dataset.eq("Enterococcus")] \
                .set_index("iteration") \
                .value,
            paired = True
        )

    p_values[metric] = pd.Series(
        p_values[metric],
        name = "value"
    )

p_values = pd.concat(
    p_values,
    names = ["metric", "tool"]
).reset_index() \
    .pivot(
        index="tool", 
        columns="metric", 
        values="value"
    )

# Apply FDR correction
p_values = pd.Series(
    false_discovery_control(p_values.unstack()),
    index = p_values.unstack().index,
    name = "pvalues"
).to_frame() \
    .reset_index() \
    .pivot(
        index = "tool",
        columns = "metric",
        values = "pvalues"
    )

filepath = str(outdir.joinpath("reconstruction.taxa.pvalues.tsv"))
p_values.to_csv(
    filepath, 
    sep="\t"
)
print(
    f"Tested if the tools have similar performance for both taxa. Saved p-values from \
bootstrapping samples as {filepath}.",
    file = report
)

# Get the number of perfectly reconstructed plasmids

results_perfect = {}

for taxon in taxa:

    # Keep plasmids from target taxon
    X_subset = X[
        X["Ground-truth class"].eq("plasmid") & \
        X.Taxon.eq(taxon)
    ]

    perfect = []

    for sample in X_subset.index.get_level_values("Sample ID").unique():

        subset = X_subset.loc[[sample]]

        for tool in tools:

            subset[tool] = subset[tool].astype(str)

            for gt_bin in subset["Hybrid contig ID"] \
                .str.split(";") \
                .explode() \
                .unique():
                    
                    # Are all contigs from this plasmid in the same bin?
                    plasmid = subset[subset["Hybrid contig ID"].str.contains(gt_bin)]

                    unique_bin = any(
                        [
                            plasmid[tool].str.contains(x).all() 
                            for x in plasmid[tool] \
                                .str.split(";") \
                                .explode() \
                                .unique()
                        ]
                    )

                    if (
                        plasmid[tool].isna().any() or \
                        plasmid[tool].isin(["chr", "chromosome"]).any() \
                        or not unique_bin
                    ): 
                        continue

                    for pred in subset[tool] \
                        .str.split(";") \
                        .explode() \
                        .unique():
                        
                            # Does that bin have contigs from other plasmids?
                            if (
                                subset[
                                    subset[tool].str.contains(pred)
                                ]["Hybrid contig ID"].str \
                                .contains(gt_bin) \
                                .all()
                            ) and not pred in ["chr", "chromosome"]:
                                perfect.append([sample, gt_bin, tool, pred, len(plasmid)])
                                break

    perfect = pd.DataFrame(
        perfect, 
        columns = [
            "sample", 
            "Hybrid contig ID", 
            "tool", 
            "prediction", 
            "n_contigs"
        ]
    )

    n_pla_gt = X_subset["Hybrid contig ID"] \
        .str.split(";") \
        .explode() \
        .reset_index() \
        .groupby("Sample ID") \
        ["Hybrid contig ID"].nunique() \
        .sum()
    
    # Find the number of plasmids with >1 frag-only contig
    n_contig_per_pla = X_subset["Hybrid contig ID"] \
        .str.split(";") \
        .explode(). \
        reset_index() \
        .groupby(
            ["Sample ID", "Hybrid contig ID"], 
            as_index = False
        ).size()
    
    pla_g1 = n_contig_per_pla[
        n_contig_per_pla["size"].ge(2)
    ]

    n_pla_g1 = pla_g1.groupby("Sample ID")["Hybrid contig ID"].nunique().sum()

    n_perfect = perfect.groupby("tool") \
        ["Hybrid contig ID"].size() \
        .to_frame("n_pla")
    
    n_perfect = n_perfect.assign(
        pct_pla = n_perfect.n_pla/n_pla_gt,
        n_pla_g1 = perfect[
                perfect.n_contigs.ge(2)
            ].groupby("tool") \
            ["Hybrid contig ID"].size()
    )

    n_perfect = n_perfect.assign(pct_pla_g1 = n_perfect.n_pla_g1/n_pla_g1)
    
    results_perfect[taxon] = n_perfect

results_perfect = pd.concat(results_perfect)

filepath = str(outdir.joinpath("reconstruction.n_perfect_pla.tsv"))
results_perfect.to_csv(filepath, sep = "\t")
print(
    f"Saved the number of perfectly reconstructed plasmids as {filepath}.",
    "\t- n_pla: Number of perfectly reconstructed plasmids.",
    "\t- pct_pla: Proportion of perfectly reconstructed plasmids.",
    "\t- n_pla_g1: Number of perfectly reconstructed plasmids with more than one short-read contig.",
    "\t- pct_pla_g1: Proportion of plasmids with more than one short-read contig that were perfectly reconstructed",
    "Perfectly reconstructed means that all contigs from that plasmid were binned together, and the bin did not \
contain contigs from other plasmids. Note that this analysis excludes contigs < 1kbp.",
    file = report,
    sep = "\n"
)

# Plot results

fig, axs = plots.get_figure(1, 5, (14,3.5))

cmap = {
    k: v 
    for k, v in zip(
        taxa,
        sns.color_palette("Set1", 2)
    )
}

tool_names = [ 
    x
    for x in TOOL_ORDER_RECONSTRUCTION[::-1]
    if x in results.Tool.unique()
]

for n, (metric, ax) in enumerate(
    zip(
        metric_names,
        axs.flatten()
    )
):
    
    
    estimator = lambda x: np.median(x)
    errorbar = lambda x: np.percentile(x, [2.5, 97.5])
    ylim = (0.8, 1.01)

    for y, tool in enumerate(tool_names):

        for y_gap, taxon in enumerate(taxa):

            plot_data = results[ 
                results.Tool.eq(tool) & \
                results.Metric.eq(metric) & \
                results.dataset.eq(taxon)
            ]

            median = plot_data.value.median()
            ci = np.percentile(plot_data.value, [2.5, 97.5])
            if tool == "PlasBin-Flow":
                is_significant = True
            else:
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
        xlim = (.75, 1.01),
        xlabel = metric,
        ylabel = (
            "Tool"
            if n == 0
            else None
        ),
        xpercent = True,
        legend_title = ""
    )

    if n > 0:
        ax.set_yticks(
            axs[0].get_yticks(),
            []
        )
    else:
        ax.set_yticks(
            np.arange(len(tool_names)),
            tool_names
        )

# Plot the proportion of perfectly reconstructed plasmids and the proportion
# of perfectly reconstructed incomplete plasmids

for n, (ax, col) in enumerate(
    zip(axs[3:], ["pct_pla", "pct_pla_g1"])
):
    
    for y, tool in enumerate(tool_names):

        for y_gap, taxon in enumerate(taxa):

            x = results_perfect.loc[ 
                (taxon, tool),
                col
            ]

            ax.scatter(
                [x], [y + y_gap*.1 - .05],
                edgecolor = cmap[taxon],
                facecolor = cmap[taxon],
                zorder = y_gap + 2,
                label = taxon if y == 0 else None
            )

    if n == 1:
        ax.legend()

    plots.config_axes(
        ax,
        move_legend = (n == 1),
        xlabel = (
            "Proportion of perfectly\nreconstructed plasmids"
            if n == 0
            else "Proportion of perfectly\nreconstructed incomplete\nplasmids"
        ),
        xlim = (0, 1.05),
        ylabel = "",
        xpercent = True
    )

    ax.set_ylim(axs[0].get_ylim())

plots.save(
    outdir.joinpath("reconstruction.metrics"),
    ["eps", "png"]
)


report.close()

