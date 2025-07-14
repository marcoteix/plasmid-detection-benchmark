
from utils.cli import ReconstructionARGsParser
from utils.report import create_report
from utils.plots import TOOL_ORDER_RECONSTRUCTION, TOOL_CMAP
from utils import metrics
from scipy.stats import false_discovery_control
from sklearn import metrics as skm
from scipy import stats as sps
import numpy as np
import pandas as pd
from matiss import plots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
plots.set_style()

parser = ReconstructionARGsParser()
args = parser.parse()

outdir = Path(args.output)
outdir.mkdir(exist_ok=True, parents=True)

report = create_report(outdir, args, "reconstruction.args.py")

# Load predictions
X = pd.read_excel(
    args.input,
    sheet_name = "Plasmid Reconstruction",
    index_col = [0, 1]
).fillna("chromosome")

# Load ground-truth and join with predictions
X = X.join(
        pd.read_excel(
        args.input,
        sheet_name = "Ground-truth",
        index_col = [0, 1]
    )
).join(
    pd.read_excel(
        args.input,
        sheet_name = "Plasmid Characteristics",
        index_col = [0, 1]
    )
)

# Drop chromosomes and incomplete assemblies
X = X[ 
    X["Ground-truth class"].eq("plasmid") & \
    X["Has complete hybrid assembly?"] & \
    X["Taxon"].eq(args.taxon)
]

# For Enterococcus, ignore PlasBin-Flow as it did not find any plasmids
tools = TOOL_ORDER_RECONSTRUCTION.copy()
if args.taxon == "Enterococcus":
    tools.remove("PlasBin-Flow")

# Keep track of results for each class (ARGs vs. no ARGs)
results = {}

# Add a tag to each hybrid contig name and bin name with the sample name
Xb = X.copy().assign(
    tmp_sample = X.index.get_level_values("Sample ID")
)

for idx, row in Xb.iterrows():

    Xb.loc[idx, "hybrid_contig"] = ";".join(
        [
            x + "_" + row.tmp_sample 
            for x in row["Hybrid contig ID"].split(";")
        ]
    )

    for tool in tools:

        Xb.loc[idx, tool] = ";".join(
            [
                x + "_" + row.tmp_sample 
                for x in str(row[tool]).split(";")
            ]
        )
        
Xb = Xb.drop(columns="tmp_sample")

metrics_f = {
    "NMI": skm.normalized_mutual_info_score,
    "Homogeneity": skm.homogeneity_score, 
    "Completeness": skm.completeness_score
}

metric_names = list(metrics_f.keys())

for has_args in [True, False]:

    # Keep plasmids with/without ARGs
    Xb_subset = Xb[
        Xb["Hybrid contig has ARGs"].eq(has_args)
    ]

    # Estimate metrics through bootstrapping
    results[has_args] = {}

    for n, Xbg in tqdm(
        enumerate(
            metrics.get_bootstrap_samples(Xb_subset, niter = args.niter)
        ), 
        total = args.niter,
        desc = "Bootstrapping..."
    ):
        
        results[has_args][n] = {}
        
        for metric, mfunc in metrics_f.items():

            results[has_args][n][metric] = pd.Series(
                {
                    tool: mfunc(
                        *metrics.multilabel_to_single_label(
                            Xbg.hybrid_contig, 
                            Xbg[tool]
                        )
                    ) 
                    for tool in tools
                }
            )

        results[has_args][n] = pd.concat(
            results[has_args][n]
        )
    results[has_args] = pd.concat(
            results[has_args]
        )
results = pd.concat(
    results,
    names = ["Hybrid contig has ARGs", "iteration", "metric", "tool"]
)

filename = str(outdir.joinpath("reconstruction.args.tsv"))
results.to_csv(filename, sep = "\t")
print(
    f"Saved plasmid reconstruction metrics for ARG and non-ARG-containing \
plasmids as {filename}.",
    file = report
)

# Aggregate results

summary = results.rename("value") \
    .to_frame() \
    .reset_index() \
    .groupby(
        ["metric", "Hybrid contig has ARGs", "tool"]
    ).value.apply(
        lambda x: pd.Series(
            [
                x.median(),
                np.percentile(x, 2.5),
                np.percentile(x, 97.5),
                f"{x.median():.3f} ({np.percentile(x, 2.5):.3f}, {np.percentile(x, 97.5):.3f})"
            ],
            index=["median", "2.5", "97.5", "fmt"]
        )
    ).reset_index()

summary = summary[summary.level_3.eq("fmt")] \
    .pivot(
        index="tool", 
        columns=["metric", "Hybrid contig has ARGs"], 
        values="value"
    )

summary.columns.names = ["Metric", "Plasmid has ARGs?"]
summary.index.name = "Tool"

filename = str(outdir.joinpath("reconstruction.args.agg.tsv"))
summary.to_csv(filename, sep="\t")
print(
    f"Saved aggregate plasmid reconstruction metrics for ARG and non-ARG-containing \
plasmids as {filename}.",
    file = report
)

# Get the number of perfectly reconstructed plasmids in each group

perfect = []
for sample in X.index.get_level_values("Sample ID").unique():
    
    subset = X.loc[[sample]]

    for tool in tools:

        subset[tool] = subset[tool].astype(str)

        for gt_bin in subset["Hybrid contig ID"].str.split(";").explode().unique():
            
            # Are all contigs from this plasmid in the same bin?
            plasmid = subset[
                subset["Hybrid contig ID"].str.contains(gt_bin)
            ]

            unique_bin = any(
                [
                    plasmid[tool].str.contains(x).all() 
                    for x in plasmid[tool].str.split(";").explode().unique()
                ]
            )

            if (
                plasmid[tool].isna().any() or \
                plasmid[tool].isin(["chr", "chromosome"]).any() \
                or not unique_bin
            ): 
                continue

            for pred in subset[tool].str.split(";").explode().unique():
                
                # Does that bin have contigs from other plasmids?
                if (
                    subset.loc[
                        subset[tool].str.contains(pred),
                        "Hybrid contig ID"
                    ].str.contains(gt_bin).all()
                ) and not pred in ["chr", "chromosome"]:
                    
                    perfect.append(
                        [
                            sample, 
                            gt_bin, 
                            tool, 
                            pred, 
                            len(plasmid), 
                            plasmid["Hybrid contig has ARGs"].any()
                        ]
                    )

                    break

perfect = pd.DataFrame(
    perfect, 
    columns=[
        "sample", 
        "hybrid_contig", 
        "tool", 
        "prediction", 
        "n_contigs", 
        "plasmid_has_arg"
    ]
)

n_pla_gt = X.assign(
        hybrid_contig = X["Hybrid contig ID"].str.split(";")
    ).explode("hybrid_contig") \
    .reset_index() \
    .groupby(
        ["Sample ID", "Hybrid contig has ARGs"],
        as_index=False
    )["Hybrid contig ID"].nunique() \
    .groupby("Hybrid contig has ARGs") \
    ["Hybrid contig ID"].sum()

# Find the number of plasmids with >1 frag-only contig
n_contig_per_pla = X["Hybrid contig ID"].str \
    .split(";") \
    .explode() \
    .reset_index() \
    .groupby(
        ["Sample ID", "Hybrid contig ID"], 
        as_index=False
    ) \
    .size() \
    .join(
        X.reset_index() \
            .drop_duplicates(["Sample ID", "Hybrid contig ID"]) \
            .set_index(["Sample ID", "Hybrid contig ID"]) \
            ["Hybrid contig has ARGs"],
        on = ["Sample ID", "Hybrid contig ID"]
    )

pla_g1 = n_contig_per_pla[n_contig_per_pla["size"].ge(2)]

n_pla_g1 = pla_g1.groupby(
        ["Hybrid contig has ARGs", "Sample ID", "Hybrid contig ID"],
        as_index=False
    )["Hybrid contig ID"] \
    .nunique() \
    .groupby("Hybrid contig has ARGs") \
    ["Hybrid contig ID"].sum()

n_perfect = perfect.groupby(
        ["plasmid_has_arg", "tool"]
    ).hybrid_contig \
    .size() \
    .to_frame("n_pla")

n_perfect = n_perfect \
    .assign(
        pct_pla = n_perfect.n_pla/n_perfect \
            .reset_index() \
            .join(n_pla_gt, on="plasmid_has_arg") \
            .set_index(["plasmid_has_arg", "tool"]) \
            ["Hybrid contig ID"],
        n_pla_g1 = perfect[perfect.n_contigs.ge(2)] \
            .groupby(["plasmid_has_arg", "tool"]) \
            .hybrid_contig \
            .size()
    )

n_perfect = n_perfect.assign(
    pct_pla_g1 = n_perfect.n_pla_g1/n_perfect \
            .reset_index() \
            .join(n_pla_g1, on="plasmid_has_arg") \
            .set_index(["plasmid_has_arg", "tool"]) \
            ["Hybrid contig ID"]
)

filename = str(outdir.joinpath("reconstruction.args.n_perfect.tsv"))
n_perfect.to_csv(filename, sep="\t")
print(
    f"Saved the number of perfectly reconstructed ARG and non-ARG-containing \
plasmids as {filename}.",
    file = report
)

print(
    "\t- n_pla: Number of perfectly reconstructed plasmids",
    "\t- pct_pla: Proportion of perfectly reconstructed plasmids",
    "\t- n_pla_g1: Number of perfectly reconstructed plasmids with more than one short-read contig", 
    "\t- pct_pla_g1: Proportion of plasmids with more than one short-read contig that were perfectly reconstructed",
    "Perfectly reconstructed means that all contigs from that plasmid were binned together, and the bin did not \
contain contigs from other plasmids. Note that this analysis excludes contigs < 1kbp.",
    sep = "\t",
    file = report
)

# Compare metrics across groups. Get p-values

p_values = {}

results = results.rename("value")
    
for metric in metric_names:

    p_values[metric] = {}

    for tool in tools:

        results_subset = results.xs(
            (metric, tool),
            axis = 0,
            level = [2, 3]
        )

        p_values[metric][tool] = metrics.difference_pvalues(
            results_subset.loc[True],
            results_subset.loc[False],
            paired = True
        )

        p_values[metric][tool] = pd.Series(
            p_values[metric][tool],
            name = "value"
        )

    p_values[metric] = pd.concat(p_values[metric])

p_values = pd.concat(
    p_values,
    names = ["metric", "tool", "x"]
).reset_index() \
    .drop(columns = "x") \
    .pivot(
        index = "tool", 
        columns = "metric", 
        values = "value"
    )

# Apply FDR correction
p_values = pd.Series(
    false_discovery_control(p_values.unstack().dropna()),
    index = p_values.unstack() \
        .dropna() \
        .index,
    name = "p"
).to_frame() \
    .reset_index() \
    .pivot(
        index = "tool",
        columns = "metric",
        values = "p"
    )

filename = str(outdir.joinpath("reconstruction.args.pvalues.tsv"))
p_values.to_csv(filename, sep="\t")
print(
    f"Saved p-values as {filename}.",
    file = report
)

# For the proportion of perfectly reconstructed plasmids, use the Fisher exact test

# Get the total number of plasmids with and without ARGs
n_pla = n_perfect.assign(
    total = (n_perfect.n_pla / n_perfect.pct_pla),
    total_g1 = (n_perfect.n_pla_g1 / n_perfect.pct_pla_g1)
)

# Add the expected frequencies
n_pla = n_pla.assign(
    expected = (n_pla.total / 2),
    expected_g1 = (n_pla.total_g1 / 2)
).reset_index()

p_values_perfect = {}

for tool in tools:

    n_subset = n_pla[
        n_pla.tool.eq(tool)
    ].fillna(0)

    # Build a contingency table for all plasmids
    contingency = [
        [
            n_subset.loc[
                n_subset.plasmid_has_arg, 
                "n_pla"
            ].values[0],
            (
                n_subset.loc[
                    n_subset.plasmid_has_arg, 
                    "total"
                ] - n_subset.loc[
                    n_subset.plasmid_has_arg, 
                    "n_pla"
                ]
            ).values[0]
        ],
        [
            n_subset.loc[
                ~n_subset.plasmid_has_arg, 
                "n_pla"
            ].values[0],
            (
                n_subset.loc[
                    ~n_subset.plasmid_has_arg, 
                    "total"
                ] - n_subset.loc[
                    ~n_subset.plasmid_has_arg, 
                    "n_pla"
                ]
            ).values[0]
        ]
    ]

    # And for those with > 1 contig
    contingency_g1 = [
        [
            n_subset.loc[
                n_subset.plasmid_has_arg, 
                "n_pla_g1"
            ].values[0],
            (
                n_subset.loc[
                    n_subset.plasmid_has_arg, 
                    "total_g1"
                ] - n_subset.loc[
                    n_subset.plasmid_has_arg, 
                    "n_pla_g1"
                ]
            ).values[0]
        ],
        [
            n_subset.loc[
                ~n_subset.plasmid_has_arg, 
                "n_pla_g1"
            ].values[0],
            (
                n_subset.loc[
                    ~n_subset.plasmid_has_arg, 
                    "total_g1"
                ] - n_subset.loc[
                    ~n_subset.plasmid_has_arg, 
                    "n_pla_g1"
                ]
            ).values[0]
        ]
    ]

    # Get the total number of plasmids with and without ARGs
    p_values_perfect[tool] = pd.Series(
        {
            "n_pla": sps.fisher_exact(contingency) \
                .pvalue,
            "n_pla_g1": sps.fisher_exact(contingency_g1) \
                .pvalue
        },
        name = "p"
    )

p_values_perfect = pd.concat(
    p_values_perfect,
    names = [
        "tool",
        "type"
    ]    
).to_frame() \
    .reset_index() \
    .pivot(
        index = "tool",
        columns = "type",
        values = "p"
)

filename = str(outdir.joinpath("reconstruction.args.n_perfect_pvalues.tsv"))
p_values_perfect.to_csv(filename, sep="\t")
print(
    f"Saved p-values for the proportion of perfectly reconstructed plasmids as {filename}.",
    file = report
)
# Plot figures

cmap = {
    k: c
    for k, c in zip(
        [True, False],
        sns.color_palette("Set1", n_colors = 2)
    )
}

fig, axs = plots.get_figure(1, 5, (14,3.5))

for n, (metric, ax) in enumerate(
    zip(
        metric_names,
        axs.flatten()
    )
):
    for y, tool in enumerate(tools[::-1]):

        for y_gap, group in enumerate([True, False]):

            plot_data = results.xs(
                (group, metric, tool),
                axis = 0,
                level = [0, 2, 3]
            )

            median = plot_data.median()
            ci = np.percentile(plot_data, [2.5, 97.5])

            is_significant = p_values.loc[tool, metric] < 0.05

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

    if n == 0:
        ax.legend(frameon = False)

    plots.config_axes(
        ax = ax,
        xlabel = metric,
        ylabel = "Tool" if n == 0 else None,
        legend_title = "Plasmid has ARGs?",
        move_legend = False,
        xlim = (.55, 1.01),
        xpercent = True,
        title = (
            args.taxon
            if n == 2
            else ""
        )
    )

    if n > 0:
        ax.set_yticks(
            axs[0].get_yticks(),
            []
        )
    else:
        ax.set_yticks(
            np.arange(len(tools)),
            tools[::-1]
        )

# Plot the proportion of perfectly reconstructed plasmids and the proportion
# of perfectly reconstructed incomplete plasmids

for n, (ax, col) in enumerate(
    zip(axs[3:], ["pct_pla", "pct_pla_g1"])
):
    
    for y, tool in enumerate(tools[::-1]):

        for y_gap, group in enumerate([True, False]):

            x = n_perfect.loc[(group, tool), col]

            ax.scatter(
                [x], [y + y_gap*.1 - .05],
                edgecolor = cmap[group],
                facecolor = cmap[group],
                zorder = y_gap + 2
            )

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

filename = str(outdir.joinpath("reconstruction.args"))
plots.save(
    filename,
    ["eps", "png"]
)
print(f"Rendered {filename}.", file = report)


report.close()

