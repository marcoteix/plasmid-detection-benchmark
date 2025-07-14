
from utils.cli import DetectionSRContigLengthParser
from utils.report import create_report
from utils import metrics
from utils.plots import BEST_TOOLS, TOOL_CMAP
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matiss import plots
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm
plots.set_style()

parser = DetectionSRContigLengthParser()
args = parser.parse()

outdir = Path(args.output)
outdir.mkdir(exist_ok=True, parents=True)

# Create report
report = create_report(args.output, args, "detection.sr_contig_length.py")

# Load predictions
predictions = pd.read_excel(
    args.input,
    index_col = [0, 1],
    sheet_name = "Plasmid Detection"
)

# Load ground-truth and SR contig characteristics
X = pd.read_excel(
    args.input,
    index_col = [0, 1],
    sheet_name = "Ground-truth"
)

# Define bins of SR contig lengths
bins = 10**np.arange(3, np.ceil(np.log10(100000))+1)
bins = np.hstack([np.linspace(a, b, 4)[:-1] for a,b in zip(bins[:-1], bins[1:])])
bins = np.hstack([bins, [100000]])
bin_centers, bin_lims = [], [[], []]

results = {}
n_pla, n_chr = {}, {}

X = X.join(predictions)
X = X[X.Taxon.eq(args.taxon)]

for n in tqdm(
    range(len(bins)-1), 
    total = len(bins)-1,
    desc = "Computing metrics for different SR contig lengths..."
):

    bin_centers.append(np.mean([bins[n], bins[n+1]]))
    bin_lims[0].append(int(bins[n]))
    bin_lims[1].append(int(bins[n+1]))

    # Get contigs with length within the bin
    y = X[
        X["SR contig length"].between(
            bins[n], 
            bins[n+1], 
            inclusive="left"
        )
    ]
    class_metrics = {}

    for m, y_bin in tqdm(
        enumerate(
            metrics.get_bootstrap_samples(y, niter=args.niter)
        ), 
        total = args.niter,
        desc = "Bootstrapping..."
    ):
        class_metrics[m] = pd.concat(
            {
                k: metrics.group_metrics(
                    y_bin.copy(), 
                    k, 
                    None, 
                    gt_col="Ground-truth class"
                ) 
                for k in predictions
            }
        )

    # Concatenate all iterations
    results[int(bins[n+1])] = pd.concat(class_metrics, names=["iteration", "tool", "metric"])

    # Get the number of plasmid and chromosomal contigs
    n_pla[int(bins[n+1])] = y["Ground-truth class"].eq("plasmid").sum()
    n_chr[int(bins[n+1])] = y["Ground-truth class"].eq("chromosome").sum()

results = pd.concat(results, names=["right_lim"])
n_pla = pd.Series(n_pla, name="n_pla", index=pd.Index(bin_lims[1], name="right_lim"))
n_chr = pd.Series(n_chr, name="n_pla", index=pd.Index(bin_lims[1], name="right_lim"))

# Write results
results = results.rename("value").reset_index()

filename = str(outdir.joinpath("classification.metrics.sr_contig_length.tsv"))
results.to_csv(
    filename, 
    sep="\t"
)
print(
    f"Saved plasmid detection metrics as {filename}.",
    file = report
)

filename = str(outdir.joinpath("n_pla.sr_contig_length.tsv"))
n_pla.to_csv(
    filename, 
    sep="\t"
)
print(
    f"Saved the number of plasmid SR contigs per bin as {filename}.",
    file = report
)

filename = str(outdir.joinpath("n_chr.sr_contig_length.tsv"))
n_chr.to_csv(
    filename, 
    sep="\t"
)
print(
    f"Saved the number of chromosomal SR contigs per bin as {filename}.",
    file = report
)

#  Compute the correlation between each metric and the contig length

# Take the mean over the bootstrapping iterations
means = results.groupby(
        ["right_lim", "tool", "metric"],
        as_index=False
    ).value.mean() \
    .groupby(["metric", "tool"], as_index=False) \
    .apply(
        lambda x: pd.Series(
            spearmanr(x.right_lim, x.value), 
            index=["r", "p_value"]
        )
    )
means = means.assign(significant=means.p_value.le(.05))

filename = str(outdir.joinpath("correlation.sr_contig_length.tsv"))
means.to_csv(
    filename, 
    sep="\t"
)
print(
    f"Saved the Spearman correlation between each metric and the maximum SR \
contig length per bin as {filename}.",
    file = report
)

pla_color, chr_color = (.8,.8,.8,.5), (.9,.9,.9,.5)

for metric in ["F1 score", "Precision", "Recall"]:

    fig, ax = plots.get_figure(1, 1, figsize=(6,4))


    sns.pointplot(
        results[
            results.tool.isin(BEST_TOOLS) & \
            results.metric.eq(metric)
        ], 
        x="right_lim", 
        y="value", 
        alpha=.75, 
        dodge=.5, 
        markersize=10, 
        ax=ax, 
        hue="tool", 
        palette=TOOL_CMAP,
        errorbar=lambda x: np.percentile(x, [2.5, 97.5]), 
        capsize=.1, 
        lw=0,
        err_kws={
            'linewidth': 2, 
            "alpha": .75
        }, 
        estimator=lambda x: np.median(x),
        zorder = 1
    )

    sns.pointplot(
        results[
            ~results.tool.isin(BEST_TOOLS) & \
            results.metric.eq(metric)
        ], 
        x = "right_lim", 
        y = "value", 
        alpha=.5, 
        dodge=.2, 
        lw=0, 
        markersize=5, 
        ax=ax, 
        hue="tool", 
        palette=TOOL_CMAP,
        errorbar=None, 
        estimator=lambda x: np.median(x)
    )

    left_lims = [1000] + list(results.right_lim.unique()[:-1])

    bins = [
        f"{b/1e3:.0f}-{a/1e3:.0f}" 
        for a,b in zip(
            results.right_lim.unique(), 
            left_lims
        )
    ]

    ax.set_xticks(
        np.arange(len(n_chr)), 
        bins
    )

    plots.config_axes(
        ax, 
        move_legend=True, 
        ylabel=metric, 
        ypercent=True,
        ylim=(0, 1.05), 
        xlabel=(
            "Short-read contig length (kbp)"
            if n == 1
            else None
        ), 
        title=args.taxon,
        grid=False
    )
    
    ax.grid(False)

filaname = str(outdir.joinpath(f"detection.metrics.contig_length.{metric}"))
plots.save(
    filename,
    format = ["png", "eps"]
)
print(
    f"Rendered {filename}.",
    file = report
)


report.close()


