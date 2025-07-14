
from utils.cli import DatasetANIParser
from utils.report import create_report
import pandas as pd
from matiss import plots as plots
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from colorcet import glasbey
from matplotlib.patches import Patch
import networkx as nx
plots.set_style()

parser = DatasetANIParser()
args = parser.parse()

outdir = Path(args.output)
outdir.mkdir(exist_ok=True, parents=True)

report = create_report(
    outdir,
    args,
    "dataset.ani.py"
)

# Load ANI 
ani = pd.read_table(
    "data/ani.tsv",
    index_col = 0
)

# Load contig info
gt = pd.read_excel(
    "data/predictions.xlsx",
    sheet_name = "Ground-truth",
    index_col = [0, 1]
)

# Load species assignments 
species = pd.read_excel(
    "data/predictions.xlsx",
    sheet_name = "Species",
    index_col = 0
)

# Join everything
ani = ani.join( 
    gt.reset_index() \
        .drop_duplicates(["Sample ID", "Hybrid contig ID"]) \
        .set_index(["Sample ID", "Hybrid contig ID"]) \
        .Taxon,
    on = ["Query sample ID", "Query hybrid contig ID"]
).join(
    species.Species.rename("query_species"),
    on = "Query sample ID"
).join(
    species.Species.rename("reference_species"),
    on = "Reference sample ID"
)

ani.loc[:, "Taxon"] = ani.Taxon.where(
    ~ani.Taxon.isna(),
    ani.query_species.str.startswith("Enterococcus").replace(
        {
            True: "Enterococcus",
            False: "Enterobacterales"
        }
    )
)

# Subset taxon
ani = ani[
    ani.Taxon.eq(args.taxon) & \
    ~ani.ANI.isna()
]
ani = ani.drop_duplicates()

# Ignore incomplete hybrid assemblies
complete = gt.reset_index() \
    .drop_duplicates("Sample ID") \
    .set_index("Sample ID") \
    ["Has complete hybrid assembly?"]
complete = complete[complete].index.to_list()

ani = ani[ 
    ani["Query sample ID"].isin(complete) & \
    ani["Reference sample ID"].isin(complete)
]

# Compare chromosomes

ani_chr = ani[ani["Query hybrid contig ID"].eq("chromosome")]

# Build a dendrogram
X = ani_chr.pivot(
    index="Query sample ID", 
    columns="Reference sample ID", 
    values="ANI"
).fillna(.7)

X = (X + X.transpose())/2
X = 1-X/100
links = linkage(squareform(X), "single")
unique_species = np.unique(
    np.hstack(
        [ani_chr.query_species.to_list(), ani_chr.reference_species.to_list()]
    )
)

colors_dict = dict(
    zip(
        unique_species, 
        sns.color_palette(glasbey, len(unique_species))
    )
)
handles = {
    k: Patch(facecolor=v) 
    for k,v in colors_dict.items() 
    if not pd.isna(k)
}

colors = species.loc[X.index, "Species"].map(colors_dict)
colors.name = "Species"
mask = np.zeros((len(X), len(X)), dtype=bool)
for n in range(len(X)):
    for m in range(len(X)):
        mask[n,m] = (colors.iloc[n]!=colors.iloc[m])

# Save the mean intra-species ANI
mean_intra_ani = (1-X).where(~mask, pd.NA).mean().mean()
print(
    f"Mean intra-species ANI: {mean_intra_ani*100: .3f}%", 
    file = report
)

grid = sns.clustermap(
    ((1-X)*100).where(((1-X)*100).ge(94)), 
    col_linkage=links, 
    row_linkage=links, 
    mask=mask, 
    row_colors=colors, 
    col_colors=colors, 
    cmap="gray_r", 
    cbar_kws={"label":"ANI (%)"}, 
    vmin=94
)
plots.config_axes(
    grid.ax_heatmap, 
    move_legend=False, 
    ylabel="Sample", 
    xlabel="Sample",
    grid=False, 
    despine=False
)
grid.ax_heatmap.set_xticklabels([])
grid.ax_heatmap.set_yticklabels([])
grid.ax_heatmap.tick_params(length=0)
grid.ax_col_colors.tick_params(length=0)
grid.ax_heatmap.legend(
    handles.values(), 
    handles.keys(), 
    title="Species",
    bbox_to_anchor=(1.05,.5), 
    loc="center left", 
    frameon=False
)
grid.ax_heatmap.grid(False)

filepath = str(outdir.joinpath("chr_dendrogram"))
plots.save(
    filepath,
    ["eps", "png"]
)

print(f"Rendered {filepath}.", file = report)

# Plot ANI between plasmids

ani_pla = ani[ani["Query hybrid contig ID"].ne("chromosome")]

# Build a dendrogram
X = ani_pla.assign(
    reference = ani_pla["Reference sample ID"] + "_" + ani_pla["Reference hybrid contig ID"],
    query = ani_pla["Query sample ID"] + "_" + ani_pla["Query hybrid contig ID"],
).pivot(
    index = "query",
    columns = "reference", 
    values = "ANI"
)

mask = X.isna()
X = X.fillna(70.0)

for n, m in zip(*np.diag_indices_from(X)): X.iloc[n,m]=100
X = (X + X.transpose())/2
X = 1-X/100
links = linkage(squareform(X), "single")

# Build a matrix of alignment fractions
X_af = ani_pla.assign(
    reference = ani_pla["Reference sample ID"] + "_" + ani_pla["Reference hybrid contig ID"],
    query = ani_pla["Query sample ID"] + "_" + ani_pla["Query hybrid contig ID"],
).pivot(
    index = "query",
    columns = "reference", 
    values = "Alignment fraction"
)
X_af = (X_af + X_af.transpose())/2

colors = species.loc[
    [x.split("_")[0] for x in X.index], 
    "Species"
].map(colors_dict)
colors.index = X.index 

grid = sns.clustermap(
        (1-X)*100, 
        col_linkage=links, 
        row_linkage=links,
        row_colors=colors, 
        col_colors=colors, 
        mask=mask,
        cmap="gray_r", 
        cbar_kws={"label":"ANI (%)"}, 
        cbar_pos=(.8,.3,.05,.5)
    )
plots.config_axes(
    grid.ax_heatmap, 
    move_legend=False, 
    ylabel="Plasmid", 
    xlabel="Plasmid",
    grid=False, 
    despine=False
)
grid.ax_heatmap.set_xticklabels([]), grid.ax_heatmap.set_yticklabels([])
grid.ax_heatmap.tick_params(length=0)

grid.ax_heatmap.grid(False)

filepath = str(outdir.joinpath("pla_dendrogram"))
plots.save(
    filepath,
    ["eps", "png"]
)

print(f"Rendered {filepath}.", file = report)

resolution_lims = (0.1, 3.0)

X_af = X_af.fillna(0.0)
X_af = X_af.where(X_af.ge(.5), 0.0)

# Cluster with Louvain clustering (go Belgium!). Select the resolution parameter
# that maximizes modularity
search_space = np.arange(*resolution_lims, .1)
search_results, clusters_ = {}, {}

for resolution in search_space:
    graph = nx.from_pandas_adjacency(X_af)
    clusters = nx.community.louvain_communities(
        graph, 
        seed=23, 
        resolution=resolution
    )
    # Compute modularity
    modularity = nx.community.modularity(graph, clusters)
    search_results[resolution] = {
        "modularity": modularity,
        "n_clusters": len(clusters)
    }
    # Save clusters
    clusters_[resolution] = clusters
search_results = pd.DataFrame(search_results) \
    .transpose() \
    .sort_values("modularity")
max_modularity = search_results.loc[search_results.modularity.idxmax()]

print(
    f"Clustered plasmid sequences based on alignment fraction with \
Louvain clustering. Selected the resolution parameter between \
{resolution_lims[0]} and {resolution_lims[1]} in increments of 0.1 \
that maximized modularity.",
    f"\t-Selected resolution: {search_results.iloc[-1].name}",
    f"\t-Modularity: {search_results.iloc[-1].modularity}",
    f"\t-Number of clusters: {search_results.iloc[-1].n_clusters}",
    sep = "\n",
    file = report
)

# Compute clustering stats with the best clustering results
best_clusters = clusters_[max_modularity.name]
# Mean intra-cluster alignment fraction
mean_ic_aln = [np.mean(X_af.loc[list(x), list(x)]) for x in best_clusters]
# Minimum intra-cluster alignment fraction
min_ic_aln = [np.min(X_af.loc[list(x), list(x)]) for x in best_clusters]
# Number of plasmids per cluster
n_pla = [len(x) for x in best_clusters]

# Maximum and mean inter-cluster distance
mask = pd.DataFrame(
    np.zeros_like(X_af.values), 
    index=X_af.index, 
    columns=X_af.columns,
    dtype="bool"
)
for cluster in best_clusters:
    for seq1 in list(cluster):
        for seq2 in list(cluster):
            mask.loc[seq1, seq2] = True
max_aln = np.max(X_af.where(~mask, None))
mean_aln = np.mean(X_af.where(~mask, None))

# Print and save results
print(
    f"Louvain clustering results for {args.taxon}:",
    f"Best resolution: {max_modularity.name:.1f}",
    f"Maximum modularity: {max_modularity.modularity:0.5f}",
    f"Number of plasmid clusters: {max_modularity.n_clusters:0.0f}\n",
    "Statistics from the best clustering (mean (SD)):",
    f"Mean intra-cluster alignment fraction: {np.mean(mean_ic_aln):.5f} ({np.std(mean_ic_aln):.5f})",
    f"Minimum intra-cluster alignment fraction: {np.mean(min_ic_aln):.5f} ({np.std(min_ic_aln):.5f})",
    f"Number of plasmids per cluster: {np.mean(n_pla):.5f} ({np.std(n_pla):.5f})",
    f"Maximum inter-cluster alignment fraction: {max_aln:.5f}",
    f"Mean inter-cluster alignment fraction: {mean_aln:.5f}",
    sep = "\n\t-",
    file = report
)

report.close()

