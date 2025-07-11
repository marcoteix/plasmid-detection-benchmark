#%%
from matplotlib.font_manager import get_font_names
import glob
import seaborn as sns
from matplotlib.font_manager import fontManager, FontProperties
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from pathlib import Path
import numpy as np
from colorcet import glasbey
from matiss import plots
from utils import metrics

TOOL_CMAP = {
    'PlaScope': (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
    'Plasmer': (0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
    'PlasmidEC': (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
    'gplas2': (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
    'Platon': (0.0, 0.6745098039215687, 0.7764705882352941),
    'HyAsP': (0.6313725490196078, 0.4588235294117647, 0.4117647058823529),
    'RFPlasmid': (1.0, 0.49411764705882355, 0.8196078431372549),
    'plASgraph2': (0.3411764705882353, 0.23137254901960785, 0.0),
    'PLASMe': (0.0, 0.33725490196078434, 0.34901960784313724),
    'PlasmidFinder': (0.5490196078431373, 0.23137254901960785, 1.0),
    'geNomad': (0.0, 0.9921568627450981, 0.8117647058823529),
    "MOB-recon": (0.592156862745098, 1.0, 0.0),
    "PlasBin-Flow": (1.0, 0.6470588235294118, 0.1843137254901961)
}

BEST_TOOLS = ["PlaScope", "gplas2", "PlasmidEC", "Plasmer"]

TOOL_ORDER = [
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

TOOL_ORDER_RECONSTRUCTION = [
    "gplas2",
    "MOB-suite",
    "HyAsP",
    "PlasBin-Flow"
]

TOOL_STYLES = {
    'Centrifuge': "-",
    'PlaScope': "-",
    'PlasmidEC': "--",
    'gplas2': "--",
    'Platon': "-",
    'HyAsP': "--",
    'RFPlasmid': "-",
    'Plasmer': "-",
    'MOB-recon ++': "-",
    'plASgraph2': "--",
    'PLASMe': "-",
    'PlasmidFinder': "-",
    'geNomad': "-",
    'MOB-recon': "-",
    "gplas2 + PlasmidEC": "-",
    "gplas2\n+ PlasmidEC": "-",
    "MOB-suite": "-",
    "PlasBin-Flow": "-"    
}

COLORS = np.array([
    [42, 76, 189],
    [255, 128, 159],
    [116, 232, 131],
    [232, 201, 116],
    [147, 90, 106]
])/255

def list_fonts():

    return [x.split("/")[-1].replace(".tff", "") for x in 
        glob.glob("/usr/share/fonts/truetype/*/*")]

def set_font(fontname: str):
    path = glob.glob("/usr/share/fonts/truetype/*/"+fontname+".ttf")[0]
    fontManager.addfont(path)

    prop = FontProperties(fname=path)
    sns.set(font=prop.get_name())

def config_axes(ax:Axes, *, move_legend:bool=True, grid:bool=True, 
    legend_title=None, xlabel=None, ylabel=None, xlog:bool=False, ylog:bool=False,
    xrotation=None, ypercent:bool = False, title=None, xlim=None, ylim=None):

    sns.despine(ax=ax)
    if move_legend:
        try: 
            sns.move_legend(ax, loc="center left", bbox_to_anchor=(1, 0.5), 
                            frameon=False, title=legend_title)
        except:
            ax.legend()
            sns.move_legend(ax, loc="center left", bbox_to_anchor=(1, 0.5), 
                            frameon=False, title=legend_title)
    if grid: ax.grid(alpha=.3)
    ax.set(ylabel=ylabel, xlabel=xlabel, title=title)
    if xlog: ax.set_xscale("log")
    if ylog: ax.set_yscale("log")
    if xrotation: ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), 
        va="top", 
        ha="right" if not xrotation in [0, 90] else "center", rotation=xrotation)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if ypercent: ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
    return ax

def plot_global_metrics(metrics_df: pd.DataFrame, title:str, 
                        figdir: Path, figname:str, palette = None):
    sns.set_context("talk")
    sns.set_style("white")
    sns.set_palette("Set1")

    metric_order = ["F1 score", "Precision", "Recall", "Accuracy"]
    if palette is None: palette = {k: v for k,v in zip(metric_order, COLORS)}
    else: palette = {k: v for k,v in zip(metric_order, sns.color_palette(palette, 4))}
    tool_order = metrics_df.loc["F1 score"].sort_values(ascending=False).index
    metrics_df = metrics_df.loc[metric_order][tool_order]

    f, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(12,6))
    lwm = metrics_df.reset_index().melt(id_vars="Metric")
    sns.barplot(lwm, x="variable", y="value", hue="Metric", ax=ax, width=.75, 
                saturation=.6, lw=1, palette=palette,
                edgecolor=palette["F1 score"])
    config_axes(ax, legend_title="Metric", title=title, xrotation=45, ypercent=True, ylim=(0,1))
    for imgtype in [".eps", ".png"]: plt.savefig(figdir.joinpath(figname+imgtype))
    plt.show()
    return f, ax

def get_bin_cmap(unique_preds: list, cmap):

    colors = {bin_: color for bin_, color in zip(unique_preds, sns.color_palette(cmap, len(unique_preds)))}
    return colors

def explode_results(X: pd.DataFrame, tool: str, explode_preds: bool = True, hybrid_length_col: str = "hybrid_contig_length"):
    X = X.assign(**{"hybrid_contig" :X[hybrid_length_col].str.split(";"),
        hybrid_length_col: X[hybrid_length_col].astype(str).str.split(";")}).explode(["hybrid_contig", 
        hybrid_length_col]) 
    X.loc[:, hybrid_length_col] = X.loc[:, hybrid_length_col].astype(int)
    if explode_preds:
        X = X.assign(**{tool: X[tool].astype(str).str.split(";")}).explode(tool)
    return X

def plot_binning_prediction(results: pd.DataFrame, tool: str,  *, hybrid_len_col: str = "hybrid_length", 
    frag_len_col: str = "frag_length", hybrid_col: str = "hybrid_contig", **kwargs):

    # Explode hybrid contig assignments and results
    X = explode_results(results, tool, False, hybrid_length_col=hybrid_len_col)
    unique_preds = X[tool].astype(str).str.split(";").explode().unique()

    fig, ax = plots.get_figure(1, 1, **kwargs)

    n_plasmids, plasmids = X.loc[:, hybrid_col].nunique(), X.loc[:, hybrid_col].unique()
    ax_width = 1/n_plasmids

    # Map predicted bins to colors
    cmap = get_bin_cmap(unique_preds, glasbey)
    cmap[np.nan] = (.8, .8, .8)
    
    for n, plasmid in enumerate(plasmids):
        x_start = ax_width*n
        pla_ax = ax.inset_axes([x_start, 0, ax_width, 1])
        X_pla = X[X[hybrid_col]==plasmid]

        radius = np.log10(X_pla[hybrid_len_col].sum())/7
        pla_ax.pie(X_pla[frag_len_col], radius=radius, labels = X_pla[tool].values, labeldistance=None, 
                    wedgeprops=dict(edgecolor=ax.get_facecolor()))
        # Color slices
        width = radius - radius*.7
        for slice in pla_ax.patches:
            if ";" in slice.get_label():
                bins = slice.get_label().split(";")
                for i in range(len(bins)):
                    bin_slice = plt.Circle((0,0), radius-(width/len(bins)*i), fc = cmap[bins[i]])
                    bin_slice.set_clip_path(slice.get_path(), transform=pla_ax.transData)
                    pla_ax.add_patch(bin_slice)
            else:
                label_ = slice.get_label()
                if label_ == "nan": label_ = np.nan
                slice.set_facecolor(cmap[label_])
        pla_ax.add_patch(plt.Circle((0,0), radius*.7, fc=ax.get_facecolor()))

    ax.set_xticks([]), ax.set_yticks([])
    sns.despine(ax=ax, bottom=True, left=True)

    return ax

# %%
