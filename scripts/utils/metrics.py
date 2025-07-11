#%%
from functools import partial
from itertools import product
from typing import Callable, Hashable, Literal
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matiss import plots as plots
from matiss.stats import pairwise_test as pair_t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorcet import glasbey
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import logging
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker
from sklearn import metrics as skm

def compare_groups(X, values: str, groups: str, index: str):
    grp = X[groups].unique()
    assert len(grp) == 2, f"Expected 2 unique groups, but found {len(grp)}."
    idx = X[index].unique()

    significant = {}
    for i in idx:
        ci_1 = np.percentile(X[X[index].eq(i) & X[groups].eq(grp[0])][values], [2.5, 97.5])
        ci_2 = np.percentile(X[X[index].eq(i) & X[groups].eq(grp[1])][values], [2.5, 97.5])

        significant[i] = not ((ci_1[0]>ci_2[0]) and (ci_1[0]<ci_2[1]) or (ci_1[1]>ci_2[0]) and (ci_1[1]<ci_2[1]))

    return pd.Series(significant)

def pairwise_test(X: pd.DataFrame, values_col: Hashable, group_col: Hashable, test: Callable, p_thresh: float = 0.05):
    """
    Applies a statistical test comparing all possible pairs of groups determined in the `group_col`.
    """

    report = []
    
    groups = {k: X[values_col][X[group_col]==k] for k in X[group_col].unique()}
    N = len(groups)
    p_table = pd.DataFrame(np.zeros((N, N)), columns=groups.keys(), index=groups.keys())
    stat_table = pd.DataFrame(np.zeros((N, N)), columns=groups.keys(), index=groups.keys())

    for i, (k, v) in enumerate(groups.items()):
        for j, (kk, vv) in enumerate(groups.items()):
            if j <= i: continue
            stat, p = test(v, vv)
            stat_table.iloc[i,j], stat_table.iloc[j,i] = stat, stat
            p_table.iloc[i,j], p_table.iloc[j,i] = p, p

            report.append(f"{group_col}={k} vs {kk}:\n\tStatistic: {stat}\n\tp-value: {p}\n\tSignificant?: {p<p_thresh}")

    return stat_table, p_table, "\n".join(report)

def group_metrics(x: pd.DataFrame, tool: str, min_len: int, sample_weight=None, 
                  pos_class="plasmid", gt_col: str = "category"):
    
    metrics = {}
    x.loc[:, tool] = x[tool].eq(pos_class)
    x.loc[:, gt_col] = x[gt_col].eq(pos_class)
    x[tool] = x[tool].astype(bool)
    x[gt_col] = x[gt_col].astype(bool)
    if min_len: x = x[x["SR contig length"]>=min_len]
    if sample_weight: sample_weight = x["SR contig length"].astype(int)
    metrics["Precision"] = precision_score(x[gt_col], x[tool], sample_weight=sample_weight, zero_division=np.nan)
    metrics["Recall"] = recall_score(x[gt_col], x[tool], sample_weight=sample_weight, zero_division=np.nan)
    metrics["Accuracy"] = accuracy_score(x[gt_col], x[tool], sample_weight=sample_weight)
    metrics["F1 score"] = f1_score(x[gt_col], x[tool], sample_weight=sample_weight, zero_division=np.nan)
        
    if not np.any(x[gt_col]): 
        # If the sample does not have any instance of the positive class, ignore metrics
        metrics["Recall"] = np.nan
        metrics["F1 score"] = np.nan
        metrics["Accuracy"] = np.nan
        metrics["Precision"] = np.nan
    elif not np.any(x[tool]): 
        metrics["Precision"] = np.nan
        metrics["F1 score"] = np.nan

    rc = pd.Series(metrics, name=tool)
    rc.index.name = "Metric"
    return rc

def multilabel_to_single_label(y_true, y_pred, sep=";"):

    if not len(y_true) == len(y_pred):
        raise ValueError(f"Got y_true and y_pred with different lengths: {len(y_true)} and {len(y_pred)}.")
    
    if isinstance(y_true, pd.Series): y_true = y_true.values
    if isinstance(y_pred, pd.Series): y_pred = y_pred.values
    
    y_true_single_label_mask = np.array([len(x.split(sep)) < 2 for x in y_true])
    y_pred_single_label_mask = np.array([len(x.split(sep)) < 2 for x in y_pred])
    assert len(y_true_single_label_mask) == len(y_pred_single_label_mask), "Mismatched len"

    single_label_mask = y_true_single_label_mask & y_pred_single_label_mask

    y_true_single_label, y_pred_single_label = y_true[single_label_mask], y_pred[single_label_mask]
    y_true_multilabel, y_pred_multilabel = y_true[~single_label_mask], y_pred[~single_label_mask]

    y_true_prod, y_pred_prod = np.array([]), np.array([])
    for y_t, y_p in zip(y_true_multilabel, y_pred_multilabel):
        crossprod = np.array(list(product(y_t.split(sep), y_p.split(sep))))
        y_true_prod = np.hstack([y_true_prod, crossprod[:,0]])
        y_pred_prod = np.hstack([y_pred_prod, crossprod[:,1]])

    return np.hstack([y_true_single_label, y_true_prod]), np.hstack([y_pred_single_label, y_pred_prod])

def plot_results(results: pd.DataFrame, metrics: list, figdir: Path, filtering: str = "intersection",
    plot_significance: bool = True):

    if filtering == "non_na":
        subtitle = "\n(chromosomal and plasmid contigs)"
        suffix = "_pla_pred"
    elif filtering == "intersection":
        subtitle = "\n(true plasmid contigs and predicted as plasmid by all tools)"
        suffix = "_all_pla"
    fmt = lambda x: x.set_xticks(x.get_xticks(), 
            [xx.get_text().replace("+","\n+\n").replace("(","\n(") for xx in x.get_xticklabels()])

    results = results[results.xs("NMI", axis=1, 
                                 level=1).mean().sort_values(ascending=False).index]
    for metric in metrics:
        X_long = results.xs(metric, level=1, axis=1).reset_index().melt(id_vars="sample")
        f, ax = plots.get_figure()
        sns.boxplot(X_long, x="Tool", y="value", ax=ax, fill=False, flierprops={"marker": None}, 
                    color="black", whis=(0, 100))
        sns.stripplot(X_long, x="Tool", y="value", ax=ax, alpha=.3, size=10, color="black")
        plots.config_axes(ax, move_legend=False, title=metric + subtitle, ylabel=metric)
        if plot_significance:
            p_table = pair_t(X_long, value="value", variable="Tool", test=stats.ttest_rel)
            plots.add_significance(ax, p_table=p_table, p_col="p_corrected")
        fmt(ax)

        plt.savefig(figdir.joinpath(f"{metric.replace(' ','_')}"+suffix+".eps"))
        plt.savefig(figdir.joinpath(f"{metric.replace(' ','_')}"+suffix+".png"), bbox_inches="tight")
        plt.show()

    # Plot the means and sds of metrics
    f, ax = plots.get_figure(figsize=(14,7))
    x = results.drop("No. plasmid contigs", level=1, axis=1).reset_index(col_fill="sample").melt(id_vars=[("sample", "sample")])
    sns.barplot(x, x="Tool", y="value", estimator=np.mean, hue="Metric", ax=ax, palette="gray", 
        errorbar="sd", capsize=.1, err_kws={"linewidth": 1}, lw=1, edgecolor="black")
    plots.config_axes(ax, xlabel="", ylabel="Average per sample", ypercent=True, ylim=(0,ax.get_ylim()[1]))
    fmt(ax)
    plt.savefig(figdir.joinpath("metrics"+suffix+".eps"))
    plt.savefig(figdir.joinpath("metrics"+suffix+".png"), bbox_inches="tight")
    plt.show()

def binning_metrics_per_sample(
    X: pd.DataFrame, 
    metrics: dict, 
    tools: list, 
    samples: list, 
    filtering: str = "intersection", 
    drop_multilabel: bool = False
):

    assert filtering in ["intersection", "non_na"]

    results = []
    for tool in tools:
        cols = pd.MultiIndex.from_tuples([(tool, x) for x in metrics.keys()], names=("Tool", "Metric"))
        tool_results = pd.DataFrame([], index=samples, columns=cols)
        for metric, m_func in metrics.items(): 

            def auxf(x):
                if x[tool].isna().all(): return 0
                y_true, y_pred = multilabel_to_single_label(x["Hybrid contig ID"], x[tool].astype(str))
                return m_func(y_true, y_pred)

            # Filter contigs according to the selected scheme
            if filtering == "intersection":
                # Keep contigs classified as plasmid by all tools
                logging.debug("Keeping frag-only contigs classified as plasmid by all tools and dropping all others.")
                X_filtered = X[~X[tools].isna().all(axis=1)]
                logging.debug(f"There are {len(X_filtered)} frag-only contigs classified as plasmid by all tools.")
            elif filtering == "non_na":
                logging.debug(f"Dropping frag-only contigs classified as non-plasmid by {tool}.")
                X_filtered = X[~X[tool].isna()]
                logging.debug(f"There are {len(X_filtered)} frag-only contigs classified as plasmid by {tool}.")

            if drop_multilabel:
                logging.debug(f"Dropping frag-only contigs with multiple GT bins or multiple predicted bins by tool '{tool}'.")
                n_init = len(X_filtered)
                X_filtered = X_filtered[~X_filtered["Hybrid contig ID"].apply(lambda x: ";" in x) & ~ X_filtered[tool].astype(str) \
                    .apply(lambda x: ";" in x)]
                logging.debug(f"Dropped {n_init-len(X_filtered)} frag-only contigs.")
            else:
                logging.debug(f"Exploding (taking the cross product) results for multiple bins per contig for tool '{tool}'")
                X_filtered = explode_results(X_filtered, tool)

            tool_results[tool, metric] = X_filtered.reset_index() \
                .groupby("Sample ID").apply(auxf, include_groups=False)
        results.append(tool_results)
    return pd.concat(results, axis=1)

def plot_binning_predictions(results: pd.DataFrame, tool: str, tools: list, ax_grid = None, drop_multilabel: bool = False,
    figdir: Path = None, *, hybrid_len_col: str = "hybrid_contig_length", frag_len_col: str = "frag_contig_length",
    hybrid_col: str = "hybrid_contig"):

    X_tool = results.drop([x for x in tools if x != tool], axis=1)
    n_samples, samples = X_tool.index.nunique(), X_tool.index.unique()

    if ax_grid is None:
        ax_grid = (np.ceil(np.sqrt(n_samples)).astype(int), np.ceil(np.sqrt(n_samples)).astype(int))
    f, axs = plt.subplots(*ax_grid, constrained_layout=True, figsize=(40,20))
    if n_samples == 1: axs = np.array([axs])

    for sample, ax in zip(samples, axs.flatten()):
        # Get results for that sample
        X = X_tool.loc[[sample], :]

        if drop_multilabel:
            logging.debug(f"Dropping frag-only contigs with multiple GT bins or multiple predicted bins by tool '{tool}'.")
            n_init = len(X)
            X = X[~X[hybrid_col].apply(lambda x: ";" in x) & ~ X[tool].astype(str).apply(lambda x: ";" in x)]
            logging.debug(f"Dropped {n_init-len(X)} frag-only contigs.")
            unique_preds = X[tool].astype(str).unique()
        else:
            logging.debug(f"Exploding (taking the cross product) results for multiple bins per contig for tool '{tool}'")
            X = explode_results(X, tool, False, hybrid_length_col=hybrid_len_col)
            unique_preds = X.assign(**{tool: X[tool].astype(str).str.split(";")}).explode(tool)[tool].unique()

        n_plasmids, plasmids = X.loc[:, hybrid_col].nunique(), X.loc[:, hybrid_col].unique()
        ax_width = 1/n_plasmids

        # Map predicted bins to colors
        cmap = get_bin_cmap(unique_preds, X[tool], glasbey)
        cmap[np.nan] = (.8, .8, .8)
        ax.set_facecolor((.95, .95, .95))

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
        ax.set_title(sample)
        sns.despine(ax=ax, bottom=True, left=True)
    if figdir is not None:
        logging.debug(f"Saving predictions figures to {str(figdir)}...")
        plt.savefig(figdir.joinpath(f"predictions_{tool.replace(' ','_')}.eps"))
        plt.savefig(figdir.joinpath(f"predictions_{tool.replace(' ','_')}.png"), bbox_inches="tight")
    plt.show()

def explode_results(X: pd.DataFrame, tool: str, explode_preds: bool = True, hybrid_length_col: str = "hybrid_contig_length"):
    try:
            X = X.assign(hybrid_contig=X[hybrid_length_col].str.split(";"),
                hybrid_contig_length=X[hybrid_length_col].astype(str).str.split(";")).explode(["hybrid_contig", 
                hybrid_length_col]) 
            X.loc[:, hybrid_length_col] = X.loc[:, hybrid_length_col].astype(int)
            if explode_preds:
                X = X.assign(**{tool: X[tool].astype(str).str.split(";")}).explode(tool)
    except AttributeError as e: logging.warning(f"Explode results for tool {tool} got AttributeError '{e}'.")
    except ValueError as e: logging.warning(f"Explode results for tool {tool} got ValueError '{e}'.")
    return X

def get_bin_cmap(unique_preds: list, preds: pd.Series, cmap):

    colors = {bin_: color for bin_, color in zip(unique_preds, sns.color_palette(cmap, len(unique_preds)))}
    return colors

def get_n50(x: pd.DataFrame, len_col="frag_contig_length"):

    total_len = x[len_col].sum()
    x_sorted = x.sort_values(len_col)
    x_sorted = x_sorted.assign(cumsum=x_sorted[len_col].cumsum())
    return x_sorted[x_sorted["cumsum"] > total_len/2][len_col].iloc[0]

def to_cm(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:

    # Convert labels to binary
    y, y_hat = y_true.eq("plasmid"), y_pred.eq("plasmid")
    cm = y.eq(y_hat).replace({True: "T", False: "F"}) + y_hat \
        .replace({True: "P", False: "N"})
    cm.name = y_hat.name
    return cm

def binning_permutations(gt: pd.Series, n_iter: int = 100, metric: Callable = skm.mutual_info_score, random_state: int = 23):

    # Explode GT bins
    gt = gt.astype(str).str.split(";").explode()
    gt = gt + "_" + pd.Series(gt.index.get_level_values(0), index=gt.index)
    outcome = []
    
    rand_preds = {}
    for sample in gt.index.get_level_values(0):
        n_contigs = len(gt.loc[sample])
        pred_bins = np.random.default_rng(random_state).integers(0, n_contigs, (n_contigs, n_iter))

        rand_preds[sample] = pd.DataFrame(np.char.add(pred_bins.astype(str), f"_{sample}"), 
            index=gt.loc[sample].index)
    rand_preds = pd.concat(rand_preds)

    # Merge GT and predictions again
    rand_preds = pd.concat([rand_preds.reset_index() \
        .groupby(["level_0", "Contig"])[x].apply(lambda x: ";".join(x))
        for x in rand_preds], axis=1)
    gt = gt.rename("gt").reset_index() \
        .groupby(["Sample", "Contig"])["gt"].apply(lambda x: ";".join(x))
    
    outcome = [metric(*multilabel_to_single_label(gt, rand_preds[it])) for it in rand_preds]
    return outcome

def add_sample_tag(X: pd.DataFrame, tools):
    Xbg = X.copy()
    Xbg["tmp_sample"] = X.index.get_level_values("Sample")
    Xbg = Xbg.sort_index()
    for idx, row in Xbg.iterrows():
        Xbg.loc[idx, "hybrid_contig"] = ";".join(
            [x+"_"+row.tmp_sample for x in row.hybrid_contig.split(";")])
        for tool in tools:
            Xbg.loc[idx, tool] = ";".join(
                [x+"_"+row.tmp_sample for x in str(row[tool]).split(";")])
    Xbg = Xbg.drop(columns="tmp_sample").loc[X.index]
    return Xbg

def ami(y_true: pd.Series, y_pred: pd.Series, *, exp_mi: float = None, add_tag: bool = True, **kwargs):
    """Adjusted mutual information, with the expected mutual information computed through sampling.

    Note: the ground truth and predicted bins must NOT have the sample names added.

    Args:
        y_true (pd.Series): Ground truth bins. Must contain a MultiIndex with samples and contigs.
        y_pred (pd.Series, pd.DataFrame): Predicted bins. Must contain a MultiIndex with samples and contigs.
            If it is a DataFrame, this function computes the AMI for each column.
    """

    # Compute the expected MI
    if exp_mi is None:
        exp_mi = np.mean(binning_permutations(y_true, metric=skm.mutual_info_score, **kwargs))
    y_true.name = "hybrid_contig"

    if isinstance(y_pred, pd.DataFrame):
        # Compute the AMI once per column
        amis = pd.Series({col: ami(y_true, y_pred[col], exp_mi=exp_mi) for col in y_pred})
        return amis
    
    else:
        if add_tag: X = add_sample_tag(pd.concat([y_true, y_pred], axis=1), tools=[y_pred.name])
        else: X = pd.concat([y_true, y_pred], axis=1)
        # Compute the MI
        mi = skm.mutual_info_score(*multilabel_to_single_label(X.hybrid_contig, X[y_pred.name]))
        # Compute the normalization factor
        norm = np.mean([skm.cluster.entropy(X.hybrid_contig), skm.cluster.entropy(X[y_pred.name])])

        # Return the AMI
        return (mi-exp_mi)/(norm-exp_mi)

def bootstrap(y_true: pd.Series, y_pred: pd.Series, metric: Callable, *, random_state: int = 23, 
    ratio: float=.9, n_iter: int = 50, ci_level: float = 95.0, 
    estimate: str = "ci", **kwargs):

    n_samples = len(y_true)
    # Sample rows with replacement
    rows = np.random.default_rng(random_state).integers(0, n_samples, (n_iter,int(n_samples*ratio)))

    results = []
    # Speed-up the AMI computation by estimating the expected MI with the complete dataset

    if not isinstance(y_true, pd.Series): y_true = pd.Series(y_true)
    if not isinstance(y_pred, pd.Series) and not isinstance(y_pred, pd.DataFrame): y_pred = pd.Series(y_pred)

    for i in range(n_iter):
        obs = metric(y_true.iloc[rows[i,:]], y_pred.iloc[rows[i,:]], **kwargs)
        results.append(obs)
    
    if isinstance(y_pred, pd.DataFrame): results = pd.concat(results, axis=1).transpose().values
    else: results = np.array(results)

    if estimate == "ci":
        # Compute the percentiles
        pct = np.percentile(results, [(100-ci_level)/2, (100+ci_level)/2], axis=0)
        if isinstance(y_pred, pd.DataFrame): pct = pd.Series([(pct[0,n], pct[1,n]) for n in range(pct.shape[1])], 
            index=y_pred.columns)
        return pct
    elif estimate == "sd": return np.std(results)
    elif estimate == "mean": return np.mean(results)

def get_bootstrap_samples(X: pd.DataFrame, niter: int = 1000, ratio: float = 1.0, *, random_state: int = 23):

    sample_ids = X.index.get_level_values("Sample ID").unique()
    # Sample from the sample IDs
    n_to_sample = int(len(sample_ids)*ratio)
    choices = np.random.default_rng(random_state).choice(sample_ids, size=(niter, n_to_sample))

    # Return an iterator with the sliced dataset
    return (X.loc[x] for x in choices)

def difference_pvalues(a: pd.Series, b: pd.Series, *, paired=True, n_samples=None,
    kind: Literal["two-sided", "greater", "less"] = "two-sided"):

    # Take the difference between groups
    if paired: diff = (b-a)
    else: diff = (b.sample(n_samples, random_state=23, replace=True) - a \
        .sample(n_samples, random_state=23, replace=True))
    p1, p2 = diff.ge(0).mean(), diff.le(0).mean()

    if kind == "two-sided": return np.minimum(1, 2*np.minimum(p1, p2))
    elif kind == "greater": return p1
    elif kind == "less": return p2
    else: raise ValueError(f"Unknown kind \"{kind}\".")