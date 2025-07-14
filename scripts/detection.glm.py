
import io
from utils.cli import DetectionGLMParser
from utils.plots import TOOL_CMAP, BEST_TOOLS
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from matplotlib import pyplot as plt 
import matplotlib as mpl
import seaborn
from matiss import plots
import pandas as pd
from pathlib import Path
import numpy as np
plots.set_style()

parser = DetectionGLMParser()
args = parser.parse()

outdir = Path(args.output)
outdir.mkdir(exist_ok=True, parents=True)

################ Create a feature matrix and label vector ################


# Load predictions, ground truth, and characteristics
with pd.ExcelFile(args.input) as predictions_file:

    X = pd.concat(
        [
            pd.read_excel(
                predictions_file,
                index_col = [0, 1],
                sheet_name = sheet
            )
            for sheet in predictions_file.sheet_names
            if not sheet in ["Species", "Plasmid Reconstruction", "Plasmid Reconst. (Best Det.)"]
        ],
        axis = 1
    )

# Subset target taxon and complete assemblies
X = X[
    X.Taxon.eq(args.taxon) & \
    X["Has complete hybrid assembly?"]
]

# Explode rows with multiple hybrid contigs
cols = [ 
    "Hybrid contig ID",
    "Hybrid contig length",
    "Is a large plasmid?",
    "Mash identity to closest PLSDB plasmid",
    "Number of insertion sequences (IS) in assembly"
]
identity_col = "Mash identity to closest PLSDB plasmid"

# Fill identity to closest PLSDB hit for chromosomes with the median across plasmids
X = X.assign(
    **{
        identity_col: X[identity_col].fillna(
            (X[identity_col][~X[identity_col].astype(str).str.contains(";")]).astype(float).median()
        )
    }
)
X = X.assign(
    **{
        k: X[k].astype(str).str.split(";")
        for k in cols
    }
)
X = X.explode(cols)

# Get a list of tool names
tools = pd.read_excel(
    args.input,
    index_col = [0,1],
    sheet_name = "Plasmid Detection"
).columns.to_list()

# Convert the predictions into "correct" (TP/TN) -> True and "incorrect" (FP/FN) -> False
for tool in tools: 
    X.loc[:, tool] = X[tool].eq(X["Ground-truth class"])

# Target vectors
Y = X[tools].astype(int)
# Feature matrix
X = X[ 
    [
        "SR contig length",
        identity_col,
        "Number of insertion sequences (IS) in assembly",
        "Is a large plasmid?",
        "SR contig has ARGs"
    ]
]

# Convert contig lengths to Mb and boolean variables to 0/1
X = X.assign(
    **{
        "SR contig length": X["SR contig length"]/1e6,
        "SR contig has ARGs": X["SR contig has ARGs"].astype(int),
        "Mash identity to closest PLSDB plasmid": X["Mash identity to closest PLSDB plasmid"].astype(float),
        "Number of insertion sequences (IS) in assembly": X["Number of insertion sequences (IS) in assembly"].astype(float)
    }
)

# Dummify (one-hot encode) the categorical "large_pla" variable
X = pd.concat(
    [ 
        X,
        pd.get_dummies(
            X["Is a large plasmid?"], 
            prefix = "large_pla"
        ).drop(
            columns = "large_pla_False"
        ).astype(int)
    ],
    axis = 1
).drop(
    columns = "Is a large plasmid?"
)

X = sm.add_constant(X)

################ Fit the logistic regression model ################

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, 
    test_size = 0.2, 
    random_state = 23, 
    stratify = X["large_pla_chromosome"]
)

summaries, scores = {}, {}
for tool in ["Plasmer", "PlasmidEC", "gplas2", "PlaScope"]:

    # Fit the model
    glm = sm.Logit(y_train[tool], X_train)
    result = glm.fit(maxiter = 10000)

    # Save the summary
    
    summaries[tool] = result.summary() \
        .as_csv() \
        .replace(",", "\t")
    
    scores[tool] = pd.Series(
        {
            "train": roc_auc_score(y_train[tool], result.predict(X_train)),
            "test": roc_auc_score(y_test[tool], result.predict(X_test))
        }
    )

scores = pd.concat(
    scores, 
    names=["Tool"]
)

scores.to_csv(
    outdir.joinpath(f"detection.glm.auc.tsv"), 
    sep="\t"
)

################ Generate tables ################

# Maps row names to feature names
row_to_feature = {
    "const": "1 Intercept",
    "SRcontiglength": "2 SR contig length (per Mbp)",
    "SRcontighasARGs": "4 Has ARGs",
    "Numberofinsertionsequences(IS)inassembly": "5 # transposases in hybrid assembly",
    "large_pla_True": "6 Is from a large plasmid",
    "large_pla_chromosome": "7 Is from a chromosome",
    "MashidentitytoclosestPLSDBplasmid": "8 Identity of the best PLSDB hit"
}

# Holds tables for each taxon and tool
tables = []

# Holds model p-values
model_ps = []

for tool, summary in summaries.items():

    # Extract results for the log-likelihood ratio test
    model_p = float(
        summary \
            .split("LLR p-value:")[1] \
            .split("\n")[0]
    )

    if model_p < 1e-3:
        model_p = "< 0.001"
    else:
        model_p = f"{model_p:0.3f}"

    model_ps.append(
        pd.Series(
            {
                "p": model_p,
                "dataset": args.taxon,
                "tool": tool
            }
        )
    )
    
    # Read coefficients and p-values
    coeffs = pd.read_table(
        io.StringIO(summary),
        skiprows = 8
    ).iloc[:7,:]

    coeffs = coeffs.assign(
        attr = coeffs.iloc[:,0] \
            .str.replace(" ", "")
    )

    coeffs.columns = [
        str(x).replace(" ", "")
        for x in coeffs
    ]

    # Add formatted coefficients with 95% CI limits
    tables.append(
        pd.DataFrame(
            dict(
                coeff_fmt = coeffs.apply(
                    lambda x: (
                        f"{x.coef:0.3f} (" + \
                        f"{x['[0.025']:0.3f}, " + \
                        f"{x['0.975]']:0.3f})"
                    ),
                    axis = 1
                ),
                p_fmt = coeffs["P>|z|"].apply(
                    lambda x: (
                        f"{x:0.3f}"
                        if x >= 1e-3
                        else "< 0.001"
                    )
                ),
                zeffect = coeffs.apply(
                    lambda x: (
                        "=" 
                        if x["P>|z|"] > .05
                        else (
                            "▲" 
                            if x.coef > 0
                            else "▼"
                        )
                    ),
                    axis = 1
                ),
                attr = coeffs.attr,
                dataset = args.taxon,
                tool = tool
            )
        )
    )

# Concatenate
tables = pd.concat(tables)

tables = tables.assign(
        attr = tables.attr \
            .replace(row_to_feature)
    ).melt(
        id_vars = ["dataset", "tool", "attr"]
    ).pivot(
        columns = [
            "tool",
            "variable"
        ],
        index = [
            "dataset",
            "attr"
        ],
        values = "value"
    ).sort_index(
        axis = 1
    )

tables.to_csv(
    Path(args.output).joinpath("coeff.table.glm.tsv"),
    sep = "\t"
)

# Format AUC table
aucs = scores.to_frame() \
    .reset_index() \
    .rename(
        columns = {
            "Tool": "tool",
            "level_1": "subset",
            0: "value"
        }
    ).assign(dataset = args.taxon) \
    .pivot(
        columns = "tool",
        index = ["dataset", "subset"],
        values = "value"
    )*100

aucs = aucs.round(1) \
    .astype(str) + "%"

aucs.to_csv(
    Path(args.output) \
        .joinpath("auc.table.glm.tsv"),
    sep = "\t"
)

# Format model p-values
model_ps = pd.concat(
        [
            x.to_frame().transpose()
            for x in model_ps
        ]
    ) \
    .pivot(
        columns = "tool",
        index = "dataset",
        values = "p"
    )

model_ps.to_csv(
    outdir.joinpath("model_p.table.glm.tsv"),
    sep = "\t"
)

################ Plot coefficients ################

# Convert to a machine-friendly table
coeffs = {}

for tool in tables.columns.get_level_values(level = 0):

    tool_table = tables[tool].reset_index() 
    
    # Extract coeff and interval limits
    tool_table = tool_table.assign(
        coeff = tool_table.coeff_fmt \
            .apply(
                lambda x: float(x.split(" ")[0])
            ),
        left = tool_table.coeff_fmt \
            .apply(
                lambda x: float(x.split(" ")[1][1:-1])
            ),
        right = tool_table.coeff_fmt \
            .apply(
                lambda x: float(x.split(" ")[2][:-1])
            ),
        p = tool_table.p_fmt \
            .replace("< 0.001", "0.001") \
            .astype(float)
    )

    coeffs[tool] = tool_table

attr_names = {
    "2 SR contig length (per Mbp)": "SR contig length\n(Mbp)",
    "8 Identity of the best PLSDB hit": "Identity of the\nbest PLSDB hit",
    "5 # transposases in hybrid assembly": "# transposases in\n hybrid assembly",
    "7 Is from a chromosome": "Is from the\nchromosome*",
    "6 Is from a large plasmid": "Is from a\nlarge plasmid*",
    "4 Has ARGs": "Has ARGs",
    "1 Intercept": "Intercept",
}

coeffs = pd.concat(
        coeffs,
        names = ["tool"]
    ).reset_index() \
    .drop(
        columns="level_1"
    ).replace(
        attr_names
    )

order = list(attr_names.values())[::-1]


fig, ax = plots.get_figure(1, 1, (6,4))

for n, tool in enumerate(BEST_TOOLS[::-1]):

    subset = coeffs[ 
        coeffs.tool.eq(tool) & \
        coeffs.dataset.eq(args.taxon)
    ].set_index("attr")

    subset = subset.loc[order]

    for m, (attr, row) in enumerate(
        subset.iterrows()
    ):
        ax.plot(
            [row.left, row.right],
            [m + ((.5/4)*n-.2)]*2,
            color = TOOL_CMAP[tool],
            zorder = 5
        )

        ax.scatter(
            row.coeff,
            [m + ((.5/4)*n-.2)],
            color = TOOL_CMAP[tool],
            zorder = 5,
            s = 10
        )

    # Add line at 0
    ax.vlines(
        0, *ax.get_ylim(),
        color = ".3",
        lw = 1,
        zorder = 1,
        ls = "-"
    )

    # Add legend
    if n == 1:
        ax.legend(
            handles = [
                mpl.lines.Line2D(
                    [0], [0],
                    marker = "o",
                    color = color,
                    label = tool
                )
                for tool, color in TOOL_CMAP.items()
                if tool in BEST_TOOLS
            ]
        )

# Add bands
xlims = ax.get_xlim()


for m in range(len(order)):
    ax.fill_between(
        xlims,
        y1 = m - .4,
        y2 = m + .4,
        color = ".95",
        zorder = -1
    )

plots.config_axes(
    ax = ax,
    move_legend = True,
    xlabel = "Coefficient",
    ylim = (-.5, len(subset)-.5),
    xlim = xlims,
    title = args.taxon
)

ax.set_yticks(
    np.arange(len(order)),
    order
)

ax.grid(
    False,
    axis="y"
)
ax.grid(
    axis = "x",
    color = ax.get_facecolor(),
    zorder = 0,
    lw = 1,
    alpha = 1
)

ax.set_xscale(
    "symlog",
    base = 2
)


plots.save(
    outdir.joinpath("detection.glm.coeffs"),
    format = ["png", "eps"]
)




