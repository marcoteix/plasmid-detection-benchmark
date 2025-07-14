
from utils.cli import ReconstructionGLMParser
from utils.report import create_report
from utils.metrics import binning_metrics_per_sample
from sklearn import metrics as skm
import statsmodels.api as sm
import pandas as pd
from pathlib import Path 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

parser = ReconstructionGLMParser()
args = parser.parse()

outdir = Path(args.output)
outdir.mkdir(exist_ok=True, parents=True)

report = create_report(outdir, args, "reconstruction.glm.py")

########################### Create feature and target matrices ###########################

# Load assembly and plasmid characteristics
X = pd.read_excel(
    args.input,
    sheet_name = "Plasmid Characteristics",
    index_col = [0, 1]
).join(
    pd.read_excel(
        args.input,
        sheet_name = "Assembly Statistics",
        index_col = [0, 1]
    )
).join(
    pd.read_excel(
        args.input,
        sheet_name = "Ground-truth",
        index_col = [0, 1]
    )
)

# Load plasmid reconstruction predictions
predictions = pd.read_excel(
    args.input,
    sheet_name = "Plasmid Reconstruction",
    index_col = [0, 1]
).fillna("chromosome")

tools = predictions.columns

X = X.join(predictions)

# Discard incomplete hybrid assemblies
X = X[X["Has complete hybrid assembly?"]]

Y = binning_metrics_per_sample(
    X.rename(
        columns={"Hybrid contig length": "hybrid_contig_length"}
    ), 
    {"NMI": skm.normalized_mutual_info_score}, 
    tools, 
    X.reset_index()["Sample ID"], 
    filtering="non_na"
).reset_index() \
.drop_duplicates() \
.set_index("Sample ID") \
.xs(
    "NMI",
    axis = 1,
    level = 1
)

# Create a feature matrix by taking aggregate statistics per sample
features = [
    "Mean number of SR contigs per hybrid contig",
    "Number of hybrid contigs",
    "Number of insertion sequences (IS) in assembly",
    "Number of SR contigs with ARGs",
]

X = X[features].assign(
    **{
        "Number of insertion sequences (IS) in assembly":
            X["Number of insertion sequences (IS) in assembly"].apply(
                lambda x: int(str(x).split(";")[0])
            ).astype(int)
    }
)

X = X.reset_index() \
    .drop_duplicates(["Sample ID"]) \
    .drop(columns = ["SR contig ID"]) \
    .set_index("Sample ID") \
    .loc[Y.index]

X = sm.add_constant(X)

# Ignore samples for which the tools predicted no plasmids
has_pla = predictions.ne("chromosome") \
    .groupby(["Sample ID"]) \
    [predictions.columns].any() \
    .loc[X.index]

print(
    f"Number of samples in the dataset: {len(X)}",
    "Fitting linear regression models...",
    sep = "\n",
    file = report
)

########################### Fit linear regression model ###########################

reg_metrics = {"R2": r2_score, "MSE": mean_squared_error}
summaries, scores = {}, {}
for tool in tools:

    # Ignore samples for which the tools predicted no plasmids
    X_tool = X[has_pla[tool]]
    Y_tool = Y[has_pla[tool]]

    #% Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_tool, Y_tool, test_size=0.2, random_state=23
    )

    # Fit the model
    glm = sm.OLS(y_train[tool], X_train)
    result = glm.fit()

    # Save the summary
    summaries[tool] = result
    
    scores[tool] = pd.concat(
        {
            "train": pd.Series(
                {
                    k: v(y_train[tool], result.predict(X_train)) 
                    for k,v in reg_metrics.items()
                }
            ),
            "test": pd.Series(
                {
                    k: v(y_test[tool], result.predict(X_test))
                    for k,v in reg_metrics.items()
                }
            )
        }
    )

    print(
        f"Fitted model for {tool}.",
        f"\t- Number of samples after discarding samples for which {tool} predicted \
no plasmids: {len(X_tool)}",
        f"\t- Model p-value: {result.f_pvalue}",
        f"\t- Training R2: {result.rsquared}",
        sep = "\n",
        file = report
    )

scores = pd.concat(scores, names=["Tool"])

# ################ Generate tables ################

# Holds tables for each tool
tables = {}
# Holds model p-values
model_ps = {}

for tool, summary in summaries.items():

    # Format coefficients
    coeffs = summary.params \
        .rename("coeff") \
        .to_frame() \
        .join(
            summary.conf_int()
        ).apply(
            lambda x: f"{x.coeff:.3f} ({x[0]:.3f}, {x[1]:.3f})",
            axis = 1
        ).rename("Coefficient")
    
    # Format p-values
    pvalues = summary.pvalues.apply(
        lambda x: "< 0.001" if x < .001 else f"{x:.3f}"
    ).rename("p")

    # Indicator for the effect
    effect = summary.pvalues.gt(.05).replace({True: "="}).astype(str)
    effect = effect.where(
        effect.eq("="),
        summary.params.gt(0).replace(
            {
                True: "▲",
                False: "▼"
            }
        )
    ).rename("Covariate effect")

    tables[tool] = pd.concat(
        [
            coeffs,
            pvalues,
            effect
        ],
        axis = 1
    ).rename(
        index = {
            "const": "Intercept",
            "Mean number of SR contigs per hybrid contig": "Average number of SR contigs per plasmid and chromosome",
            "Number of hybrid contigs": "Number of plasmids and chromosomes",
            "Number of insertion sequences (IS) in assembly": "Number of transposases in the hybrid assembly"
        }
    )

    model_ps[tool] = result.f_pvalue

tables = pd.concat(
    tables, 
    names = ["Tool", "Covariate"]
)

model_ps = pd.Series(model_ps, name = "Model p")

filename = str(outdir.joinpath("reconstruction.glm.coeffs.tsv"))
tables.to_csv(filename, sep = "\t")
print(
    f"Saved linear regression coefficients as {filename}",
    file = report
)

filename = str(outdir.joinpath("reconstruction.glm.model_p.tsv"))
model_ps.to_csv(filename, sep = "\t")
print(
    f"Saved linear regression model p-values as {filename}",
    file = report
)

filename = str(outdir.joinpath("reconstruction.glm.scores.tsv"))
scores.to_csv(filename, sep = "\t")
print(
    f"Saved linear regression training and test regression metrics as {filename}",
    file = report
)

report.close()

