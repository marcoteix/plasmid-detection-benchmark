# Circling in on plasmids: benchmarking plasmid detection and reconstruction tools for short-read data from diverse species

Supporting analysis code for _Circling in on plasmids: benchmarking plasmid detection and reconstruction tools for short-read data from diverse species_.

The predictions of plasmid detection and reconstruction tools, assembly statistics, other plasmid information (such as plasmid size, ARG presence/absence and the number of transposases), and ANI between chromosomes and plasmids are in `./data/`.

Python scripts used for analysis are in `./scripts/` and include: 

| Category | Script  | Description |
| -------- | ------- | ----------- |
| Dataset  | dataset.args.fragmentation.py | Compares the number of IS, SR contigs, and plasmid length between plasmids with ARGs and plasmids without ARGs |
| Dataset  | dataset.plasmid_length.py | Plots the distribution of plasmid lengths in samples with complete hybrid assemblies |
| Dataset  | dataset.ani.py | Plots heatmaps of ANI between chromosomes and plasmids. Also clusters plasmids based on their alignment fraction |
| Dataset  | dataset.plsdb.py | Compares the Mash identity to the closest PLSDB plasmids for different taxa |
| Detection  | detection.metrics.py | Calculates plasmid detection metrics from a set of predictions and plots the results |
| Detection  | detection.glm.py | Fits a Logistic Regression model to estimate the contribution of certain plasmid and assembly features to plasmid detection |
| Detection  | detection.sr_contig_length.py | Compares the plasmid detection metrics for SR contigs of different lenghts |
| Detection  | detection.rep_cluster.py | Compares the plasmid detection metrics for plasmids of different Inc types/rep clusters |
| Detection | detection.args.py | Compares the plasmid detection metrics for SR contigs with and without ARGs |
| Detection | detection.plasmid_size.py | Compares the plasmid detection metrics for plasmids of different sizes (large vs. small) |
| Reconstruction | reconstruction.metrics.py | Calculates plasmid reconstruction metrics from a set of predictions and plots the results |
| Reconstruction | reconstruction.glm.py | Fits a Linear Regression model to estimate the contribution of certain assembly features to plasmid reconstruction |
| Reconstruction | reconstruction.metrics_best_detector.py | Calculates plasmid reconstruction metrics from a set of predictions, using initial contig classifications from the best plasmid detection tools, and plots the results |
| Reconstruction | reconstruction.args.py | Compares plasmid reconstruction metrics for plasmids with and without ARGs |

You can run each script individually or all at once with the script in `./data/run-everything`.

The exact package versions in machine-readable format used to generate the results presented in the manuscript are in `pkgs.versions.txt`.

## Usage

1. Clone this repository and navigate into it:

    ```
    cd plasmid-detection-benchmark
    ```

2. Create a new virtual environment and install the required packages. If you are using conda:

    ```
    conda create --name plasmid-env --file requirements.yaml
    conda activate plasmid-env
    ```

3. Run all scripts with:
    ```
    sh scripts/run-everything
    ```
    Or run individual scripts with:
    ```
    python scripts dataset.ani.py
    ````
    You can check the command line arguments for each script with `python script.name.py --help`.

## Outputs

By default, the output files are written to `./outputs/`. If running the `run-everything` script, the result of each individual script will be written to a separate directory within `./outputs/`.

## Authors

Marco Teixeira (mcarvalh@broadinstitute.org), Celia Souque, Colin J. Worby, Terrance Shea, Nicoletta Commins, Joshua T. Smith, Arjun M. Miklos, Thomas Abeel, Ashlee M. Earl, and Abigail L. Manson.