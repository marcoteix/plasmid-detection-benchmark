# plasmid-detection-benchmark
Supporting analysis code for "Circling in on plasmids: benchmarking plasmid detection and reconstruction tools for short-read data from diverse species"

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