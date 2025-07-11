import argparse
import sys 
from abc import ABC, abstractmethod
import os 

class CLIParser(ABC): 

    def __init__(
        self,
        name: str,
        description: str = "" 
    ):
        
        # Check if it is running in a Jupyter notebook
        self.notebook = sys.argv[0].endswith("ipykernel_launcher.py")

        self.parser = argparse.ArgumentParser(
            name,
            description = description
        )

        self.add_arguments()

    @abstractmethod
    def add_arguments(self, **kwargs):
        """
        Add arguments to the parser.
        This method should be implemented by subclasses.
        """
        pass

    def parse(self):

        return self.parser.parse_args("" if self.notebook else None)

class DatasetArgsISParser(CLIParser):

    def __init__(self):
        super().__init__(
            "Number of IS, SR contigs, and plasmid length in ARG+ and ARG- plasmids",
            description="Checks for diferences in the number of insertion sequences, \
short-read contigs, and plasmid length, between plasmids in finished hybrid \
assemblies with and without ARGs."
        )

    def add_arguments(self, **kwargs):

        self.parser.add_argument(
            "--input", "-i",
            type = str,
            default = "data/predictions.xlsx",
            help = "Table containing predictions. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--output", "-o",
            type = str,
            default = "outputs/dataset/dataset.args.fragmentation",
            help = "Output directory. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--taxon", "-t",
            type = str, choices = ["Enterococcus", "Enterobacterales"],
            default = "Enterobacterales",
            help = "Target taxon. Must be either \"Enterococcus\" or \
\"Enterobacterales\". Defaults to \"%(default)s\". "
        )

class DatasetPlasmidLengthParser(CLIParser):

    def __init__(self):
        super().__init__(
            "Plasmid Length",
            description="Plots histograms of plasmid lengths."
        )

    def add_arguments(self, **kwargs):

        self.parser.add_argument(
            "--input", "-i",
            type = str,
            default = "data/predictions.xlsx",
            help = "Table containing predictions. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--output", "-o",
            type = str,
            default = "outputs/dataset/dataset.plasmid_length",
            help = "Output directory. Defaults to %(default)s."
        )

class DatasetANIParser(CLIParser):

    def __init__(self):
        super().__init__(
            "ANI",
            description="Plots heatmaps of ANI between chromosomes and \
between plasmids. Clusters plasmid sequences according to ANI."
        )

    def add_arguments(self, **kwargs):

        self.parser.add_argument(
            "--input", "-i",
            type = str,
            default = "data/predictions.xlsx",
            help = "Table containing predictions. Defaults to %(default)s."
        )
        
        self.parser.add_argument(
            "--ani", "-a",
            type = str,
            default = "data/ani.tsv",
            help = "Table containing ANI values. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--output", "-o",
            type = str,
            default = "outputs/dataset/dataset.ani",
            help = "Output directory. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--taxon", "-t",
            type = str, choices = ["Enterococcus", "Enterobacterales"],
            default = "Enterobacterales",
            help = "Target taxon. Must be either \"Enterococcus\" or \
\"Enterobacterales\". Defaults to \"%(default)s\". "
        )

class DatasetPLSDBParser(CLIParser):

    def __init__(self):
        super().__init__(
            "PLSDB",
            description="Compares the Mash identity to the closest PLSDB plasmids for different taxa."
        )

    def add_arguments(self, **kwargs):

        self.parser.add_argument(
            "--input", "-i",
            type = str,
            default = "data/predictions.xlsx",
            help = "Table containing predictions. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--output", "-o",
            type = str,
            default = "outputs/dataset/dataset.plsdb",
            help = "Output directory. Defaults to %(default)s."
        )

class DetectionMetricsParser(CLIParser):

    def __init__(self):
        super().__init__(
            "Plasmid detection metrics",
            description="Computes the performance metrics for plasmid detection \
tools given their predictions."
        )

    def add_arguments(self, **kwargs):

        self.parser.add_argument(
            "--input", "-i",
            type = str,
            default = "data/predictions.xlsx",
            help = "Table containing predictions. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--output", "-o",
            type = str,
            default = "outputs/detection/detection.metrics",
            help = "Output directory. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--niter", "-n",
            type = int, default = 1000,
            help = "Number of bootstrapping iterations. Defaults to %(default)s."
        )

class DetectionGLMParser(CLIParser):

    def __init__(self):
        super().__init__(
            "Logistic regression (plasmid detection)",
            description="Checks for characteristics impacting plasmid detection by \
fitting a logistic regression model to the predictions of each of the top four tools."
        )

    def add_arguments(self, **kwargs):

        self.parser.add_argument(
            "--input", "-i",
            type = str,
            default = "data/predictions.xlsx",
            help = "Table containing predictions. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--output", "-o",
            type = str,
            default = "outputs/detection/detection.glm",
            help = "Output directory. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--taxon", "-t",
            type = str, choices = ["Enterococcus", "Enterobacterales"],
            default = "Enterobacterales",
            help = "Target taxon. Must be either \"Enterococcus\" or \
\"Enterobacterales\". Defaults to \"%(default)s\". "
        )

class DetectionSRContigLengthParser(CLIParser):

    def __init__(self):
        super().__init__(
            "Plasmid detection performance as a function of SR contig length",
            description="Compares the plasmid detection metrics for SR contigs of \
different lenghts."
        )

    def add_arguments(self, **kwargs):

        self.parser.add_argument(
            "--input", "-i",
            type = str,
            default = "data/predictions.xlsx",
            help = "Table containing predictions. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--output", "-o",
            type = str,
            default = "outputs/detection/detection.sr_contig_length",
            help = "Output directory. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--taxon", "-t",
            type = str, choices = ["Enterococcus", "Enterobacterales"],
            default = "Enterobacterales",
            help = "Target taxon. Must be either \"Enterococcus\" or \
\"Enterobacterales\". Defaults to \"%(default)s\". "
        )

        self.parser.add_argument(
            "--niter", "-n",
            type = int, default = 1000,
            help = "Number of bootstrapping iterations. Defaults to %(default)s."
        )

class DetectionRepClusterParser(CLIParser):

    def __init__(self):
        super().__init__(
            "Plasmid detection performance as a function of inc type/rep cluster",
            description="Compares the plasmid detection metrics for plasmids of \
different Inc types/rep clusters."
        )

    def add_arguments(self, **kwargs):

        self.parser.add_argument(
            "--input", "-i",
            type = str,
            default = "data/predictions.xlsx",
            help = "Table containing predictions. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--output", "-o",
            type = str,
            default = "outputs/detection/detection.rep_cluster",
            help = "Output directory. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--taxon", "-t",
            type = str, choices = ["Enterococcus", "Enterobacterales"],
            default = "Enterobacterales",
            help = "Target taxon. Must be either \"Enterococcus\" or \
\"Enterobacterales\". Defaults to \"%(default)s\". "
        )

        self.parser.add_argument(
            "--niter", "-n",
            type = int, default = 1000,
            help = "Number of bootstrapping iterations. Defaults to %(default)s."
        )

class DetectionARGsParser(CLIParser):

    def __init__(self):
        super().__init__(
            "Plasmid detection performance as a function of ARG presence",
            description="Compares the plasmid detection metrics for SR contigs \
with and without ARGs."
        )

    def add_arguments(self, **kwargs):

        self.parser.add_argument(
            "--input", "-i",
            type = str,
            default = "data/predictions.xlsx",
            help = "Table containing predictions. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--output", "-o",
            type = str,
            default = "outputs/detection/detection.args",
            help = "Output directory. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--taxon", "-t",
            type = str, choices = ["Enterococcus", "Enterobacterales"],
            default = "Enterobacterales",
            help = "Target taxon. Must be either \"Enterococcus\" or \
\"Enterobacterales\". Defaults to \"%(default)s\". "
        )

        self.parser.add_argument(
            "--niter", "-n",
            type = int, default = 1000,
            help = "Number of bootstrapping iterations. Defaults to %(default)s."
        )

class DetectionPlasmidSizeParser(CLIParser):

    def __init__(self):
        super().__init__(
            "Plasmid detection performance as a function of plasmid size",
            description="Compares the plasmid detection metrics for plasmids \
of different sizes (large vs. small)."
        )

    def add_arguments(self, **kwargs):

        self.parser.add_argument(
            "--input", "-i",
            type = str,
            default = "data/predictions.xlsx",
            help = "Table containing predictions. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--output", "-o",
            type = str,
            default = "outputs/detection/detection.plasmid_size",
            help = "Output directory. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--taxon", "-t",
            type = str, choices = ["Enterococcus", "Enterobacterales"],
            default = "Enterobacterales",
            help = "Target taxon. Must be either \"Enterococcus\" or \
\"Enterobacterales\". Defaults to \"%(default)s\". "
        )

        self.parser.add_argument(
            "--niter", "-n",
            type = int, default = 1000,
            help = "Number of bootstrapping iterations. Defaults to %(default)s."
        )

class ReconstructionMetricsParser(CLIParser):

    def __init__(self):
        super().__init__(
            "Plasmid reconstruction metrics",
            description="Computes the performance metrics for plasmid reconstruction \
tools given their predictions."
        )

    def add_arguments(self, **kwargs):

        self.parser.add_argument(
            "--input", "-i",
            type = str,
            default = "data/predictions.xlsx",
            help = "Table containing predictions. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--output", "-o",
            type = str,
            default = "outputs/reconstruction/reconstruction.metrics",
            help = "Output directory. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--niter", "-n",
            type = int, default = 1000,
            help = "Number of bootstrapping iterations. Defaults to %(default)s."
        )

class ReconstructionGLMParser(CLIParser):

    def __init__(self):
        super().__init__(
            "Linear regression (plasmid reconstruction)",
            description="Checks for characteristics impacting plasmid reconstruction by \
fitting a linear regression model to the NMI of each of the top four tools."
        )

    def add_arguments(self, **kwargs):

        self.parser.add_argument(
            "--input", "-i",
            type = str,
            default = "data/predictions.xlsx",
            help = "Table containing predictions. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--output", "-o",
            type = str,
            default = "outputs/reconstruction/reconstruction.glm",
            help = "Output directory. Defaults to %(default)s."
        )

class ReconstructionARGsParser(CLIParser):

    def __init__(self):
        super().__init__(
            "Plasmid reconstruction performance as a function of ARG presence",
            description="Compares the plasmid reconstruction metrics for plasmids with and \
without ARGs."
        )

    def add_arguments(self, **kwargs):

        self.parser.add_argument(
            "--input", "-i",
            type = str,
            default = "data/predictions.xlsx",
            help = "Table containing predictions. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--output", "-o",
            type = str,
            default = "outputs/reconstruction/reconstruction.args",
            help = "Output directory. Defaults to %(default)s."
        )

        self.parser.add_argument(
            "--taxon", "-t",
            type = str, choices = ["Enterococcus", "Enterobacterales"],
            default = "Enterococcus",
            help = "Target taxon. Must be either \"Enterococcus\" or \"Enterobacterales\". \
Defaults to \"%(default)s\". "
        )

        self.parser.add_argument(
            "--niter", "-n",
            type = int, default = 1000,
            help = "Number of bootstrapping iterations. Defaults to %(default)s."
        )