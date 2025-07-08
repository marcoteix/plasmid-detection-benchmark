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