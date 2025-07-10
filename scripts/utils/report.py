from pathlib import Path
from argparse import Namespace

def create_report(
    outdir: str,
    args: Namespace,
    program_name: str
):
    
    filepath = Path(outdir).joinpath("report.log")

    file = open(filepath, "w")

    print(
        f":::::::::::::::: {program_name} ::::::::::::::::",
        *["\t-" + k + ": " + str(v) for k,v in vars(args).items()],
        sep = "\n",
        file = file
    )

    return file