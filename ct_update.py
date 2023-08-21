import click
import json
from pathlib import Path

# local
from utils import *

@click.command()
@click.option("--input", required=True, help="input arguments in json for CT update")
def ct_update(input):

    with open(input) as f:
        args = json.load(f)
    
    preop = args["preop_data"]
    intraop = args["intraop_data"]

    breakpoint()

    return



if __name__ == "__main__":
    ct_update()