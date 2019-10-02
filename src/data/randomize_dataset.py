import pandas as pd
import numpy as np
from datetime import datetime
import click

def randomize_binary_dataset(input_path, output_path):
    """
    Takes a gzipped binary dataset at `input_path`, and randomizes the labels
    so that the overall composition of positives and negatives is identical.
    Stores the result as a gzipped table of identical format at `output_path`.
    """
    print("Reading in input BED file...", end=" ", flush=True)
    start = datetime.now()
    input_table = pd.read_csv(
        input_path, sep="\t", header=0, compression="gzip"
    )
    print(str((datetime.now() - start).seconds) + "s")

    values = input_table.iloc[:, 3:].values  # 3rd column and beyond

    print("Shuffling values...", end=" ", flush=True)
    start = datetime.now()
    values_flat = values.ravel()
    np.random.shuffle(values_flat)  # values is also shuffled
    values_shuf = values_flat.reshape(values.shape)
    print(str((datetime.now() - start).seconds) + "s")

    print("Setting the shuffled values...", end=" ", flush=True)
    start = datetime.now()
    input_table.iloc[:, 3:] = values_shuf
    print(str((datetime.now() - start).seconds) + "s")

    print("Writing the output BED file...", end=" ", flush=True)
    start = datetime.now()
    input_table.to_csv(
        output_path, sep="\t", header=True, index=False, compression="gzip"
    )
    print(str((datetime.now() - start).seconds) + "s")


@click.command()
@click.option(
    "--input-path", "-i", nargs=1, required=True, type=click.Path(),
    help="Path to gzipped BED to randomize"
)
@click.option(
    "--output-path", "-o", nargs=1, required=True, type=click.Path(),
    help="Output path for randomized gzipped BED"
)
def main(input_path, output_path):
    randomize_binary_dataset(input_path, output_path)

if __name__ == "__main__":
    main()
