import h5py
import tqdm
import click
import gzip
import pandas as pd
import math
import os

@click.command()
@click.option(
    "--input-hdf5", "-i", nargs=1, required=True, help="HD5F file to convert"
)
@click.option(
    "--output-bed", "-o", nargs=1, required=True, help="Output BED file"
)
@click.option(
    "--gzip", "-z", "gzip_flag", is_flag=True, help="If specified, gzip the output"
)
def main(input_hdf5, output_bed, gzip_flag):
    os.makedirs(os.path.dirname(output_bed), exist_ok=True)
    if gzip_flag and not output_bed.endswith(".gz"):
        output_bed += ".gz"

    # Get size of table in total
    f = h5py.File(input_hdf5)
    total_size = f["data"]["table"].shape[0]
    f.close()

    # Iterate through
    chunk_size = 2048
    iterator = pd.read_hdf(input_hdf5, chunksize=chunk_size)
    
    for i, chunk in tqdm.tqdm(enumerate(iterator), total=math.ceil(total_size / chunk_size)):
        if i == 0:
            # Write header
            header = "\t".join(["chr", "start", "end"] + list(chunk)) + "\n"
            if gzip_flag:
                f = gzip.open(output_bed, "wb")
                f.write(header.encode())
            else:
                f = open(output_bed, "w")
                f.write(header)
            f.close()

        chunk = chunk.fillna(value=-1).astype(int)
        chunk.to_csv(
            output_bed, sep="\t", header=False, mode="a",
            compression="gzip" if gzip_flag else None
        )


if __name__ == "__main__":
    main()

