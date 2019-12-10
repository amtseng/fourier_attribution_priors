import gzip
import click
import sys

@click.command()
@click.option(
    "--skiprows", "-n", default=0,
    help="Skip checking the first n rows of the BED file (default 0)"
)
@click.option(
    "--gzipped", "-z", is_flag=True,
    help="The BED file is gzipped (default False)"
)
@click.argument("file_path", nargs=1, type=click.Path())
@click.argument("chrom_sizes_path", nargs=1, type=click.Path())
def main(skiprows, gzipped, file_path, chrom_sizes_path):
    if gzipped:
        infile = gzip.open(file_path, "rt")
    else:
        infile = open(file_path, "r")

    with open(chrom_sizes_path, "r") as chromfile:
        chroms = set([
            line.strip().split("\t")[0].lower() for line in chromfile
        ])

    for _ in range(skiprows):
        sys.stdout.write(next(infile))

    for line in infile:
        if line.strip().split("\t")[0].lower() in chroms:
            sys.stdout.write(line)

    infile.close()

if __name__ == "__main__":
    main()
