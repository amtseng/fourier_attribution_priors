import click
import numpy as np

@click.command()
@click.option(
    "-t", "--thresh", default=5, help="Threshold for initial prediction"
)
@click.argument("jaspar_motif_file")
def main(thresh, jaspar_motif_file):
    motifs = []
    with open(jaspar_motif_file, "r") as f:
        while True:
            try:
                line = next(f)
            except StopIteration:
                break
            assert line.startswith(">")
            motif_name = line.strip().split("\t")[0][1:]
            base_counts = []
            for _ in range(4):
                line = next(f)
                tokens = line.strip().split()[2:-1]
                base_counts.append([int(x) for x in tokens])
            motifs.append((motif_name, np.transpose(np.array(base_counts))))

    for motif_name, counts in motifs:
        # Compute frequencies, with LaPlacian smoothing
        freqs = (counts + 1) / np.sum(counts + 1, axis=1, keepdims=True)
        print(">NULL\t%s\t%f" % (motif_name, thresh))
        for row in freqs.astype(str):
            print("\t".join(row))

if __name__ == "__main__":
    main()
