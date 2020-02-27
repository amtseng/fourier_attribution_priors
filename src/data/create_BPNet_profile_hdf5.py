import os
import h5py
import pyBigWig
import numpy as np
import tqdm
import click

def fetch_bigwig_paths(base_path):
    """
    Reads in the set of BigWig paths corresponding to ChIPseq profiles.
    Arguments:
        `base_path`: path containing the BigWig profiles
    Returns a list of pairs, where each pair is the BigWig tracks for the
    negative and positive strands (in that order). The first T pairs of this
    list-of-pairs is for the TF-ChIPseq profiles, and the last pair is the
    control profiles. The BigWigs are ordered such that the TF names are in
    sorted order.
    """
    bigwig_list = [
        item for item in os.listdir(base_path) if item.endswith(".bw")
    ]

    # Read in names, organizing into
    # tf_name/control : strand
    bigwig_dict = {}
    for name in bigwig_list:
        tokens = name[:-3].split("_")
        assert len(tokens) == 3, \
            "Found BigWig of improperly formatted name: %s" % name
        stem, tf_name, strand = tokens
        assert stem == "BPNet", \
            "Found BigWig of improperly formatted name: %s" % name
        assert strand in ("neg", "pos"), "Found BigWig of strand other than pos/neg"
    
        if tf_name not in bigwig_dict:
            bigwig_dict[tf_name] = {}
        assert strand not in bigwig_dict[tf_name], \
            "Found duplicate BigWig for %s, %s" % (tf_name, strand)
        bigwig_dict[tf_name][strand] = os.path.join(base_path, name)

    assert "control" in bigwig_dict, "Did not find control"
    for strand_dict in bigwig_dict.values():
        assert sorted(strand_dict.keys()) == ["neg", "pos"], \
            "Did not find both strands for %s" % tf_name

    # Reformat dictionary into list of pairs
    paths = []
    for tf_name in sorted(bigwig_dict.keys()):
        if tf_name == "control":
            continue  # Leave for last
        paths.append((bigwig_dict[tf_name]["neg"], bigwig_dict[tf_name]["pos"]))
    paths.append((bigwig_dict["control"]["neg"], bigwig_dict["control"]["pos"]))
    return paths


def create_hdf5(
    bigwig_paths, chrom_sizes_path, out_path, chunk_size, batch_size=100
):
    """
    Creates an HDF5 file containing all BigWig tracks.
    Arguments:
        `bigwig_paths`: a list of pairs of paths, as returned by
            `fetch_bigwig_paths`
        `chrom_sizes_path`: path to canonical chromosome sizes
        `out_path`: where to write the HDF5
        `chunk_size`: chunk size to use in HDF5 along the chromosome size
            dimension; this is recommended to be the expected size of the
            queries made
        `batch_size`: number of chunks to write at a time
    This creates an HDF5 file, containing a dataset for each chromosome. Each
    dataset will be a large array of shape L x 2T x 2, where L is the length of
    the chromosome, T is the number of tasks (i.e. T experiment/cell lines, one
    for each TF and one for matched control), 2 is for both strands. The HDF5
    will also contain a dataset which has the paths to the corresponding source
    BigWigs, stored as a 2T x 2 array of paths.
    """
    bigwig_readers = [
        [pyBigWig.open(path1), pyBigWig.open(path2)]
        for path1, path2 in bigwig_paths
    ]
   
    # Read in chromosome sizes
    with open(chrom_sizes_path, "r") as f:
        chrom_sizes = {}
        for line in f:
            tokens = line.strip().split("\t")
            chrom_sizes[tokens[0]] = int(tokens[1])
   
    # Convert batch size to be in terms of rows, not number of chunks
    batch_size = batch_size * chunk_size

    with h5py.File(out_path, "w") as f:
        # Store source paths
        f.create_dataset("bigwig_paths", data=np.array(bigwig_paths, dtype="S"))
        for chrom in sorted(chrom_sizes.keys()):
            chrom_size = chrom_sizes[chrom]
            num_batches = int(np.ceil(chrom_size / batch_size))
            chrom_dset = f.create_dataset(
                chrom, (chrom_size, len(bigwig_paths), 2), dtype="f",
                compression="gzip", chunks=(chunk_size, len(bigwig_paths), 2)
            )
            for i in tqdm.trange(num_batches, desc=chrom):
                start = i * batch_size
                end = min(chrom_size, (i + 1) * batch_size)

                values = np.stack([
                    np.stack([
                        np.nan_to_num(reader1.values(chrom, start, end)),
                        np.nan_to_num(reader2.values(chrom, start, end))
                    ], axis=1) for reader1, reader2 in bigwig_readers
                ], axis=1)

                chrom_dset[start : end] = values

@click.command()
@click.option(
    "--base-path", "-b", default=None,
    help="Path to directory containing BigWigs; defaults to /users/amtseng/att_priors/data/raw/BPNet_ChIPseq/"
)
@click.option(
    "--chrom-sizes-path", "-c",
    default="/users/amtseng/genomes/mm10.canon.chrom.sizes",
    help="Path to canonical chromosome sizes"
)
@click.option(
    "--out-path", "-o", default=None,
    help="Destination for new HDF5; defaults to /users/amtseng/att_priors/data/processed/BPNet_ChIPseq/profile/labels/BPNet_profiles.h5"
)
@click.option(
    "--chunk-size", "-s", default=1500,
    help="Chunk size along chromosome length dimension for HDF5"
)
def main(base_path, chrom_sizes_path, out_path, chunk_size):
    """
    Converts a set of BPNet ChIPseq and one pair of control ChIPseq profile
    BigWigs into an HDF5 file. HDF5 has separate datasets for each chromosome.
    Each chromosome's dataset is stored as an L x (T + 1) x 2 array, where L is
    the size of the chromosome, T is the number of TFs (i.e. tasks), and 2 is
    for each strand. The dataset under the key `bigwig_paths` stores the paths
    for the source BigWigs.
    """
    if not base_path: 
        base_path = "/users/amtseng/att_priors/data/raw/BPNet_ChIPseq/"
    if not out_path:
        out_path = \
            "/users/amtseng/att_priors/data/processed/BPNet_ChIPseq/profile/labels/BPNet_profiles.h5"

    bigwig_paths = fetch_bigwig_paths(base_path)
    print("Found %d TFs/tasks" % (len(bigwig_paths) - 1))
    os.makedirs(os.path.dirname(out_path))
    create_hdf5(bigwig_paths, chrom_sizes_path, out_path, chunk_size)


if __name__ == "__main__":
    main()
