import os
import h5py
import pyBigWig
import numpy as np
import tqdm
import click

def fetch_bigwig_paths(base_path, cell_type):
    """
    Reads in the set of BigWig paths corresponding to DNase-seq profiles.
    These BigWigs should be downloaded by `download_ENCODE_DNase_data.py`. This
    function performs error-checking to ensure that the proper files are there
    under `base_path`.
    Arguments:
        `base_path`: path containing the BigWig profiles
        `cell_type`: name of the cell type (i.e. the profiles should start with
            this name)
    Returns a list of pairs, where each pair is the BigWig tracks for the
    negative and positive strands (in that order). The BigWigs are ordered such
    that the DNase-seq experiment IDs are in sorted order.
    """
    bigwig_list = [
        item for item in os.listdir(base_path) if item.endswith(".bw")
    ]

    # Read in names, grouping into negative/positive pairs
    bigwig_dict = {}
    for name in bigwig_list:
        tokens = name[:-3].split("_")
        assert len(tokens) == 3, \
            "Found BigWig of improperly formatted name: %s" % name
        
        cond, expid, strand = tokens
        assert cond == cell_type, "Found BigWig not of cell type %s" % cell_type
        assert strand in ("neg", "pos"), \
            "Found BigWig of strand other than pos/neg"
    
        if expid not in bigwig_dict:
            bigwig_dict[expid] = {}
        assert strand not in bigwig_dict[expid], \
            "Found duplicate BigWig for %s, %s" % (expid, strand)
        bigwig_dict[expid][strand] = os.path.join(base_path, name)

    for expid, expid_dict in bigwig_dict.items():
        assert sorted(expid_dict.keys()) == ["neg", "pos"], \
            "Did not find both strands for %s" % expid

    # Reformat dictionary into list of pairs
    paths = []
    for expid in sorted(bigwig_dict.keys()):
        strand_dict = bigwig_dict[expid]
        paths.append([strand_dict["neg"], strand_dict["pos"]])

    return paths


def fetch_control_bigwig_paths(control_path):
    """
    Reads in a pair of BigWig paths corresponding to DNase-seq bias tracks.
    These BigWigs should be 5-prime count tracks, and be named like:
        {stem}_neg.bw {stem}_pos.bw
    The given path should only have two such files named like this.
    Arguments:
        `control_path`: path containing the two BigWig profiles
    Returns a single pair of paths to BigWig tracks for the negative and
    positive strands (in that order).
    """
    neg_paths = [
        item for item in os.listdir(control_path) if item.endswith("_neg.bw")
    ]
    assert len(neg_paths) == 1
    neg_path = neg_paths[0]
    pos_paths = [
        item for item in os.listdir(control_path) if item.endswith("_pos.bw")
    ]
    assert len(pos_paths) == 1
    pos_path = pos_paths[0]
    assert pos_path[:-7] == neg_path[:-7]  # Same stem
    return (
        os.path.join(control_path, neg_path),
        os.path.join(control_path, pos_path)
    )


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
    dataset will be a large array of shape L x T x 2, where L is the length of
    the chromosome, T is the number of tasks (i.e. T experiments/conditions), 2
    is for both strands. The HDF5 will also contain a dataset which has the
    paths to the corresponding source BigWigs, stored as a T x 2 array of paths.
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
    "--cell-type", "-t", required=True,
    help="Name of cell type, needs to match prefix of DNase-seq BigWig files"
)
@click.option(
    "--base-path", "-b", default=None,
    help="Path to directory containing BigWigs; defaults to /users/amtseng/att_priors/data/interim/ENCODE_DNase/profile/{cell_type}/"
)
@click.option(
    "--control-path", "-r", default=None,
    help="Path to directory containing control BigWigs; defaults to /users/amtseng/att_priors/data/raw/DNase_bias/"
)
@click.option(
    "--chrom-sizes-path", "-c",
    default="/users/amtseng/genomes/hg38.canon.chrom.sizes",
    help="Path to canonical chromosome sizes"
)
@click.option(
    "--out-path", "-o", default=None,
    help="Destination for new HDF5; defaults to /users/amtseng/att_priors/data/processed/ENCODE_DNase/profile/labels/{cell_type}/{cell_type}_profiles.h5"
)
@click.option(
    "--chunk-size", "-s", default=1500,
    help="Chunk size along chromosome length dimension for HDF5"
)
def main(
    cell_type, base_path, control_path, chrom_sizes_path, out_path, chunk_size
):
    """
    Converts DNase-seq profile BigWigs into an HDF5 file. The HDF5 has separate
    datasets for each chromosome. Each chromosome's dataset is stored as an
    L x (T + 1) x 2 array, where L is the size of the chromosome, T is the
    number of experiments (i.e. tasks), and 2 is for each strand. The last pair
    of tracks is the control track. The dataset under the key `bigwig_paths`
    stores the paths for the source BigWigs.
    """
    if not base_path: 
        base_path = "/users/amtseng/att_priors/data/interim/ENCODE_DNase/profile/%s" % cell_type
    if not control_path:
        control_path = "/users/amtseng/att_priors/data/raw/DNase_bias/"
    if not out_path:
        out_path = \
            "/users/amtseng/att_priors/data/processed/ENCODE_DNase/profile/labels/%s/%s_profiles.h5" % (cell_type, cell_type)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    bigwig_paths = fetch_bigwig_paths(base_path, cell_type)
    control_bigwig_paths = fetch_control_bigwig_paths(control_path)

    print(bigwig_paths)
    print(control_bigwig_paths)

    # Tack on the pair of control BigWigs at the end
    bigwig_paths.append(control_bigwig_paths)

    print("Found %d experiments and 1 control" % len(bigwig_paths))
    create_hdf5(bigwig_paths, chrom_sizes_path, out_path, chunk_size)


if __name__ == "__main__":
    main()
