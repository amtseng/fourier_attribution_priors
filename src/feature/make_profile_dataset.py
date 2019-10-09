import torch
import numpy as np
import pandas as pd
import sacred
from datetime import datetime
import pyBigWig
import util

dataset_ex = sacred.Experiment("dataset")

@dataset_ex.config
def config():
    # Path to reference genome FASTA
    reference_fasta = "/users/amtseng/genomes/hg38.fasta"

    # Path to chromosome sizes
    chrom_sizes = "/users/amtseng/genomes/hg38.canon.chrom.sizes"

    # For each input sequence in the raw data, center it and pad to this length 
    input_length = 1000

    # One-hot encoding has this depth
    input_depth = 4

    # Whether or not to perform reverse complement augmentation
    revcomp = True

    # Maximum size of jitter to the input for augmentation; set to 0 to disable
    jitter_size = 50

    # Batch size; will be multiplied by two if reverse complementation is done
    batch_size = 128

    # Sample X negatives randomly from the genome for every positive example
    negative_ratio = 20

    # Number of workers for the data loader
    num_workers = 10

    # Dataset seed (for shuffling)
    dataset_seed = None


class GenomeIntervalSampler:
    """
    Samples a random interval from the genome. The sampling is performed
    uniformly at random (i.e. longer chromosomes are more likely to be sampled
    from).
    Arguments:
        `chrom_sizes`: Path to 2-column TSV listing sizes of each chromosome
        `sample_length`: Length of sampled sequence
        `chroms_keep`: An iterable of chromosomes that specifies which
            chromosomes to keep from the sizes; sampling will only occur from
            these chromosomes
    """
    def __init__(self, chrom_sizes, sample_length, chroms_keep=None, seed=None):
        self.sample_length = sample_length
       
        # Create DataFrame of chromosome sizes
        chrom_table = pd.read_csv(
            chrom_sizes, sep="\t", header=None, names=["chrom", "size"]
        )
        if chroms_keep:
            chrom_table = chrom_table[chrom_table["chrom"].isin(chroms_keep)]
        chrom_table["size"] -= sample_length  # Cut off sizes to avoid overrun
        chrom_table["weight"] = chrom_table["size"] / chrom_table["size"].sum()
        self.chrom_table = chrom_table

        if seed:
            np.random.seed(seed)

    def sample_intervals(self, num_intervals):
        """
        Returns a 2D NumPy array of randomly sampled coordinates. Returns
        `num_intervals` intervals, uniformly randomly sampled from the genome.
        """
        chrom_sample = self.chrom_table.sample(
            n=num_intervals,
            replace=True,
            weights=self.chrom_table["weight"]
        )
        chrom_sample["start"] = \
            (np.random.rand(num_intervals) * chrom_sample["size"]).astype(int)
        chrom_sample["end"] = chrom_sample["start"] + self.sample_length

        return chrom_sample[["chrom", "start", "end"]].values.astype(object)


class CoordsToVals:
    """
    From a single BigWig file mapping genomic coordinates to profiles, this
    creates an object that maps a list of coordinates to a NumPy array of
    profiles.
    Arguments:
        `bigwig_path`: Path to BigWig containing profile
    """
    def __init__(self, bigwig_path):
        self.bigwig_path = bigwig_path

    def _get_ndarray(self, coords):
        """
        From an iterable of coordinates, retrieves a 2D NumPy array of
        corresponding profile values. Note that all coordinate intervals need
        to be of the same length. 
        """
        reader = pyBigWig.open(self.bigwig_path)
        return np.array(
            [np.nan_to_num(reader.values(*coord)) for coord in coords]
        )

    def __call__(self, coords):
        return self._get_ndarray(coords)


class CoordsBatcher(torch.utils.data.sampler.Sampler):
    """
    Creates a batch producer that batches positive coordinates and samples
    negative coordinates.
    Arguments:
        `pos_coords_bed`: Path to gzipped BED file containing the set of
            positive coordinates
        `batch_size`: Number of samples per batch
        `neg_ratio`: Number of negatives to select for each positive example
        `jitter`: Random amount to jitter each positive coordinate example by
        `genome_sampler`: A GenomeIntervalSampler instance, which samples
            intervals randomly from the genome
        `shuffle_before_epoch`: Whether or not to shuffle all examples before
            each epoch
    """
    def __init__(
        self, pos_coords_bed, batch_size, neg_ratio, jitter, genome_sampler,
        shuffle_before_epoch=False, seed=None
    ):
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.jitter = jitter
        self.genome_sampler = genome_sampler
        self.shuffle_before_epoch = shuffle_before_epoch

        # Read in the positive coordinates
        pos_coords_table = pd.read_csv(
            pos_coords_bed, sep="\t", header=None, compression="gzip"
        )
        self.pos_coords = pos_coords_table.values.astype(object)

        # Number of positives and negatives per batch
        self.num_pos = len(self.pos_coords)
        self.neg_per_batch = int(batch_size * neg_ratio / (neg_ratio + 1))
        self.pos_per_batch = batch_size - self.neg_per_batch

        if shuffle_before_epoch:
            self.rng = np.random.RandomState(seed)

    def __getitem__(self, index):
        """
        Fetches a full batch of positive and negative coordinates by filling the
        batch with some positive coordinates, and sampling randomly from the
        rest of the genome for the negatives. Returns a 2D NumPy array of
        coordinates, along with a parallel NumPy array of binary status.
        """
        pos_coords = self.pos_coords[
            index * self.pos_per_batch : (index + 1) * self.pos_per_batch
        ]

        # If specified, apply random jitter to each positive coordinate
        if self.jitter:
            jitter_vals = np.random.randint(
                -self.jitter, self.jitter + 1, size=len(pos_coords)
            )
            pos_coords[:,1] += jitter_vals
            pos_coords[:,2] += jitter_vals

        neg_coords = self.genome_sampler.sample_intervals(self.neg_per_batch)
        status = np.concatenate(
            [np.ones(len(pos_coords)), np.zeros(len(neg_coords))]
        )
        return np.concatenate([pos_coords, neg_coords]), status

    def __len__(self):
        return int(np.ceil(self.num_pos / float(self.pos_per_batch)))
   
    def on_epoch_start(self):
        if self.shuffle_before_epoch:
            self.pos_coords = self.rng.permutation(self.pos_coords)


class CoordDataset(torch.utils.data.IterableDataset):
    """
    Generates single samples of a one-hot encoded sequence and value.
    Arguments:
        `coords_batcher (CoordsDownsampler): Maps indices to batches of
            coordinates (split into positive and negative binding)
        `coords_to_seq (CoordsToSeq)`: Maps coordinates to 1-hot encoded
            sequences
        `coords_to_vals (CoordsToVals)`: Maps coordinates to profiles to predict
        `revcomp`: Whether or not to perform revcomp to the batch; this will
            double the batch size implicitly
        `return_coords`: If True, each batch returns the set of coordinates for
            that batch along with the 1-hot encoded sequences and values
    """
    def __init__(
        self, coords_batcher, coords_to_seq, coords_to_vals, revcomp=False,
        return_coords=False
    ):
        self.coords_batcher = coords_batcher
        self.coords_to_seq = coords_to_seq
        self.coords_to_vals = coords_to_vals
        self.revcomp = revcomp
        self.return_coords = return_coords

    def get_batch(self, index):
        """
        Returns a batch, which consists of a 2D NumPy array of 1-hot encoded
        sequence, a 2D NumPy array of profiles, a 1D NumPy array of read counts,
        and a 1D NumPy array of binding status (1 or 0).
        """
        # Get batch of coordinates for this index
        coords, status = self.coords_batcher[index]

        # Map this batch of coordinates to 1-hot encoded sequences
        seqs = self.coords_to_seq(coords, revcomp=self.revcomp)

        # Map this batch of coordinates to the associated profiles
        profiles = self.coords_to_vals(coords)
        counts = np.sum(profiles, axis=1)

        # If reverse complementation was done, double sizes of everything else
        if self.revcomp:
            profiles = np.concatenate([profiles, profiles])
            counts = np.concatenate([counts, counts])
            status = np.concatenate([status, status])

        if self.return_coords:
            if self.revcomp:
                coords_ret = np.concatenate([coords.values, coords.values])
            return coords_ret, seqs, profiles, counts, status
        else:
            return seqs, profiles, counts, status

    def __iter__(self):
        """
        Returns an iterator over the batches. If the dataset iterator is called
        from multiple workers, each worker will be give a shard of the full
        range.
        """
        worker_info = torch.utils.data.get_worker_info()
        num_batches = len(self.coords_batcher)
        if worker_info is None:
            # In single-processing mode
            start, end = 0, num_batches
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shard_size = int(np.ceil(num_batches / num_workers))
            start = shard_size * worker_id
            end = min(start + shard_size, num_batches)
        return (self.get_batch(i) for i in range(start, end))

    def __len__(self):
        return len(self.coords_batcher)
    
    def on_epoch_start(self):
        """
        This should be called manually before the beginning of every epoch (i.e.
        before the iteration begins).
        """
        self.coords_batcher.on_epoch_start()


@dataset_ex.capture
def data_loader_from_bedfile(
    peaks_bed_path, profile_bigwig_path, batch_size, reference_fasta,
    chrom_sizes, input_length, negative_ratio, num_workers, revcomp,
    jitter_size, dataset_seed, shuffle=True, return_coords=False
):
    """
    From the path to a gzipped BED file containing coordinates of positive
    peaks and the path to a BigWig containing reads mapped to each location in
    the gnome, returns an IterableDataset object. If `shuffle` is True, shuffle
    the dataset before each epoch.
    """
    # Maps set of coordinates to profiles
    coords_to_vals = CoordsToVals(profile_bigwig_path)

    # Randomly samples from genome
    genome_sampler = GenomeIntervalSampler(
        chrom_sizes, input_length, seed=dataset_seed
    )
    
    # Coordinate batcher, yielding batches of positive and negative coordinates
    coords_batcher = CoordsBatcher(
        peaks_bed_path, batch_size, negative_ratio, jitter_size, genome_sampler,
        shuffle_before_epoch=shuffle, seed=dataset_seed
    )

    # Maps set of coordinates to 1-hot encoding, padded
    coords_to_seq = util.CoordsToSeq(
        reference_fasta, center_size_to_use=input_length
    )

    # Dataset
    dataset = CoordDataset(
        coords_batcher, coords_to_seq, coords_to_vals, revcomp=revcomp,
        return_coords=return_coords
    )

    # Dataset loader: dataset is iterable and already returns batches
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=num_workers,
        collate_fn=lambda x: x
    )

    return loader


data = None
loader = None
@dataset_ex.automain
def main():
    global data, loader
    import os
    import tqdm

    base_path = "/users/amtseng/att_priors/data/processed/ENCODE/profile/labels"

    peaks_bed_file = os.path.join(
        base_path,
        "SPI1/SPI1_ENCSR000BGW_K562_holdout_peakints.bed.gz"
    )
    profile_bigwig_file = os.path.join(
        base_path,
        "SPI1/SPI1_ENCSR000BGW_K562_pos.bw"
    )

    loader = data_loader_from_bedfile(
        peaks_bed_file, profile_bigwig_file,
        reference_fasta="/users/amtseng/genomes/hg38.fasta"
    )
    loader.dataset.on_epoch_start()
    start_time = datetime.now()
    for batch in tqdm.tqdm(loader, total=len(loader.dataset)):
        data = batch
    end_time = datetime.now()
    print("Time: %ds" % (end_time - start_time).seconds)
