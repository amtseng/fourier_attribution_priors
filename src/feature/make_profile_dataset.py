import torch
import numpy as np
import pandas as pd
import sacred
from datetime import datetime
import h5py
import feature.util as util

dataset_ex = sacred.Experiment("dataset")

@dataset_ex.config
def config():
    # Path to reference genome FASTA
    reference_fasta = "/users/amtseng/genomes/hg38.fasta"

    # Path to chromosome sizes
    chrom_sizes_tsv = "/users/amtseng/genomes/hg38.canon.chrom.sizes"

    # The size of DNA sequences to fetch as input sequences
    input_length = 1346

    # The size of profiles to fetch for each coordinate
    profile_length = 1000

    # One-hot encoding has this depth
    input_depth = 4

    # Whether or not to perform reverse complement augmentation
    revcomp = True

    # Maximum size of jitter to the input for augmentation; set to 0 to disable
    jitter_size = 128

    # Batch size; will be multiplied by two if reverse complementation is done
    batch_size = 64

    # Sample X negatives randomly from the genome for every positive example
    negative_ratio = 1

    # Use this stride when tiling coordinates across a peak
    peak_tiling_stride = 25

    # Amount of dataset for each task to keep; can be a set number of peaks, or
    # a fraction (if < 1); set to None to keep everything
    peak_retention = None

    # Number of workers for the data loader
    num_workers = 10

    # Negative seed (for selecting negatives)
    negative_seed = None

    # Jitter seed (for applying random jitter to peaks)
    jitter_seed = None

    # Shuffle seed (for shuffling data points)
    shuffle_seed = None


class GenomeIntervalSampler:
    """
    Samples a random interval from the genome. The sampling is performed
    uniformly at random (i.e. longer chromosomes are more likely to be sampled
    from).
    Arguments:
        `chrom_sizes_tsv`: path to 2-column TSV listing sizes of each chromosome
        `sample_length`: length of sampled sequence
        `chroms_keep`: an iterable of chromosomes that specifies which
            chromosomes to keep from the sizes; sampling will only occur from
            these chromosomes
    """
    def __init__(
        self, chrom_sizes_tsv, sample_length, chroms_keep=None, seed=None
    ):
        self.sample_length = sample_length
       
        # Create DataFrame of chromosome sizes
        chrom_table = self._import_chrom_sizes(chrom_sizes_tsv)
        if chroms_keep:
            chrom_table = chrom_table[chrom_table["chrom"].isin(chroms_keep)]
        # Cut off sizes to avoid overrunning ends of chromosome
        chrom_table["max_size"] -= sample_length
        chrom_table["weight"] = \
            chrom_table["max_size"] / chrom_table["max_size"].sum()
        self.chrom_table = chrom_table

        self.rng = np.random.RandomState(seed)

    def _import_chrom_sizes(self, chrom_sizes_tsv):
        """
        Imports a TSV of chromosome sizes, mapping chromosome to maximum size.
        Arguments:
            `chrom_sizes_tsv`: a 2-column TSV mapping chromosome name to size
        Returns a Pandas DataFrame
        """
        return pd.read_csv(
            chrom_sizes_tsv, sep="\t", header=None, names=["chrom", "max_size"]
        )

    def sample_intervals(self, num_intervals):
        """
        Returns a 2D NumPy array of randomly sampled coordinates. Returns
        `num_intervals` intervals, uniformly randomly sampled from the genome.
        """
        chrom_sample = self.chrom_table.sample(
            n=num_intervals,
            replace=True,
            weights=self.chrom_table["weight"],
            random_state=self.rng
        )
        chrom_sample["start"] = (
            self.rng.rand(num_intervals) * chrom_sample["max_size"]
        ).astype(int)
        chrom_sample["end"] = chrom_sample["start"] + self.sample_length

        return chrom_sample[["chrom", "start", "end"]].values.astype(object)


class CoordsToVals:
    """
    From an HDF5 file that maps genomic coordinates to profiles, this creates an
    object that maps a list of coordinates to a NumPy array of profiles.
    Arguments:
        `hdf5_path`: path to HDF5 containing profiles; this HDF5 must have a
            separate dataset for each chromosome, and it is expected to return
            profiles of shape L x S x 2, where L is the coordinate dimension,
            S is the number of profile tracks, and 2 is for each strand
        `profile_size`: for each genomic coordinate, center it and pad it on
            both sides to this length to get the final profile; if this is
            smaller than the coordinate interval given, then the interval will
            be cut to this size by centering
    """
    def __init__(self, hdf5_path, profile_size):
        self.hdf5_path = hdf5_path
        self.profile_size = profile_size

    def _resize_interval(start, end, size):
        """
        Resizes the interval by centering and trimming/padding to the given
        size.
        """
        center = int(0.5 * (start + end))
        half_size = int(0.5 * size)
        left = center - half_size
        right = left + size
        return left, right

    def _get_profile(self, chrom, start, end, hdf5_reader):
        """
        Fetches the profile for the given coordinates, with an instantiated
        HDF5 reader. Returns the profile as a NumPy array of numbers. This may
        pad or cut from the center to a specified length.
        """
        if self.profile_size:
            start, end = CoordsToVals._resize_interval(
                start, end, self.profile_size
            )
        return hdf5_reader[chrom][start:end]

    def _get_ndarray(self, coords):
        """
        From an iterable of coordinates, retrieves a values for that coordinate.
        This will be a 4D NumPy array of corresponding profile values.
        Note that all coordinate intervals need to be of the same length (after
        padding).
        """
        with h5py.File(self.hdf5_path, "r") as reader:
            profiles = np.stack([
                self._get_profile(chrom, start, end, reader) \
                for chrom, start, end in coords
            ])
        return profiles

    def __call__(self, coords):
        return self._get_ndarray(coords)


class SamplingCoordsBatcher(torch.utils.data.sampler.Sampler):
    """
    Creates a batch producer that batches positive coordinates and samples
    negative coordinates. Each batch will have some positives and negatives
    according to `neg_ratio`. When multiple sets of positive coordinates are
    given, the coordinates are all pooled together and drawn from uniformly.
    Arguments:
        `pos_coords_beds`: list of paths to gzipped BED files containing the
            sets of positive coordinates for various tasks; these BED files
            should be in ENCODE NarrowPeak format
        `batch_size`: number of samples per batch
        `neg_ratio`: number of negatives to select for each positive example
        `jitter`: maximum random amount to jitter each positive coordinate
            example by
        `chrom_sizes_tsv`: path to 2-column TSV listing sizes of each chromosome
        `sample_length`: length of sampled sequence
        `genome_sampler`: a GenomeIntervalSampler instance, which samples
            intervals randomly from the genome
        `chroms_keep`: if specified, only considers this set of chromosomes from
            the coordinate BEDs
        `peak_retention`: if specified, keeps only this amount of peaks (taking
            most confident peaks preferentially); can be a fraction of the
            original BED file (if value is < 1), or a number of peaks (if value
            is >= 1)
        `return_peaks`: if True, returns the peaks and summits sampled from the
            peak set as a B x 3 array
        `shuffle_before_epoch`: Whether or not to shuffle all examples before
            each epoch
    """
    def __init__(
        self, pos_coords_beds, batch_size, neg_ratio, jitter, chrom_sizes_tsv,
        sample_length, genome_sampler, chroms_keep=None, peak_retention=None,
        return_peaks=False, shuffle_before_epoch=False, jitter_seed=None,
        shuffle_seed=None
    ):
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.jitter = jitter
        self.genome_sampler = genome_sampler
        self.return_peaks = return_peaks
        self.shuffle_before_epoch = shuffle_before_epoch

        chrom_sizes_table = self._import_chrom_sizes(chrom_sizes_tsv)

        all_pos_table = []
        for i, pos_coords_bed in enumerate(pos_coords_beds):
            peaks_table = self._import_peaks(pos_coords_bed)
            coords = self._format_peaks_table(
                peaks_table, chroms_keep, peak_retention, sample_length, jitter,
                chrom_sizes_table
            )
            
            # Add in the status column
            coords = np.concatenate(
                [coords, np.tile(i + 1, (len(coords), 1))], axis=1
            )  # Shape: _ x 7
            
            all_pos_table.append(coords)

        self.all_pos_table = np.concatenate(all_pos_table)  # Shape: N x 7
        self.num_total_pos = len(self.all_pos_table)

        # Number of positives and negatives per batch
        self.neg_per_batch = int(batch_size * neg_ratio / (neg_ratio + 1))
        self.pos_per_batch = batch_size - self.neg_per_batch

        if shuffle_before_epoch:
            self.shuffle_rng = np.random.RandomState(shuffle_seed)

        if self.jitter:
            self.jitter_rng = np.random.RandomState(jitter_seed)

    def _import_peaks(self, peaks_bed):
        """
        Imports a peaks BED file in NarrowPeak format as a Pandas DataFrame.
        Arguments:
            `peaks_bed`: a BED file (gzipped or not) containing peaks in
                ENCODE NarrowPeak format
        Returns a Pandas DataFrame
        """
        return pd.read_csv(
            peaks_bed, sep="\t", header=None,  # Infer compression
            names=[
                "chrom", "peak_start", "peak_end", "name", "score",
                "strand", "signal", "pval", "qval", "summit_offset"
            ]
        )

    def _import_chrom_sizes(self, chrom_sizes_tsv):
        """
        Imports a TSV of chromosome sizes, mapping chromosome to maximum size.
        Arguments:
            `chrom_sizes_tsv`: a 2-column TSV mapping chromosome name to size
        Returns a Pandas DataFrame
        """
        return pd.read_csv(
            chrom_sizes_tsv, sep="\t", header=None, names=["chrom", "max_size"]
        )

    def _format_peaks_table(
        self, peaks_table, chroms_keep, peak_retention, sample_length, jitter,
        chrom_sizes_table
    ):
        """
        Takes a table of imported peaks and formats it. This function performs
        the following tasks:
        1. Optionally filters peaks to only retain a subset of chromosomes
        2. Optionally cuts down the set of peaks to a subset of high-confidence
            peaks
        3. Computes the intervals for inputs being centered around the summits
        4. Drops any intervals that would overrun chromosome boundaries
        Arguments:
            `peaks_table`: a table imported by `_import_peaks`
            `chrom_sizes_table`: a table imported by `_import_chrom_sizes`
        Returns an N x 6 array, where columns 1-3 are the coordinate of the
        input samples centered at summits, columns 4-5 are the original peak
        location, and column 6 is the summit location.
        """
        if chroms_keep:
            # Keep only chromosomes specified
            peaks_table = peaks_table[peaks_table["chrom"].isin(chroms_keep)]

        if peak_retention is not None:
            # Keep only a subset of the peaks in the table
            # Sort coordinates by confidence first
            peaks_table = peaks_table.sort_values(by="signal", ascending=False)

            keep_num = int(len(peaks_table) * peak_retention) if \
                peak_retention < 1 else peak_retention
            peaks_table = peaks_table.head(keep_num)

        # Expand the coordinates to be of size `sample_length`, centered
        # around the summits
        peaks_table["start"] = peaks_table["peak_start"] + \
            peaks_table["summit_offset"] - (sample_length // 2)
        peaks_table["end"] = peaks_table["start"] + sample_length

        # Toss out any coordinates that (with jittering) may go past
        # chromosome boundaries
        # Add in the maximum size column
        peaks_table = peaks_table.merge(chrom_sizes_table, on="chrom")
        # Keep only coordinates that won't overrun the ends
        left_mask = peaks_table["start"] - jitter >= 0
        right_mask = peaks_table["end"] + jitter < peaks_table["max_size"]
        peaks_table = peaks_table.loc[left_mask & right_mask]

        # Compute the summit location from offset and peak start
        peaks_table["summit"] = peaks_table["peak_start"] + \
            peaks_table["summit_offset"]

        # Extract the columns desired
        coords = peaks_table[[
            "chrom", "start", "end", "peak_start", "peak_end", "summit"
        ]].values.astype(object)

        return coords

    def __getitem__(self, index):
        """
        Fetches a full batch of positive and negative coordinates by filling the
        batch with some positive coordinates, and sampling randomly from the
        rest of the genome for the negatives. Returns a B x 3 2D NumPy array of
        coordinates, along with a parallel B-array of status. Status is 0
        for negatives, and [1, n] for positives, where the status is 1 plus the
        index of the coordinate BED file it came from.
        This method may also perform jittering for dataset augmentation.
        If `return_peaks` was specified at object creation-time, also return a
        B x 3 2D NumPy array containing the peak information for the original
        peaks, consisting of the peak boundaries and the summit location
        (respectively); for negative samples drawn from the GenomeSampler, these
        values are all -1.
        """
        pos_table = self.all_pos_table[
            index * self.pos_per_batch : (index + 1) * self.pos_per_batch
        ]
        pos_coords, pos_peaks, pos_statuses = \
            pos_table[:, :3], pos_table[:, 3:6], pos_table[:, 6]

        # If specified, apply random jitter to each positive coordinate
        if self.jitter:
            jitter_vals = self.jitter_rng.randint(
                -self.jitter, self.jitter + 1, size=len(pos_coords)
            )
            pos_coords[:, 1] += jitter_vals
            pos_coords[:, 2] += jitter_vals
       
        # Fetch the negatives to fill out the batch
        if self.neg_per_batch:
            neg_coords = self.genome_sampler.sample_intervals(
                self.neg_per_batch
            )
        else:
            neg_coords = np.empty(shape=(0, 3), dtype=object)

        # At this point, `pos_coords` and `neg_coords` are both _ x 3
        # Concatenate the coordinates together
        coords = np.concatenate([pos_coords, neg_coords], axis=0)

        # Concatenate the statuses together; status for negatives is just 0
        status = np.concatenate(
            [pos_statuses, np.zeros(len(neg_coords))]  # Col 7
        ).astype(int)

        if self.return_peaks:
            # Concatenate the peaks together; peaks for negatives is all -1
            neg_peaks = np.full((len(neg_coords), 3), -1)
            peaks = np.concatenate([pos_peaks, neg_peaks])
            return coords, status, peaks
        else:
            return coords, status

    def __len__(self):
        return int(np.ceil(self.num_total_pos / float(self.pos_per_batch)))
   
    def on_epoch_start(self):
        if self.shuffle_before_epoch:
            perm = self.shuffle_rng.permutation(self.num_total_pos)
            self.all_pos_table = self.all_pos_table[perm]


class SummitCenteringCoordsBatcher(SamplingCoordsBatcher):
    """
    Creates a batch producer that batches positive coordinates only, each one
    centered at a summit.
    Arguments:
        `pos_coords_beds`: list of paths to gzipped BED files containing the
            sets of positive coordinates for various tasks
        `batch_size`: number of samples per batch
        `chroms_keep`: if specified, only considers this set of chromosomes from
            the coordinate BEDs
        `chrom_sizes_tsv`: path to 2-column TSV listing sizes of each chromosome
        `sample_length`: length of sampled sequence
        `return_peaks`: if True, returns the peaks and summits sampled from the
            peak set as a B x 3 array
        `shuffle_before_epoch`: Whether or not to shuffle all examples before
            each epoch
    """
    def __init__(
        self, pos_coords_beds, batch_size, chrom_sizes_tsv, sample_length,
        chroms_keep=None, return_peaks=False, shuffle_before_epoch=False,
        shuffle_seed=None
    ):
        # Same as a normal SamplingCoordsBatcher, but with no negatives and no
        # jitter, since the coordinates in the positive coordinate BEDs are
        # already centered at the summits
        super().__init__(
            pos_coords_beds=pos_coords_beds,
            batch_size=batch_size,
            neg_ratio=0,
            jitter=0,
            chrom_sizes_tsv=chrom_sizes_tsv,
            sample_length=sample_length,
            genome_sampler=None,
            chroms_keep=chroms_keep,
            return_peaks=return_peaks,
            shuffle_before_epoch=shuffle_before_epoch,
            shuffle_seed=shuffle_seed
        )

        
class PeakTilingCoordsBatcher(SamplingCoordsBatcher):
    """
    Creates a batch producer that batches positive coordinates only, where the
    coordinates are tiled such that all coordinate centers overlap with a peak.
    Arguments:
        `pos_coords_beds`: list of paths to gzipped BED files containing the
            sets of positive coordinates for various tasks
        `stride`: amount of stride when tiling the coordinates
        `batch_size`: number of samples per batch
        `chrom_sizes_tsv`: path to 2-column TSV listing sizes of each chromosome
        `sample_length`: length of sampled sequence
        `chroms_keep`: if specified, only considers this set of chromosomes from
            the coordinate BEDs
        `return_peaks`: if True, returns the peaks and summits sampled from the
            peak set as a B x 3 array
        `shuffle_before_epoch`: Whether or not to shuffle all examples before
            each epoch
    """
    def __init__(
        self, pos_coords_beds, stride, batch_size, chrom_sizes_tsv,
        sample_length, chroms_keep=None, return_peaks=False,
        shuffle_before_epoch=False, shuffle_seed=None
    ):
        # Similar to normal SamplingCoordsBatcher, but with no negatives and no
        # jitter; initialization is similar, but replicate the peaks so that
        # there are many summits tiled across each peak
        self.batch_size = batch_size
        self.jitter = 0
        self.chroms_keep = chroms_keep
        self.return_peaks = return_peaks
        self.shuffle_before_epoch = shuffle_before_epoch

        chrom_sizes_table = self._import_chrom_sizes(chrom_sizes_tsv)

        def tile_peaks(row):
            peak_start, peak_end = row[1], row[2]
            summit_offsets = np.arange(0, peak_end - peak_start, stride)
            num_expand = len(summit_offsets)
            row_expand = np.tile(row, (num_expand, 1))
            row_expand[:, -1] = summit_offsets
            return row_expand

        all_pos_table = []
        for i, pos_coords_bed in enumerate(pos_coords_beds):
            peaks_table = self._import_peaks(pos_coords_bed)
            
            # Formatting peaks table will expand the peaks to the right sample
            # length and filter for anything that overruns the chromosome
            # boundaries, so perform replication before
            peaks_values = peaks_table.values
            expanded_peaks_values = np.concatenate([
                tile_peaks(row) for row in peaks_values
            ], axis=0)
            expanded_peaks_table = pd.DataFrame.from_records(
                expanded_peaks_values, columns=list(peaks_table)
            )
            
            # Now format peaks table into N x 6 array
            coords = self._format_peaks_table(
                expanded_peaks_table, chroms_keep, None, sample_length, 0,
                chrom_sizes_table
            )
            
            # Add in the status column
            coords = np.concatenate(
                [coords, np.tile(i + 1, (len(coords), 1))], axis=1
            )  # Shape: _ x 7
            
            all_pos_table.append(coords)

        self.all_pos_table = np.concatenate(all_pos_table)  # Shape: N x 7
        self.num_total_pos = len(self.all_pos_table)

        # Number of positives and negatives per batch
        self.num_coords = len(self.all_pos_table)
        self.neg_per_batch = 0
        self.pos_per_batch = self.batch_size

        if shuffle_before_epoch:
            self.shuffle_rng = np.random.RandomState(shuffle_seed)


class CoordDataset(torch.utils.data.IterableDataset):
    """
    Generates single samples of a one-hot encoded sequence and value.
    Arguments:
        `coords_batcher (CoordsDownsampler): maps indices to batches of
            coordinates (split into positive and negative binding)
        `coords_to_seq (CoordsToSeq)`: maps coordinates to 1-hot encoded
            sequences
        `coords_to_vals (CoordsToVals)`: instantiated CoordsToVals object,
            mapping coordinates to profiles
        `revcomp`: whether or not to perform revcomp to the batch; this will
            double the batch size implicitly
        `return_coords`: if True, along with the 1-hot encoded sequences and
            values, the batch also returns the set of coordinates used for the
            batch, and the peak/summit locations for the positive examples
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

        # The dataset returns coordinates iff the batcher returns peak info
        assert coords_batcher.return_peaks == return_coords

    def get_batch(self, index):
        """
        Returns a batch, which consists of an B x L x 4 NumPy array of 1-hot
        encoded sequence, the associated profiles, and a 1D length-B NumPy array
        of statuses. The profiles will be a B x L x S x 2 array of profiles. The
        coordinates and peaks may also be returned as a B x 3 array.
        """
        # Get batch of coordinates for this index
        if self.return_coords:
            coords, status, peaks = self.coords_batcher[index]
        else:
            coords, status = self.coords_batcher[index]

        # Map this batch of coordinates to 1-hot encoded sequences
        seqs = self.coords_to_seq(coords, revcomp=self.revcomp)

        # Map this batch of coordinates to the associated values
        profiles = self.coords_to_vals(coords)

        # Profiles are returned as B x L x S x 2, so transpose to get
        # B x S x L x 2
        profiles = np.swapaxes(profiles, 1, 2)

        # If reverse complementation was done, double sizes of everything else
        if self.revcomp:
            profiles = np.concatenate(
                # To reverse complement, we must swap the strands AND the
                # directionality of each strand (i.e. we are assigning the other
                # strand to be the plus strand, but still 5' to 3')
                [profiles, np.flip(profiles, axis=(2, 3))]
            )
            status = np.concatenate([status, status])

        if self.return_coords:
            if self.revcomp:
                coords_ret = np.concatenate([coords, coords])
                peaks_ret = np.concatenate([peaks, peaks])
            else:
                coords_ret, peaks_ret = coords, peaks
            return seqs, profiles, status, coords_ret, peaks_ret
        else:
            return seqs, profiles, status

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
def create_data_loader(
    peaks_bed_paths, profile_hdf5_path, sampling_type, batch_size,
    reference_fasta, chrom_sizes_tsv, input_length, profile_length,
    negative_ratio, peak_tiling_stride, peak_retention, num_workers, revcomp,
    jitter_size, negative_seed, shuffle_seed, jitter_seed, chrom_set=None,
    shuffle=True, return_coords=False
):
    """
    Creates an IterableDataset object, which iterates through batches of
    coordinates and returns profiles for the coordinates.
    Arguments:
        `peaks_bed_paths`: a list of paths to gzipped 6-column BED files
            containing coordinates of positive-binding coordinates
        `profile_hdf5_path`: path to HDF5 containing reads mapped to each
            coordinate; this HDF5 must be organized by chromosome, with each
            dataset being L x S x 2, where L is the length of the chromosome,
            S is the number of tracks stored, and 2 is for each strand
        `sampling_type`: one of ("SamplingCoordsBatcher",
            "SummitCenteringCoordsBatcher", or "PeakTilingCoordsBatcher"), which
            corresponds to sampling positive and negative regions, taking only
            positive regions centered around summits, and taking only positive
            regions tiled across peaks
        `chrom_set`: a list of chromosomes to restrict to for the positives and
            sampled negatives; defaults to all coordinates in the given BEDs and
            sampling over the entire genome
        `shuffle`: if specified, shuffle the coordinates before each epoch
        `return_coords`: if specified, also return the underlying coordinates
            and peak data along with the profiles in each batch
    """
    assert sampling_type in (
            "SamplingCoordsBatcher", "SummitCenteringCoordsBatcher",
            "PeakTilingCoordsBatcher"
    )

    # Maps set of coordinates to profiles
    coords_to_vals = CoordsToVals(profile_hdf5_path, profile_length)

    if sampling_type == "SamplingCoordsBatcher":
        # Randomly samples from genome
        genome_sampler = GenomeIntervalSampler(
            chrom_sizes_tsv, input_length, chroms_keep=chrom_set,
            seed=negative_seed
        )
        # Yields batches of positive and negative coordinates
        coords_batcher = SamplingCoordsBatcher(
            peaks_bed_paths, batch_size, negative_ratio, jitter_size,
            chrom_sizes_tsv, input_length, genome_sampler,
            chroms_keep=chrom_set, peak_retention=peak_retention,
            return_peaks=return_coords, shuffle_before_epoch=shuffle,
            jitter_seed=jitter_seed, shuffle_seed=shuffle_seed
        )
    elif sampling_type == "SummitCenteringCoordsBatcher":
        # Yields batches of positive coordinates, centered at summits
        coords_batcher = SummitCenteringCoordsBatcher(
            peaks_bed_paths, batch_size, chrom_sizes_tsv, input_length,
            chroms_keep=chrom_set, return_peaks=return_coords,
            shuffle_before_epoch=shuffle, shuffle_seed=shuffle_seed
        )
    else:
        # Yields batches of positive coordinates, tiled across peaks
        coords_batcher = PeakTilingCoordsBatcher(
            peaks_bed_paths, peak_tiling_stride, batch_size, chrom_sizes_tsv,
            input_length, chroms_keep=chrom_set, return_peaks=return_coords,
            shuffle_before_epoch=shuffle, shuffle_seed=shuffle_seed
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
    import json
    import matplotlib.pyplot as plt

    paths_json_path = "/users/amtseng/att_priors/data/processed/ENCODE_TFChIP/profile/config/SPI1/SPI1_training_paths.json"
    with open(paths_json_path, "r") as f:
        paths_json = json.load(f)
    peak_beds = paths_json["peak_beds"]
    profile_hdf5 = paths_json["profile_hdf5"]

    splits_json_path = "/users/amtseng/att_priors/data/processed/chrom_splits.json"
    with open(splits_json_path, "r") as f:
        splits_json = json.load(f)
    train_chroms, val_chroms, test_chroms = \
        splits_json["1"]["train"], splits_json["1"]["val"], \
        splits_json["1"]["test"]

    loader = create_data_loader(
        peak_beds, profile_hdf5, "SamplingCoordsBatcher",
        return_coords=True, chrom_set=val_chroms,
        jitter_size=128, jitter_seed=123, peak_retention=0.1,
        shuffle_seed=123, negative_seed=123
    )
    loader.dataset.on_epoch_start()

    start_time = datetime.now()
    for batch in tqdm.tqdm(loader, total=len(loader.dataset)):
        data = batch
    end_time = datetime.now()
    print("Time: %ds" % (end_time - start_time).seconds)

    k = 2
    rc_k = int(len(data[0]) / 2) + k

    seqs, profiles, statuses, coords, peaks = data
    
    seq, prof, status = seqs[k], profiles[k], statuses[k]
    rc_seq, rc_prof, rc_status = \
        seqs[rc_k], profiles[rc_k], statuses[rc_k]

    coord, peak = coords[k], peaks[k]
    rc_coord, rc_peak = coords[rc_k], peaks[rc_k]

    def print_one_hot_seq(one_hot):
        s = util.one_hot_to_seq(one_hot)
        print(s[:20] + "..." + s[-20:])
   
    print_one_hot_seq(seq)
    print_one_hot_seq(rc_seq)

    print(status, rc_status)

    print(coord, rc_coord)
    print(peak, rc_peak)

    task_ind = 0
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(prof[task_ind][:, 0])
    ax[0].plot(prof[task_ind][:, 1])
    ax[1].plot(rc_prof[task_ind][:, 0])
    ax[1].plot(rc_prof[task_ind][:, 1])

    plt.show()
