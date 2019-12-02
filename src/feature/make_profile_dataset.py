import torch
import numpy as np
import pandas as pd
import sacred
from datetime import datetime
import pyBigWig
import feature.util as util

dataset_ex = sacred.Experiment("dataset")

@dataset_ex.config
def config():
    # Path to reference genome FASTA
    reference_fasta = "/users/amtseng/genomes/hg38.fasta"

    # Path to chromosome sizes
    chrom_sizes = "/users/amtseng/genomes/hg38.canon.chrom.sizes"

    # The size of DNA sequences to fetch as input sequences
    input_length = 1346

    # The size of profiles to fetch for each coordinate
    profile_length = 1000

    # One-hot encoding has this depth
    input_depth = 4

    # Whether or not to perform reverse complement augmentation
    revcomp = True

    # Maximum size of jitter to the input for augmentation; set to 0 to disable
    jitter_size = 50

    # Batch size; will be multiplied by two if reverse complementation is done
    batch_size = 128

    # Sample X negatives randomly from the genome for every positive example
    negative_ratio = 1

    # Use this stride when tiling coordinates across a peak
    peak_tiling_stride = 25

    # Probability of noising/varying a positive example
    noise_prob = 0

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
        `chrom_sizes`: path to 2-column TSV listing sizes of each chromosome
        `sample_length`: length of sampled sequence
        `chroms_keep`: an iterable of chromosomes that specifies which
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
        `bigwig_path`: path to BigWig containing profile
        `center_size_to_use`: for each genomic coordinate, center it and pad it
            on both sides to this length to get the final profile; if this is
            smaller than the coordinate interval given, then the interval will
            be cut to this size by centering
    """
    def __init__(self, bigwig_path, center_size_to_use):
        self.bigwig_path = bigwig_path
        self.center_size_to_use = center_size_to_use

    def _get_profile(self, chrom, start, end, bigwig_reader):
        """
        Fetches the profile for the given coordinates, with an instantiated
        BigWig reader. Returns the profile as a NumPy array of numbers. This may
        pad or cut from the center to a specified length.
        """
        if self.center_size_to_use:
            center = int(0.5 * (start + end))
            half_size = int(0.5 * self.center_size_to_use)
            left = center - half_size
            right = center + self.center_size_to_use - half_size
            return np.nan_to_num(bigwig_reader.values(chrom, left, right))
        else:
            return np.nan_to_num(bigwig_reader.values(chrom, start, end))
        
    def _get_ndarray(self, coords):
        """
        From an iterable of coordinates, retrieves a 2D B x L NumPy array of
        corresponding profile values. Note that all coordinate intervals need
        to be of the same length. 
        """
        reader = pyBigWig.open(self.bigwig_path)
        return np.stack([
            self._get_profile(chrom, start, end, reader) \
            for chrom, start, end in coords
        ])

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
            sets of positive coordinates for various tasks
        `batch_size`: number of samples per batch
        `neg_ratio`: number of negatives to select for each positive example
        `jitter`: random amount to jitter each positive coordinate example by
        `genome_sampler`: a GenomeIntervalSampler instance, which samples
            intervals randomly from the genome
        `noise_prob`: probability of corruption for a positive example; a
            roughly equal number of negatives are also corrupted
        `return_peaks`: if True, returns the peaks and summits sampled from the
            peak set as a B x 3 array
        `shuffle_before_epoch`: Whether or not to shuffle all examples before
            each epoch
    """
    def __init__(
        self, pos_coords_beds, batch_size, neg_ratio, jitter, genome_sampler,
        noise_prob=0, return_peaks=False, shuffle_before_epoch=False, seed=None
    ):
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.jitter = jitter
        self.genome_sampler = genome_sampler
        self.noise_prob = noise_prob
        self.return_peaks = return_peaks
        self.shuffle_before_epoch = shuffle_before_epoch

        assert neg_ratio * noise_prob == 0, \
            "Can only have noise without negatives, and vice versa"

        # Read in the positive coordinates and make N x 7 array, where the
        # 7th column is the identifier of the source BED
        # Cols 1-3 should be the coordinate, 4-5 are the original peak location,
        # and col 6 is the summit location
        sample_coords = []
        for i, pos_coords_bed in enumerate(pos_coords_beds):
            pos_coords_table = pd.read_csv(
                pos_coords_bed, sep="\t", header=None, compression="gzip"
            )
            coords = pos_coords_table.values.astype(object)
            coords = np.concatenate(
                [coords, np.tile(i + 1, (len(coords), 1))], axis=1
            )
            sample_coords.append(coords)
        self.sample_coords = np.concatenate(sample_coords)

        # Randomly select a subset of the sample coordinates to have different
        # profiles
        if noise_prob:
            num_to_noise = int(noise_prob * len(self.sample_coords))
            rand_mask = np.random.choice(
                len(self.sample_coords), size=num_to_noise, replace=False
            )
            neg_coords = genome_sampler.sample_intervals(num_to_noise)
            self.noise_coords = self.sample_coords[:, :3].copy()
            self.noise_coords[rand_mask] = neg_coords
            self.sample_coords[rand_mask, 6] *= -1  # Negative, to denote noise

        # Number of positives and negatives per batch
        self.num_coords = len(self.sample_coords)
        self.neg_per_batch = int(batch_size * neg_ratio / (neg_ratio + 1))
        self.pos_per_batch = batch_size - self.neg_per_batch

        if shuffle_before_epoch:
            self.rng = np.random.RandomState(seed)

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
        If `noise_prob` was nonzero at object creation-time, coordinates will be
        a B x 6 2D NumPy array, where the first 3 columns are the original
        coordinate, and the next 3 columns are a randomly selected coordinate
        or the same coordinate.
        """
        pos_coords = self.sample_coords[
            index * self.pos_per_batch : (index + 1) * self.pos_per_batch
        ]

        # If specified, apply random jitter to each positive coordinate
        if self.jitter:
            jitter_vals = np.random.randint(
                -self.jitter, self.jitter + 1, size=len(pos_coords)
            )
            pos_coords[:,1] += jitter_vals
            pos_coords[:,2] += jitter_vals
        
        if self.neg_per_batch:
            neg_coords = self.genome_sampler.sample_intervals(
                self.neg_per_batch
            )
        else:
            neg_coords = np.empty(shape=(0, 3), dtype=object)
        coords = np.concatenate([pos_coords[:, :3], neg_coords])  # Cols 1-3

        # If noising is to be done, add in the noisy coordinates
        if self.noise_prob:
            noise_coords = self.noise_coords[
                index * self.pos_per_batch : (index + 1) * self.pos_per_batch
            ]
            coords = np.concatenate([coords, noise_coords], axis=1)

        status = np.concatenate(
            [pos_coords[:, 6], np.zeros(len(neg_coords))]  # Col 7
        ).astype(int)

        if self.return_peaks:
            pos_peaks = pos_coords[:, 3:6]  # Cols 4-6
            neg_peaks = np.full((len(neg_coords), 3), -1)
            peaks = np.concatenate([pos_peaks, neg_peaks])
            return coords, status, peaks
        else:
            return coords, status

    def __len__(self):
        return int(np.ceil(self.num_coords / float(self.pos_per_batch)))
   
    def on_epoch_start(self):
        if self.shuffle_before_epoch:
            perm = self.rng.permutation(len(self.sample_coords))
            self.sample_coords = self.sample_coords[perm]
            if self.noise_prob:
                self.noise_coords = self.noise_coords[perm]


class SummitCenteringCoordsBatcher(SamplingCoordsBatcher):
    """
    Creates a batch producer that batches positive coordinates only, each one
    centered at a summit.
    Arguments:
        `pos_coords_beds`: list of paths to gzipped BED files containing the
            sets of positive coordinates for various tasks
        `batch_size`: number of samples per batch
        `return_peaks`: if True, returns the peaks and summits sampled from the
            peak set as a B x 3 array
        `shuffle_before_epoch`: Whether or not to shuffle all examples before
            each epoch
    """
    def __init__(
        self, pos_coords_beds, batch_size, return_peaks=False,
        shuffle_before_epoch=False, seed=None
    ):
        # Same as a normal SamplingCoordsBatcher, but with no negatives and no
        # jitter, since the coordinates in the positive coordinate BEDs are
        # already centered at the summits
        super().__init__(
            pos_coords_beds=pos_coords_beds,
            batch_size=batch_size,
            neg_ratio=0,
            jitter=0,
            genome_sampler=None,
            return_peaks=return_peaks,
            shuffle_before_epoch=shuffle_before_epoch,
            seed=seed
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
        `return_peaks`: if True, returns the peaks and summits sampled from the
            peak set as a B x 3 array
        `shuffle_before_epoch`: Whether or not to shuffle all examples before
            each epoch
    """
    def __init__(
        self, pos_coords_beds, stride, batch_size, return_peaks=False,
        shuffle_before_epoch=False, seed=None
    ):
        self.stride = stride
        self.batch_size = batch_size
        self.jitter = 0
        self.return_peaks = return_peaks
        self.shuffle_before_epoch = shuffle_before_epoch

        # Read in the positive coordinates and make N x 4 array, containing only
        # cols 1, 4-6, which are the original peak chromosome, start/end, and
        # summit location, as well as a status indicating the original BED file
        peak_coords = []
        for i, pos_coords_bed in enumerate(pos_coords_beds):
            pos_coords_table = pd.read_csv(
                pos_coords_bed, sep="\t", header=None, compression="gzip",
                usecols=[0, 3, 4, 5]
            )
            coords = pos_coords_table.values.astype(object)
            coords = np.concatenate(
                [coords, np.tile(i + 1, (len(coords), 1))], axis=1
            )
            peak_coords.append(coords)
        all_peak_coords = np.concatenate(peak_coords)

        # For each peak, tile a set of coordinate centers across the peak and
        # make an N x 7 array
        def tile_peak(peak_coord_row):
            # Creates M x 7 array from a single length-4 row of all_peak_coords
            # Result is chromosome (col 1), tiled coordinates (cols 2-3) where
            # each coordinate is length 1, peak coordinates (cols 4-5), summit
            # location (col 6), and status (col 7)
            peak_start, peak_end = peak_coord_row[1], peak_coord_row[2]
            coord_starts = np.expand_dims(
                np.arange(peak_start, peak_end, stride), axis=1
            )
            coord_ends = coord_starts + 1
            num_coords = len(coord_starts)
            row_expand = np.tile(peak_coord_row, (num_coords, 1))
            return np.concatenate([
                row_expand[:, :1],
                coord_starts,
                coord_ends,
                row_expand[:, 1:]
            ], axis=1)

        self.sample_coords = np.concatenate(
            [tile_peak(row) for row in all_peak_coords], axis=0
        )

        # Number of positives and negatives per batch
        self.num_coords = len(self.sample_coords)
        self.neg_per_batch = 0
        self.pos_per_batch = self.batch_size

        if shuffle_before_epoch:
            self.rng = np.random.RandomState(seed)


class CoordDataset(torch.utils.data.IterableDataset):
    """
    Generates single samples of a one-hot encoded sequence and value.
    Arguments:
        `coords_batcher (CoordsDownsampler): maps indices to batches of
            coordinates (split into positive and negative binding)
        `coords_to_seq (CoordsToSeq)`: maps coordinates to 1-hot encoded
            sequences
        `coords_to_vals_list (list of CoordsToVals)`: list of instantiated
            CoordsToVals objects, each of which maps coordinates to profiles
            to predict
        `revcomp`: whether or not to perform revcomp to the batch; this will
            double the batch size implicitly
        `return_coords`: if True, along with the 1-hot encoded sequences and
            values, the batch also returns the set of coordinates used for the
            batch, and the peak/summit locations for the positive examples
    """
    def __init__(
        self, coords_batcher, coords_to_seq, coords_to_vals_list, revcomp=False,
        return_coords=False
    ):
        self.coords_batcher = coords_batcher
        self.coords_to_seq = coords_to_seq
        self.coords_to_vals_list = coords_to_vals_list
        self.revcomp = revcomp
        self.return_coords = return_coords

        # The dataset returns coordinates iff the batcher returns peak info
        assert coords_batcher.return_peaks == return_coords

    def get_batch(self, index):
        """
        Returns a batch, which consists of an B x L x 4 NumPy array of 1-hot
        encoded sequence, an B x T x L x 2 NumPy array of profiles, and a 1D
        length-B NumPy array of statuses. The profile for each of the T tasks in
        `coords_to_vals_list` is returned, in the same order as in this list,
        and each task contains 2 tracks, for the plus and minus strand,
        respectively. This function will also perform noising if specified by
        the coordinate statuses
        """
        # Get batch of coordinates for this index
        if self.return_coords:
            coords, status, peaks = self.coords_batcher[index]
        else:
            coords, status = self.coords_batcher[index]

        # If noising was done, extract out the noise coordinates and use those
        # for profiles
        if self.coords_batcher.noise_prob:
            seq_coords, prof_coords = coords[:, :3], coords[:, 3:]
        else:
            seq_coords, prof_coords = coords, coords

        # Map this batch of coordinates to 1-hot encoded sequences
        seqs = self.coords_to_seq(seq_coords, revcomp=self.revcomp)

        # Map this batch of coordinates to the associated profiles
        profiles = np.stack([
            np.stack([ctv_1(prof_coords), ctv_2(prof_coords)], axis=2) \
            for ctv_1, ctv_2 in self.coords_to_vals_list
        ], axis=1)

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
                coords_ret = np.concatenate([seq_coords, seq_coords])
                peaks_ret = np.concatenate([peaks, peaks])
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
    peaks_bed_paths, profile_bigwig_paths, sampling_type, batch_size,
    reference_fasta, chrom_sizes, input_length, profile_length, negative_ratio,
    peak_tiling_stride, noise_prob, num_workers, revcomp, jitter_size,
    dataset_seed, shuffle=True, return_coords=False
):
    """
    Creates an IterableDataset object, which iterates through batches of
    coordinates and returns profiles for the coordinates.
    Arguments:
        `peaks_bed_paths`: a list of paths to gzipped 6-column BED files
            containing coordinates of positive-binding coordinates
        `profile_bigwig_paths`: a list of paths to BigWigs containing reads
            mapped to each location in the genome; each entry must be a pair of
            BigWig paths, corresponding to the profile of the plus and minus
            strand, respectively
        `sampling_type`: one of ("SamplingCoordsBatcher",
            "SummitCenteringCoordsBatcher", or "PeakTilingCoordsBatcher"), which
            corresponds to sampling positive and negative regions, taking only
            positive regions centered around summits, and taking only positive
            regions tiled across peaks
        `shuffle`: if specified, shuffle the coordinates before each epoch
        `return_coords`: if specified, also return the underlying coordinates
            and peak data along with the profiles in each batch
    """
    assert sampling_type in (
            "SamplingCoordsBatcher", "SummitCenteringCoordsBatcher",
            "PeakTilingCoordsBatcher"
    )

    # Maps set of coordinates to profiles
    coords_to_vals_list = [
        (
            CoordsToVals(path_1, profile_length),
            CoordsToVals(path_2, profile_length)
        ) for path_1, path_2 in profile_bigwig_paths
    ]

    if sampling_type == "SamplingCoordsBatcher":
        # Randomly samples from genome
        genome_sampler = GenomeIntervalSampler(
            chrom_sizes, input_length, seed=dataset_seed
        )
        # Yields batches of positive and negative coordinates
        coords_batcher = SamplingCoordsBatcher(
            peaks_bed_paths, batch_size, negative_ratio, jitter_size,
            genome_sampler, noise_prob, return_peaks=return_coords,
            shuffle_before_epoch=shuffle, seed=dataset_seed
        )
    elif sampling_type == "SummitCenteringCoordsBatcher":
        # Yields batches of positive coordinates, centered at summits
        coords_batcher = SummitCenteringCoordsBatcher(
            peaks_bed_paths, batch_size, return_peaks=return_coords,
            shuffle_before_epoch=shuffle, seed=dataset_seed
        )
    else:
        # Yields batches of positive coordinates, tiled across peaks
        coords_batcher = PeakTilingCoordsBatcher(
            peaks_bed_paths, peak_tiling_stride, batch_size,
            return_peaks=return_coords, shuffle_before_epoch=shuffle,
            seed=dataset_seed
        )

    # Maps set of coordinates to 1-hot encoding, padded
    coords_to_seq = util.CoordsToSeq(
        reference_fasta, center_size_to_use=input_length
    )

    # Dataset
    dataset = CoordDataset(
        coords_batcher, coords_to_seq, coords_to_vals_list, revcomp=revcomp,
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

    base_path = "/users/amtseng/att_priors/data/interim/ENCODE/profile/labels/"

    peaks_bed_files = [
        os.path.join(base_path, ending) for ending in [
            "SPI1/SPI1_ENCSR000BGQ_GM12878_all_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BGW_K562_all_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BIJ_GM12891_all_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BUW_HL-60_all_peakints.bed.gz"
        ]
    ]
            
    profile_bigwig_files = [
        (os.path.join(base_path, e_1), os.path.join(base_path, e_2))
        for e_1, e_2 in [
            ("SPI1/SPI1_ENCSR000BGQ_GM12878_neg.bw",
            "SPI1/SPI1_ENCSR000BGQ_GM12878_pos.bw"),
            ("SPI1/SPI1_ENCSR000BGW_K562_neg.bw",
            "SPI1/SPI1_ENCSR000BGW_K562_pos.bw"),
            ("SPI1/SPI1_ENCSR000BIJ_GM12891_neg.bw",
            "SPI1/SPI1_ENCSR000BIJ_GM12891_pos.bw"),
            ("SPI1/SPI1_ENCSR000BUW_HL-60_neg.bw",
            "SPI1/SPI1_ENCSR000BUW_HL-60_pos.bw"),
            ("SPI1/control_ENCSR000BGG_K562_neg.bw",
            "SPI1/control_ENCSR000BGG_K562_pos.bw"),
            ("SPI1/control_ENCSR000BGH_GM12878_neg.bw",
            "SPI1/control_ENCSR000BGH_GM12878_pos.bw"),
            ("SPI1/control_ENCSR000BIH_GM12891_neg.bw",
            "SPI1/control_ENCSR000BIH_GM12891_pos.bw"),
            ("SPI1/control_ENCSR000BVU_HL-60_neg.bw",
            "SPI1/control_ENCSR000BVU_HL-60_pos.bw")
        ]
    ]

    loader = create_data_loader(
        peaks_bed_files, profile_bigwig_files, "SamplingCoordsBatcher",
        return_coords=True, noise_prob=0.5, negative_ratio=0
    )
    loader.dataset.on_epoch_start()
    start_time = datetime.now()
    for batch in tqdm.tqdm(loader, total=len(loader.dataset)):
        data = batch
        break
    end_time = datetime.now()
    print("Time: %ds" % (end_time - start_time).seconds)

    k = 2
    rc_k = int(len(data[0]) / 2) + k

    seqs, profiles, statuses, coords, peaks = data
    
    seq, prof, status = seqs[k], profiles[k], statuses[k]
    rc_seq, rc_prof, rc_status = seqs[rc_k], profiles[rc_k], statuses[rc_k]

    coord, peak = coords[k], peaks[k]
    rc_coord, rc_peak = coords[rc_k], peaks[rc_k]
    
    print(util.one_hot_to_seq(seq))
    print(util.one_hot_to_seq(rc_seq))

    print(status, rc_status)

    print(coord, rc_coord)
    print(peak, rc_peak)

    import matplotlib.pyplot as plt
    task_ind = 0
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(prof[task_ind][:, 0])
    ax[0].plot(prof[task_ind][:, 1])

    ax[1].plot(rc_prof[task_ind][:, 0])
    ax[1].plot(rc_prof[task_ind][:, 1])

    plt.show()
