import torch
import numpy as np
import pandas as pd
import sacred
from datetime import datetime
import h5py
import feature.util as util
import tqdm

dataset_ex = sacred.Experiment("dataset")

@dataset_ex.config
def config():
    # Path to reference genome FASTA
    reference_fasta = "/users/amtseng/genomes/hg38.fasta"

    # Path to chromosome sizes
    chrom_sizes_tsv = "/users/amtseng/genomes/hg38.canon.chrom.sizes"

    # For each input sequence in the raw data, center it and pad to this length 
    input_length = 1000

    # One-hot encoding has this depth
    input_depth = 4

    # Whether or not to perform reverse complement augmentation
    revcomp = True

    # Batch size; will be multiplied by two if reverse complementation is done
    batch_size = 128

    # Sample X negatives randomly for every positive example
    negative_ratio = 1

    # Number of workers for the data loader
    num_workers = 10

    # Shuffle seed (for shuffling data points)
    shuffle_seed = None


class BinsToVals():
    """
    From an HDF5 file that maps genomic coordinates to profiles, this creates an
    object that maps a list of bind indices to a NumPy array of coordinates and
    output values.
    Arguments:
        `label_hdf5`: path to HDF5 containing labels; this HDF5 must be a single
            dataset created by `generate_ENCODE_TFChIP_binary_labels.sh`; each
            row must be: (index, values, end, start, chrom), where the values is
            a T-array of values, for each task T, containing 0, 1, or nan
    """
    def __init__(self, label_hdf5):
        self.label_hdf5 = label_hdf5

    def _get_ndarray(self, bin_inds):
        """
        From a list of bin indices, returns a B x 3 array of coordinates and a
        B x T array of corresponding values.
        """
        with h5py.File(self.label_hdf5, "r") as f:
            batch = f["data"]["table"][np.sort(bin_inds)]
            coords = np.array(
                [[row[4], row[3], row[2]] for row in batch], dtype=object
            )
            coords[:, 0] = coords[:, 0].astype(str)  # Convert chrom to string
            vals = np.nan_to_num(
                np.array([row[1] for row in batch]), nan=-1
            )
        return coords, vals.astype(int)

    def __call__(self, bin_inds):
        return self._get_ndarray(bin_inds)


class SamplingBinsBatcher(torch.utils.data.sampler.Sampler):
    """
    Creates a batch producer that batches bin indices for positive bins and
    negative bins. Each batch will have some positives and negatives according
    to `neg_ratio`.
    Arguments:
        `label_hdf5`: path to HDF5 containing labels; this HDF5 must be a single
            dataset created by `generate_ENCODE_TFChIP_binary_labels.sh`; each
            row must be: (index, values, end, start, chrom), where the values is
            a T-array of values, for each task T, containing 0, 1, or nan
        `batch_size`: number of samples per batch
        `neg_ratio`: number of negatives to select for each positive example
        `chroms_keep`: if specified, only considers this set of chromosomes from
            the coordinate BEDs
        `shuffle_before_epoch`: whether or not to shuffle all examples before
            each epoch
        `shuffle_seed`: seed for shuffling
    """
    def __init__(
        self, label_hdf5, batch_size, neg_ratio, chroms_keep=None,
        shuffle_before_epoch=False, shuffle_seed=None
    ):
        self.batch_size = batch_size
        self.shuffle_before_epoch = shuffle_before_epoch

        # From the set of labels, determine indices that are to be considered
        # positives (i.e. has a 1) or negatives (i.e. all 0)
        # Note that this will disregard any bins that are all ambiguous or only
        # ambiguous and negative
        print("Gathering bin label indices:")
        with h5py.File(label_hdf5, "r") as f:
            data = f["data"]["table"]
            labels = -np.ones(data.shape)
            if chroms_keep:
                chroms_keep = np.array(chroms_keep)
           
            chunk_size = 20000
            num_chunks = int(np.ceil(data.shape[0] / chunk_size))
            for i in tqdm.trange(num_chunks):
                chunk_slice = slice(i * chunk_size, (i + 1) * chunk_size)
                chunk = data[chunk_slice]
                vals = np.array([row[1] for row in chunk])

                # Mask for where a row is positive or negative
                pos_mask = np.any(vals == 1, axis=1)
                neg_mask = np.all(vals == 0, axis=1)
                if chroms_keep is not None:
                    # Only keep the places that are the right chromosome
                    chroms = np.array([row[4] for row in chunk]).astype(str)
                    chrom_mask = np.isin(chroms, chroms_keep)
                    pos_mask = pos_mask & chrom_mask
                    neg_mask = neg_mask & chrom_mask

                labels[chunk_slice][pos_mask] = 1
                labels[chunk_slice][neg_mask] = 0
                # -1 wherever ambiguous or did not pass mask (e.g. wrong chrom)

        self.pos_inds = np.where(labels == 1)[0]
        self.neg_inds = np.where(labels == 0)[0]

        self.neg_per_batch = int(batch_size * neg_ratio / (neg_ratio + 1))
        self.pos_per_batch = batch_size - self.neg_per_batch

        if shuffle_before_epoch:
            self.rng = np.random.RandomState(shuffle_seed)

    def __getitem__(self, index):
        """
        Fetches a full batch of positive and negative bin indices. Returns
        a B-array of bin indices, and a B-array of statuses. The status is
        either 1 or 0: 1 if that bin has a positive binding event for some task,
        and 0 if that bin has no binding events over all tasks.
        """
        pos_inds = self.pos_inds[
            index * self.pos_per_batch : (index + 1) * self.pos_per_batch
        ]
        neg_inds = self.neg_inds[
            index * self.neg_per_batch : (index + 1) * self.neg_per_batch
        ]
        bin_inds = np.concatenate([pos_inds, neg_inds])
        status = np.concatenate([
            np.ones(len(pos_inds), dtype=int),
            np.zeros(len(neg_inds), dtype=int)
        ])
        return bin_inds, status

    def __len__(self):
        return int(np.ceil(len(self.pos_inds) / self.pos_per_batch))
   
    def _shuffle(self):
        """
        Shuffles the positive and negative indices, if appropriate.
        """
        if (self.shuffle_before_epoch):
            self.rng.shuffle(self.pos_inds)
            self.rng.shuffle(self.neg_inds)

    def on_epoch_start(self):
        self._shuffle()


class BinDataset(torch.utils.data.IterableDataset):
    """
    Generates single samples of a one-hot encoded sequence and value.
    Arguments:
        `bin_batcher (SamplingBinsBatcher): maps indices to batches of
            bin indices
        `coords_to_seq (CoordsToSeq)`: maps coordinates to 1-hot encoded
            sequences
        `bins_to_vals (BinsToVals)`: maps bin indices to values to predict
        `revcomp`: whether or not to perform revcomp to the batch; this will
            double the batch size implicitly
        `return_coords`: if True, each batch returns the set of coordinates for
            that batch along with the 1-hot encoded sequences and values
    """
    def __init__(
        self, bins_batcher, coords_to_seq, bins_to_vals, revcomp=False,
        return_coords=False
    ):
        self.bins_batcher = bins_batcher
        self.coords_to_seq = coords_to_seq
        self.bins_to_vals = bins_to_vals
        self.revcomp = revcomp
        self.return_coords = return_coords

    def get_batch(self, index):
        """
        Returns a batch, which consists of an B x L x 4 NumPy array of 1-hot
        encoded sequence, the associated B x T values, and a 1D length-B NumPy
        array of statuses. The coordinates may also be returned as a B x 3
        array.
        """
        # Get batch of bin indices for this index
        bin_inds_batch, status = self.bins_batcher[index]

        # Map this batch of bin indices to coordinates and output values
        coords, vals = self.bins_to_vals(bin_inds_batch)

        # Map the batch of coordinates to 1-hot encoded sequences
        seqs = self.coords_to_seq(coords, revcomp=self.revcomp)

        if self.revcomp:
            vals = np.concatenate([vals, vals])
            status = np.concatenate([status, status])

        if self.return_coords:
            if self.revcomp:
                coords = np.concatenate([coords, coords])
            return seqs, vals, status, coords
        else:
            return seqs, vals, status

    def __iter__(self):
        """
        Returns an iterator over the batches. If the dataset iterator is called
        from multiple workers, each worker will be give a shard of the full
        range.
        """
        worker_info = torch.utils.data.get_worker_info()
        num_batches = len(self.bins_batcher)
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
        return len(self.bins_batcher)
    
    def on_epoch_start(self):
        """
        This should be called manually before the beginning of every epoch (i.e.
        before the iteration begins).
        """
        self.bins_batcher.on_epoch_start()


@dataset_ex.capture
def create_data_loader(
    labels_hdf5_path, batch_size, reference_fasta, input_length, negative_ratio,
    num_workers, revcomp, shuffle_seed, chrom_set=None, shuffle=True,
    return_coords=False
):
    """
    Creates an IterableDataset object, which iterates through batches of
    bins and returns values for the bins.
    Arguments:
        `labels_hdf5_path`: path to HDF5 containing labels; this HDF5 must be a
            single dataset created by `generate_ENCODE_TFChIP_binary_labels.sh`;
            each row must be: (index, values, end, start, chrom), where the
            values is a T-array of values, for each task T, containing 0, 1, or
            nan
        `chrom_set`: a list of chromosomes to restrict to for the positives and
            negatives; defaults to all coordinates in HDF5
        `shuffle`: if specified, shuffle the coordinates before each epoch
        `return_coords`: if specified, also return the underlying coordinates
            along with the values in each batch
    """
    # Maps set of bin indices to values and coordinates
    bins_to_vals = BinsToVals(labels_hdf5_path)

    # Yields batches of positive and negative bin indices
    bins_batcher = SamplingBinsBatcher(
        labels_hdf5_path, batch_size, negative_ratio, chroms_keep=chrom_set,
        shuffle_before_epoch=shuffle, shuffle_seed=shuffle_seed
    )

    print("Total class counts:")
    num_pos, num_neg = len(bins_batcher.pos_inds), len(bins_batcher.neg_inds)
    print("Pos: %d\nNeg: %d" % (num_pos, num_neg))
    if num_pos:
        print("Neg/Pos = %f" % (num_neg / num_pos))

    # Maps set of coordinates to 1-hot encoding, padded
    coords_to_seq = util.CoordsToSeq(
        reference_fasta, center_size_to_use=input_length
    )
    
    # Dataset
    dataset = BinDataset(
        bins_batcher, coords_to_seq, bins_to_vals, revcomp=revcomp,
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

    paths_json_path = "/users/amtseng/att_priors/data/processed/ENCODE_TFChIP/binary/config/SPI1/SPI1_training_paths.json"
    
    with open(paths_json_path, "r") as f:
        paths_json = json.load(f)
    labels_hdf5 = paths_json["labels_hdf5"]

    splits_json_path = "/users/amtseng/att_priors/data/processed/chrom_splits.json"
    with open(splits_json_path, "r") as f:
        splits_json = json.load(f)
    train_chroms, val_chroms, test_chroms = \
        splits_json["1"]["train"], splits_json["1"]["val"], \
        splits_json["1"]["test"]

    loader = create_data_loader(
        labels_hdf5, return_coords=True, chrom_set=None, shuffle_seed=123
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

    seqs, vals, statuses, coords = data
    
    seq, val, status, coord = seqs[k], vals[k], statuses[k], coords[k]
    rc_seq, rc_val, rc_status, rc_coord = \
        seqs[rc_k], vals[rc_k], statuses[rc_k], coords[rc_k]

    def print_one_hot_seq(one_hot):
        s = util.one_hot_to_seq(one_hot)
        print(s[:20] + "..." + s[-20:])
   
    print_one_hot_seq(seq)
    print_one_hot_seq(rc_seq)

    print(np.sum(val), np.sum(rc_val))
    print(status, rc_status)
    print(coord, rc_coord)
