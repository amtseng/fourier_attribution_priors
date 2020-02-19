import feature.util as feature_util
import feature.make_profile_dataset as make_profile_dataset
import feature.make_binary_dataset as make_binary_dataset
import pandas as pd
import numpy as np
import json

def get_profile_input_func(
    files_spec_path, input_length, profile_length, reference_fasta
):
    """
    Returns a data function needed to run profile models. This data function
    will take in an N x 3 object array of coordinates, and return the
    corresponding data needed to run the model.
    Arguments:
        `files_spec_path`: path to the JSON files spec for the model
        `input_length`: length of input sequence
        `profile_length`: length of output profiles
        `reference_fasta`: path to reference fasta
    Returns a function that takes in an N x 3 array of coordinates, and returns
    the following: the N x I x 4 one-hot encoded sequences, and the
    N x (T or T + 1 or 2T) x O x 2 profiles (perhaps with controls).
    """
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)

    # Maps coordinates to 1-hot encoded sequence
    coords_to_seq = feature_util.CoordsToSeq(
        reference_fasta, center_size_to_use=input_length
    )
    
    # Maps coordinates to profiles
    coords_to_vals = make_profile_dataset.CoordsToVals(
        files_spec["profile_hdf5"], profile_length
    )
    
    def input_func(coords):
        input_seq = coords_to_seq(coords)
        profs = coords_to_vals(coords)
        return input_seq, np.swapaxes(profs, 1, 2)

    return input_func
        

def get_binary_input_func(files_spec_path, input_length, reference_fasta):
    """
    Returns a data function needed to run binary models. This data function will
    take in an N-array of bin indices, and return the corresponding data needed
    to run the model.
    Arguments:
        `files_spec_path`: path to the JSON files spec for the model
        `input_length`: length of input sequence
        `reference_fasta`: path to reference fasta
    Returns a function that takes in an N-array of bin indices, and returns the
    following: the N x I x 4 one-hot encoded sequences, the N x T array of
    output values, and the N x 3 object array of input sequence coordinates.
    """
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)

    # Maps coordinates to 1-hot encoded sequence
    coords_to_seq = feature_util.CoordsToSeq(
        reference_fasta, center_size_to_use=input_length
    )
    
    # Maps bin index to values
    bins_to_vals = make_binary_dataset.BinsToVals(files_spec["labels_hdf5"])
    
    def input_func(bin_inds):
        coords, output_vals = bins_to_vals(bin_inds) 
        input_seqs = coords_to_seq(coords)
        return input_seqs, output_vals, coords

    return input_func


def get_positive_profile_coords(files_spec_path, chrom_set=None):
    """
    Gets the set of positive coordinates for a profile model from the files
    specs. The coordinates consist of peaks collated over all tasks.
    Arguments:
        `files_spec_path`: path to the JSON files spec for the model
        `chrom_set`: if given, limit the set of coordinates to these chromosomes
            only
    Returns an N x 3 array of coordinates.
    """
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)

    peaks = []
    for peaks_bed in files_spec["peak_beds"]:
        table = pd.read_csv(peaks_bed, sep="\t", header=None)
        if chrom_set is not None:
            table = table[table[0].isin(chrom_set)]
        peaks.append(table.values[:, :3])
    return np.concatenate(peaks)       


def get_positive_binary_bins(files_spec_path, chrom_set=None):
    """
    Gets the set of positive bin indices from the files specs. This is all
    bins that contain at least one task which is positive in that bin.
    Arguments:
        `files_spec_path`: path to the JSON files spec for the model
        `chrom_set`: if given, limit the set bin indices to these chromosomes
            only
    Returns an N-array of bin indices.
    """
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)

    labels_array = np.load(files_spec["bin_labels_npy"], allow_pickle=True)
    mask = labels_array[:, 1] == 1
    if chrom_set is not None:
        chrom_mask = np.isin(labels_array[:, 0], np.array(chrom_set))
        mask = chrom_mask & mask
    return np.where(mask)[0]
