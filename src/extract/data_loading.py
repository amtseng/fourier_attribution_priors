import feature.util as feature_util
import feature.make_profile_dataset as make_profile_dataset
import feature.make_binary_dataset as make_binary_dataset
import pandas as pd
import numpy as np
import json


def get_input_func(
    model_type, files_spec_path, input_length, reference_fasta,
    profile_length=None
):
    """
    Returns a data function needed to run models. This data function will take
    in either coordinates or bin indices, and return the corresponding data
    needed to run the model.
    Arguments:
        `model_type`: either "binary" or "profile"
        `files_spec_path`: path to the JSON files spec for the model
        `input_length`: length of input sequence
        `reference_fasta`: path to reference fasta
        `profile_length`: if profile model, length of output profiles
    Returns a function that takes in either an array of bin indices or an array
    of coordinates, and returns data needed for the model; for a binary model,
    this function returns the N x I x 4 one-hot encoded sequences, the N x T
    array of output values, and the N x 3 array of coordinates; for a profile
    model, this function returns the one-hot sequences and the
    N x (T or 2T) x O x 2 profiles (perhaps with controls).
    """
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)

    # Maps coordinates to 1-hot encoded sequence
    coords_to_seq = feature_util.CoordsToSeq(
        reference_fasta, center_size_to_use=input_length
    )
    
    if model_type == "binary":
        # Maps bin index to values
        bins_to_vals = make_binary_dataset.BinsToVals(files_spec["labels_hdf5"])
    
        def data_func(bin_inds):
            coords, output_vals = bins_to_vals(bin_inds) 
            input_seqs = coords_to_seq(coords)
            return input_seqs, output_vals, coords

        return data_func
    else:
        # Maps coordinates to profiles
        coords_to_vals = make_profile_dataset.CoordsToVals(
            files_spec["profile_hdf5"], profile_length
        )
    
        def data_func(coords):
            input_seq = coords_to_seq(coords)
            profs = coords_to_vals(coords)
            return input_seq, np.swapaxes(profs, 1, 2)

        return data_func
        

def get_positive_inputs(model_type, files_spec_path, chrom_set=None):
    """
    Gets the set of positive coordinates or bin indices from the files specs.
    Arguments:
        `model_type`: either "binary" or "profile"
        `files_spec_path`: path to the JSON files spec for the model
        `chrom_set`: if given, limit the set of coordinates or bin indices to
            these chromosomes only
    Returns an N-array of bin indices (if binary model), or an N x 3 array of
    coordinates (if profile model).
    """
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)

    if model_type == "binary":
        labels_array = np.load(files_spec["bin_labels_npy"], allow_pickle=True)
        mask = labels_array[:, 1] == 1
        if chrom_set is not None:
            chrom_mask = np.isin(labels_array[:, 0], np.array(chrom_set))
            mask = chrom_mask & mask
        return np.where(mask)[0]
    else:
        peaks = []
        for peaks_bed in files_spec["peak_beds"]:
            table = pd.read_csv(peaks_bed, sep="\t", header=None)
            if chrom_set is not None:
                table = table[table[0].isin(chrom_set)]
            peaks.append(table.values[:, :3])
        return np.concatenate(peaks)
