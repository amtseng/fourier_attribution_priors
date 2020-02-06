import os
import shutil

raw_base_path = "/users/amtseng/att_priors/data/raw/ENCODE_DNase/"
interim_base_path = "/users/amtseng/att_priors/data/interim/ENCODE_DNase/"
plus_bigwig_table_path = os.path.join(raw_base_path, "count_bigwig_plus_5p")
minus_bigwig_table_path = os.path.join(raw_base_path, "count_bigwig_minus_5p")
idr_peak_table_path = os.path.join(raw_base_path, "idr.optimal.narrowPeak")
ambi_peak_table_path = os.path.join(raw_base_path, "ambiguous.optimal.narrowPeak")
to_download_path = os.path.join(raw_base_path, "to_download.tsv")

def get_bigwig_paths(exp_id, strand="+"):
    """
    From an ENCODE experiment ID, gets the set of paths to the BigWigs for that
    strand. Note that there can be multiple BigWigs for each experiment, becuase
    the pipeline is sometimes run more than once due to failure to run to
    completion (e.g. preempted or OOM), yielding to the same BigWigs being
    generated multiple times. All BigWigs should be identical.
    This will return a list of paths to BigWigs for the strand, and also a
    parallel list of hashes (these hashes will point to the directory where
    this pipeline is run).
    """
    assert strand in ("+", "-")
    table_path = plus_bigwig_table_path if strand == "+" \
        else minus_bigwig_table_path

    paths, hashes = [], []
    with open(table_path, "r") as f:
        for line in f:
            path = line.strip()
            tokens = path.split("/")
            hsh, file_name = tokens[9], tokens[13]
            file_exp_id = file_name.split(".")[0]
            if file_exp_id == exp_id:
                paths.append(path)
                hashes.append(hsh)
    return paths, hashes


def get_peak_paths(possible_hashes):
    """
    From a list of possible hashes for an experiment, fetch a list of paths
    to the IDR optimal peaks and the ambiguous peaks. This function requires a
    list of possible hashes because unlike the set of BigWig paths, the peak
    paths do not contain the experiment ID.
    This will return a list of paired paths to peak files (optimal, ambiguous)
    and also a parallel list of hashes. The lists will be empty if none of the
    provided hashes correspond to both an optimal peak path and ambiguous peak
    path.
    """
    paths, hashes = [], []

    idr_paths = {}
    with open(idr_peak_table_path, "r") as f:
        for line in f:
            path = line.strip()
            hsh = path.split("/")[9]
            if hsh in possible_hashes:
                idr_paths[hsh] = path

    ambi_paths = {}
    with open(ambi_peak_table_path, "r") as f:
        for line in f:
            path = line.strip()
            hsh = path.split("/")[9]
            if hsh in possible_hashes:
                ambi_paths[hsh] = path

    # Fetch the paths where the hash has both an optimal and ambiguous peak set
    for hsh in set(idr_paths.keys()) & set(ambi_paths.keys()):
        paths.append((idr_paths[hsh], ambi_paths[hsh]))
        hashes.append(hsh)
    return paths, hashes


def get_all_download_paths(exp_id):
    """
    From an ENCODE experiment ID, gets the set of download paths for the BigWigs
    and peak files. This will call `get_bigwig_paths` and `get_peak_paths`, and
    determine the paths to each of the files to download, all from the same
    pipeline run. If a file is missing, or a single run cannot be found which
    has all needed files, a ValueError will be raised.
    Returns the paths to the minus strand BigWig track, plus strand BigWig
    track, IDR optimal peaks NarrowPeaks BED, and ambiguous peaks NarrowPeaks
    BED, in that order.
    """
    minus_bigwig_paths, minus_hashes = get_bigwig_paths(exp_id, "-")
    if not minus_hashes:
        raise ValueError("No minus strand BigWigs found for %s" % exp_id)

    plus_bigwig_paths, plus_hashes = get_bigwig_paths(exp_id, "+")
    if not plus_hashes:
        raise ValueError("No plus strand BigWigs found for %s" % exp_id)
    possible_hashes = list(set(plus_hashes) & set(minus_hashes))
    if not possible_hashes:
        raise ValueError("No consistent pipeline run found for %s" % exp_id)

    peak_paths, peak_hashes = get_peak_paths(possible_hashes)
    if not peak_hashes:
        raise ValueError(
            "No IDR/ambiguous optimal NarrowPeak BEDs found for %s" % exp_id
        )

    # Now find the paths that corresponds to a hash that exists for all three
    # Pick one arbitrarily
    shared_hash = list(set(possible_hashes) & set(peak_hashes))[0]
    for i, hsh in enumerate(minus_hashes):
        if hsh == shared_hash:
            minus_bigwig_path = minus_bigwig_paths[i]
    for i, hsh in enumerate(plus_hashes):
        if hsh == shared_hash:
            plus_bigwig_path = plus_bigwig_paths[i]
    for i, hsh in enumerate(peak_hashes):
        if hsh == shared_hash:
            opt_peak_path, ambi_peak_path = peak_paths[i]

    return minus_bigwig_path, plus_bigwig_path, opt_peak_path, ambi_peak_path


if __name__ == "__main__":
    with open(to_download_path, "r") as f:
        next(f)  # Skip header
        for line in f:
            tokens = line.strip().split("\t")
            exp_id, cell_type = tokens[0], tokens[1]
            try:
                print("Downloading %s" % exp_id)
                minus_bigwig_source, plus_bigwig_source, opt_peak_bed_source, \
                    ambi_peak_bed_source = get_all_download_paths(exp_id)
            except ValueError as e:
                print("\tFailure: " + str(e))
                continue
            
            profile_dir_path = os.path.join(
                interim_base_path, "profile", cell_type
            )
            binary_dir_path = os.path.join(
                interim_base_path, "binary", cell_type
            )
            os.makedirs(profile_dir_path, exist_ok=True)
            os.makedirs(binary_dir_path, exist_ok=True)

            stem = cell_type + "_" + exp_id
            
            # For profile models, we need the profiles and only optimal peaks
            minus_bigwig_dest = os.path.join(profile_dir_path, stem + "_neg.bw")
            plus_bigwig_dest = os.path.join(profile_dir_path, stem + "_pos.bw")
            opt_peak_bed_dest_profile = os.path.join(
                profile_dir_path, stem + "_all_peakints.bed.gz"
            )
            
            # For binary models, we keep the optimal and ambiguous peaks
            opt_peak_bed_dest_binary = os.path.join(
                binary_dir_path, stem + "_optimal.bed.gz"
            )
            ambi_peak_bed_dest = os.path.join(
                binary_dir_path, stem + "_ambiguous.bed.gz"
            )

            # Copy the files
            shutil.copyfile(minus_bigwig_source, minus_bigwig_dest)
            shutil.copyfile(plus_bigwig_source, plus_bigwig_dest)
            shutil.copyfile(opt_peak_bed_source, opt_peak_bed_dest_profile)
            shutil.copyfile(opt_peak_bed_source, opt_peak_bed_dest_binary)
            shutil.copyfile(ambi_peak_bed_source, ambi_peak_bed_dest)
