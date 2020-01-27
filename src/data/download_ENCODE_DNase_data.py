import os
import shutil

base_path = "/users/amtseng/att_priors/data/raw/ENCODE_DNase/"
plus_bigwig_table_path = os.path.join(base_path, "count_bigwig_plus_5p")
minus_bigwig_table_path = os.path.join(base_path, "count_bigwig_minus_5p")
idr_peak_table_path = os.path.join(base_path, "idr.optimal.narrowPeak")

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
    to the IDR optimal peaks. This function requires a list of possible hashes
    because unlike the set of BigWig paths, the optimal peak paths do not
    contain the experiment ID.
    This will return a list of paths to peak files, and also a parallel list of
    hashes.
    """
    paths, hashes = [], []
    with open(idr_peak_table_path, "r") as f:
        for line in f:
            path = line.strip()
            hsh = path.split("/")[9]
            if hsh in possible_hashes:
                paths.append(path)
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
    track, and IDR optimal peaks NarrowPeaks BED, in that order.
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
        raise ValueError("No IDR optimal NarrowPeak BED found for %s" % exp_id)

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
            peak_path = peak_paths[i]

    return minus_bigwig_path, plus_bigwig_path, peak_path


if __name__ == "__main__":
    with open("to_download.tsv", "r") as f:
        next(f)  # Skip header
        for line in f:
            tokens = line.strip().split("\t")
            exp_id, cell_type = tokens[0], tokens[1]
            try:
                print("Downloading %s" % exp_id)
                minus_bigwig_source, plus_bigwig_source, peak_bed_source = \
                    get_all_download_paths(exp_id)
            except ValueError as e:
                print("\tFailure: " + str(e)) 
            
            dir_path = os.path.join(base_path, cell_type)
            os.makedirs(dir_path, exist_ok=True)

            stem = cell_type + "_" + exp_id
            minus_bigwig_dest = os.path.join(dir_path, stem + "_neg.bw")
            plus_bigwig_dest = os.path.join(dir_path, stem + "_pos.bw")
            peak_bed_dest = os.path.join(
                dir_path, stem + "_all_peakints.bed.gz"
            )

            shutil.copyfile(minus_bigwig_source, minus_bigwig_dest)
            shutil.copyfile(plus_bigwig_source, plus_bigwig_dest)
            shutil.copyfile(peak_bed_source, peak_bed_dest)
