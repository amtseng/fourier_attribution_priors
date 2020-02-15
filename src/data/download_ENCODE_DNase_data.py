import os
import shutil

raw_base_path = "/users/amtseng/att_priors/data/raw/ENCODE_DNase/"
merged_bam_table_path = os.path.join(raw_base_path, "bowtie2_bams")
idr_peak_table_path = os.path.join(raw_base_path, "idr.optimal.narrowPeak")
ambi_peak_table_path = os.path.join(raw_base_path, "ambiguous.optimal.narrowPeak")
to_download_path = os.path.join(raw_base_path, "to_download.tsv")

def get_bam_path(exp_id):
    """
    From an ENCODE experiment ID, gets the path to the merged unfiltered BAMs
    BAMs for that experiment, and the hash of the corresponding run.
    """
    with open(merged_bam_table_path, "r") as f:
        for line in f:
            path = line.strip()
            tokens = path.split("/")
            hsh, file_name = tokens[9], tokens[13]
            file_exp_id = file_name.split(".")[0]
            if file_exp_id == exp_id:
                return path, hsh
    return None, None


def get_peak_paths(hsh):
    """
    From the hash for an experiment, fetch the path to the IDR optimal peaks and
    the ambiguous peaks. This function requires a the hash because unlike the
    BAM paths, the peak paths do not contain the experiment ID. 
    """
    idr_path = None
    with open(idr_peak_table_path, "r") as f:
        for line in f:
            path = line.strip()
            if hsh == path.split("/")[9]:
                idr_path = path

    ambi_path = None
    with open(ambi_peak_table_path, "r") as f:
        for line in f:
            path = line.strip()
            if hsh == path.split("/")[9]:
                ambi_path = path

    return idr_path, ambi_path


def get_all_download_paths(exp_id):
    """
    From an ENCODE experiment ID, gets the set of download paths for the merged
    unfiltered BAM and the peak files. This will call `get_bam_path` and
    `get_peak_paths`, and determind the paths to each of the files to download,
    all from the same pipeline run. If a file is missing, a ValueError will be
    raised.
    Returns the paths to the merged unfiltered BAM, IDR optimal peaks
    NarrowPeaks BED, and ambiguous peaks NarrowPeaks BED, in that order.
    """
    bam_path, hsh = get_bam_path(exp_id)
    if not bam_path:
        raise ValueError("Did not find unfiltered merged BAM for %s" % exp_id)

    idr_peak_path, ambi_peak_path = get_peak_paths(hsh)
    if not idr_peak_path:
        raise ValueError(
            "No IDR optimal NarrowPeak BEDs found for %s" % exp_id
        )
    if not ambi_peak_path:
        raise ValueError(
            "No ambiguous NarrowPeak BEDs found for %s" % exp_id
        )

    return bam_path, idr_peak_path, ambi_peak_path


if __name__ == "__main__":
    with open(to_download_path, "r") as f:
        next(f)  # Skip header
        for line in f:
            tokens = line.strip().split("\t")
            exp_id, cell_type = tokens[0], tokens[1]
            try:
                print("Downloading %s" % exp_id)
                bam_source, idr_peak_bed_source, ambi_peak_bed_source = \
                    get_all_download_paths(exp_id)
            except ValueError as e:
                print("\tFailure: " + str(e))
                continue
            
            dest_dir_path = os.path.join(raw_base_path, cell_type)
            os.makedirs(dest_dir_path, exist_ok=True)

            stem = exp_id + "_" + cell_type

            bam_dest = os.path.join(dest_dir_path, stem + "_merged.bam")
            idr_peak_bed_dest = os.path.join(
                dest_dir_path, stem + "_peaks-idr.bed.gz"
            )
            ambi_peak_bed_dest = os.path.join(
                dest_dir_path, stem + "_peaks-ambi.bed.gz"
            )
            
            # Copy the files
            shutil.copyfile(bam_source, bam_dest)
            shutil.copyfile(bam_source + ".bai", bam_dest + ".bai")  # Index
            shutil.copyfile(idr_peak_bed_source, idr_peak_bed_dest)
            shutil.copyfile(ambi_peak_bed_source, ambi_peak_bed_dest)
