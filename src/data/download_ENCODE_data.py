import os
import pandas as pd
pd.options.mode.chained_assignment = None
import urllib
import json
import click

def import_experiments_table(path):
    """
    Imports table of relevant ENCODE experiments.
    """
    return pd.read_csv(path, sep="\t", skiprows=1, header=0)


def import_files_table(path):
    """
    Imports table of relevant ENCODE files.
    """
    return pd.read_csv(path, sep="\t", skiprows=1, header=0)


def join_experiments_file_table(exp_table, file_table):
    """
    Joins the experiments and files table to make a merged table, which has
    information about each experiment and the files associated with it.
    """
    file_table["Experiment"] = \
        file_table["Dataset"].str.split("/").apply(lambda l: l[2])
    merged = exp_table.merge(
        file_table, left_on="Accession", right_on="Experiment",
        suffixes=("_exp", "_file")
    )
    return merged

JSON_URL = "https://www.encodeproject.org/experiments/%s/?format=json"
def fetch_experiment_control(exp_id):
    """
    From the experiment ID, fetch the ID of the corresponding matched control.
    """
    url = JSON_URL % exp_id
    req = urllib.request.urlopen(url)
    data = json.loads(req.read())
    return data["possible_controls"][0]["@id"].split("/")[2]  # First control


def download_file(download_url, save_path):
    """
    Downloads the file with the given ENCODE URL to the given path.
    """
    url = "https://www.encodeproject.org/" + download_url
    urllib.request.urlretrieve(url, save_path)


def download_exp_files(exp_file_table, exp_id, save_dir):
    """
    Downloads some data for the given experiment. Specifically, this downloads
    the (filtered) alignment BAM for the ChIP-seq run, the set of called peaks
    (input to IDR), and the set of optimal IDR-filtered peaks. If some of these
    files do not exist, they will not be downloaded.
    The files are saved with the following format in `save_dir`:
    {experiment ID}_{cell line}_{output type}_{file ID}.{bed.gz/bam}

    For alignments, saves all replicates; for peaks, saves the versions with
    the most replicates.
    """
    out_to_keep = [
        ("alignments", "bam"),
        ("peaks and background as input for IDR", "bed narrowPeak"),
        ("optimal IDR thresholded peaks", "bed narrowPeak")
    ]
    shorten_out_type = {
        "alignments": "align",
        "peaks and background as input for IDR": "peaks-all",
        "optimal IDR thresholded peaks": "peaks-optimal"
    }

    exp_table = exp_file_table[exp_file_table["Experiment"] == exp_id]
    cell_line = exp_table["Biosample term name"].values[0]
    # For each kind of output type and file format...
    for (out_type, file_type), out_group in exp_table.groupby(
        ["Output type", "File type"]
    ):
        if (out_type, file_type) not in out_to_keep:
            continue

        out_group["Num replicates"] = (out_group["Biological replicates"]
                                       .str.split(",").apply(len))
        for _, row in out_group[
            out_group["Num replicates"] == out_group["Num replicates"].max()
        ].iterrows():
            file_id = row["Accession_file"]
            file_format = row["File Format"]
            download_link = row["Download URL"]
            out_type_short = shorten_out_type[out_type]

            save_name = "%s_%s_%s_%s.%s" % (
                exp_id, cell_line, out_type_short, file_id, file_format
            )
            if file_format == "bed":
                save_name += ".gz"  # BED files are really BED.GZ
            save_path = os.path.join(save_dir, save_name)

            print("\t%s" % save_name)
            download_file(download_link, save_path)
    

def download_tf_files(exp_file_table, cont_file_table, tf_name, save_dir):
    """
    Given a specific TF name, downloads its alignments and peaks using
    `download_exp_files`. Then fetches the matched control ID and downloads the
    control alignments (controls don't have called peaks). Saves results to
    `save_dir/tf_chipseq`, and controls to `save_dir/control_chipseq`.
    """
    tf_exp_path = os.path.join(save_dir, "tf_chipseq")
    cont_exp_path = os.path.join(save_dir, "control_chipseq")
    os.makedirs(tf_exp_path, exist_ok=True)
    os.makedirs(cont_exp_path, exist_ok=True)

    tf_exp_ids = exp_file_table[
        exp_file_table["Target of assay"] == tf_name
    ]["Experiment"]
    tf_exp_ids = list(set(tf_exp_ids))  # Uniquify

    print("Found %d experiments: %s" % (len(tf_exp_ids), ", ".join(tf_exp_ids)))

    for tf_exp_id in tf_exp_ids:
        print(tf_exp_id)
        download_exp_files(exp_file_table, tf_exp_id, tf_exp_path)
        cont_exp_id = fetch_experiment_control(tf_exp_id)  # Matched control ID
        download_exp_files(cont_file_table, cont_exp_id, cont_exp_path)
        print("")

@click.command()
@click.option(
    "--tf-name", "-t", nargs=1, required=True, help="Name of TF"
)
@click.option(
    "--save-path", "-s", nargs=1, required=True, help="Path to save directory"
)
def main(tf_name, save_path):
    base_path = "/users/amtseng/att_priors/data/raw/ENCODE/"
    exp_table_path = os.path.join(base_path, "encode_tf_chip_experiments.tsv")
    cont_table_path = os.path.join(base_path, "encode_control_chip_experiments.tsv")
    file_table_path = os.path.join(base_path, "encode_tf_chip_files.tsv")

    exp_table = import_experiments_table(exp_table_path)
    cont_table = import_experiments_table(cont_table_path)
    file_table = import_files_table(file_table_path)

    exp_file_table = join_experiments_file_table(exp_table, file_table)
    cont_file_table = join_experiments_file_table(cont_table, file_table)

    download_tf_files(exp_file_table, cont_file_table, tf_name, save_path)

if __name__ == "__main__":
    main()
