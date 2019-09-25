import os
import pandas as pd
import subprocess

BASE_PATH = "/users/amtseng/att_priors/data/raw/ENCODE/"
DATA_TO_USE_PATH = os.path.join(BASE_PATH, "ENCODE_data_to_use.tsv")
DESTINATION_BASE = BASE_PATH

# Read in the table of data to download
table = pd.read_csv(DATA_TO_USE_PATH, sep="\t", header=0, index_col=False)

output_type_map = {
    "peaks and background as input for IDR": "peaks-bg",
    "optimal idr thresholded peaks": "optimal"
}

# Grouping by TF name, make the subdirectories and build paths
accessions, paths = [], []
for tf_name, tf_table in table.groupby("target_label"):
    save_dir = os.path.join(DESTINATION_BASE, tf_name)
    os.makedirs(save_dir, exist_ok=True)
    for exp_name, exp_table in tf_table.groupby("dataset"):
        exp_id = exp_name.split("/")[2]
        for _, row in exp_table.iterrows():
            output_type = row["output_type"]
            output_type_short = output_type_map[output_type]
            cell_line = row["biosample_short_name"]
            accession = row["accession"]
            fname = "%s_%s_%s_%s.bed.gz" % (exp_id, cell_line, output_type_short, accession)
            save_path = os.path.join(save_dir, fname)
            accessions.append(accession)
            paths.append(save_path)

# Download all the requested accessions to the same place, manually
print("At this point, copy all these files to " + DESTINATION_BASE + ":")
print(" ".join([acc + ".bed.gz" for acc in accessions]))
x = input("Done? [y/N]")
while x.lower() != "y":
    x = input("Done? [y/N]")

# Move the files to their final locations
for acc, path in zip(accessions, paths):
    os.rename(os.path.join(DESTINATION_BASE, acc + ".bed.gz"), path)
