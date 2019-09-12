import subprocess

def file_line_count(filepath):
    """
    Returns the number of lines in the given file. If the file is gzipped (i.e.
    ends in ".gz"), unzips it first.
    """
    if filepath.endswith(".gz"):
        cat_comm = ["zcat", filepath]
    else:
        cat_comm = ["cat", filepath]
    wc_comm = ["wc", "-l"]

    cat_proc = subprocess.Popen(cat_comm, stdout=subprocess.PIPE)
    wc_proc = subprocess.Popen(
        wc_comm, stdin=cat_proc.stdout, stdout=subprocess.PIPE
    )
    output, err = wc_proc.communicate()
    return int(output.strip())
