import subprocess
import numpy as np
import pandas as pd
from pyfaidx import Fasta

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


class CoordsToSeq():
    """
    Creates an object that converts genome coordinates into a one-hot encoded
    sequence.
    Fetching sequences is thread-safe because each thread running `_get_ndarray`
    will have its own memory-mapped Fasta reader object.
    Arguments:
        `genome_fasta_path`: Path to the genome assembly FASTA
        `center_size_to_use`: For each genomic coordinate, center it and pad it
            on both sides to this length to get the final sequence
    """
    def __init__(self, genome_fasta_path, center_size_to_use=None):
        self.center_size_to_use = center_size_to_use
        self.genome_fasta_path = genome_fasta_path
        self.one_hot_base_dict = {
            "A": np.array([1, 0, 0, 0], dtype=np.float64),
            "C": np.array([0, 1, 0, 0], dtype=np.float64),
            "G": np.array([0, 0, 1, 0], dtype=np.float64),
            "T": np.array([0, 0, 0, 1], dtype=np.float64),
            "a": np.array([1, 0, 0, 0], dtype=np.float64),
            "c": np.array([0, 1, 0, 0], dtype=np.float64),
            "g": np.array([0, 0, 1, 0], dtype=np.float64),
            "t": np.array([0, 0, 0, 1], dtype=np.float64),
        }
        self.one_hot_base_map = lambda b: self.one_hot_base_dict.get(
            b, np.array([0, 0, 0, 0])  # Default to all 0s if base not found
        )

    def _get_seq(self, chrom, start, end, gen_reader):
        """
        Fetches the FASTA sequence for the given coordinates, with an
        instantiated genome reader. Returns the sequence string.
        This function performs the necessary padding to map from a coordinate
        to a full sequence to be featurized.
        """
        if self.center_size_to_use:
            center = int(0.5 * (start + end))
            pad = int(0.5 * self.center_size_to_use)
            left = center - pad
            right = center + self.center_size_to_use - pad
            return gen_reader[chrom][left:right].seq
        else:
            return gen_reader[chrom][start:end].seq
    
    def _get_ndarray(self, coords, revcomp=False):
        """
        `coords` is an iterable of coordinate tuples (e.g. a list of tuples,
        or a partial Pandas MultiIndex).
        Returns a NumPy array of one-hot encoded sequences. Note that if
        `center_size_to_use` is not specified, then all Coordinates must be of
        the same length. If `revcomp` is True, also include the reverse
        complement sequences (concatenated at the end).
        """
        # Create a new Fasta reader; this way, every thread that runs this
        # function gets its own reader; otherwise, garbage is returned
        gen_reader = Fasta(self.genome_fasta_path)

        # Fetch all sequences
        seqs = [self._get_seq(c[0], c[1], c[2], gen_reader) for c in coords]

        # Lay all sequences end-to-end and convert to a 1-column Pandas Series
        seqs_col = pd.Series(iter("".join(seqs)))
        
        # Convert each base in the Series to its one-hot encoding
        encs_col = np.array(seqs_col.map(self.one_hot_base_map).values.tolist())

        # Reshape the single column of one-hot encodings into separate sequences
        onehot = encs_col.reshape([len(coords), self.center_size_to_use, 4])

        if revcomp:
            rc = onehot[:, ::-1, ::-1]
            onehot = np.concatenate([onehot, rc])

        return onehot

    def __call__(self, coords, revcomp=False):
        return self._get_ndarray(coords, revcomp)


def one_hot_to_seq(one_hot):
    """
    Converts a one-hot encoded sequence into its original bases.
    """
    bases = "ACGT"
    seq = ""
    for arr in one_hot:
        ind = np.where(arr)[0]
        if ind.size:
            seq += bases[ind[0]]
        else:
            seq += "N"
    return seq
