import subprocess
import numpy as np
import pandas as pd
from pyfaidx import Fasta
import simdna.synthetic
import simdna.pwm
import os
import tempfile

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
            on both sides to this length to get the final sequence; if this is
            smaller than the coordinate interval given, then the interval will
            be cut to this size by centering
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
        instantiated genome reader. Returns the sequence string. This may pad
        or cut from the center to a specified length.
        """
        if self.center_size_to_use:
            center = int(0.5 * (start + end))
            half_size = int(0.5 * self.center_size_to_use)
            left = center - half_size
            right = center + self.center_size_to_use - half_size
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


def import_homer_motif(motif_file):
    """
    Imports a HOMER-output motif from `motif_file`. `motif_file` must contain
    only 1 motif.
    Returns an M x 4 array, consisting of the imported PFM of length M.
    """
    pfm = []
    with open(motif_file, "r") as f:
        next(f)  # Header
        for line in f:
            freqs = [float(x) for x in line.strip().split()]
            pfm.append(freqs)
    return np.array(pfm)
  

class StatusToSimulatedSeq():
    """
    Creates an object that generates one-hot encoded simulated sequences.
    Arguments:
        `sequence_length`: Length of sequence to return
        `motif_path`: Path to motif to embed for positive sequences, in HOMER
            format
        `motif_bound`: maximum distance from sequence center to place motifs,
            for positive sequences
        `gc_prob`: Probability of G/C in background sequence; default to 50%
    """
    def __init__(self, sequence_length, motif_path, motif_bound=0, gc_prob=0.5):
        self.sequence_length = sequence_length

        # Import the motif PFM
        motif_pfm = import_homer_motif(motif_path)
        
        # Create the object that generates motif-embedded sequences
        pwm = simdna.pwm.PWM("PWM").addRows(
            motif_pfm
        ).finalise(pseudocountProb=0.00001)
        embedder = simdna.synthetic.RepeatedEmbedder(
            simdna.synthetic.SubstringEmbedder(
                simdna.synthetic.ReverseComplementWrapper(
                    substringGenerator=simdna.synthetic.PwmSampler(pwm),
                    reverseComplementProb=0.5
                ),
                positionGenerator=simdna.synthetic.InsideCentralBp(
                    len(motif_pfm) + (2 * motif_bound)
                )
            ),
            quantityGenerator=simdna.synthetic.FixedQuantityGenerator(1)
        )
        assert gc_prob >= 0 and gc_prob <= 1
        at_prob = (1.0 - gc_prob) / 2
        gc_prob = gc_prob / 2
        pos_base_freqs = {
            "A": at_prob, "C": gc_prob, "G": gc_prob, "T": at_prob
        }
        self.pos_seq_generator = simdna.synthetic.EmbedInABackground(
            backgroundGenerator=simdna.synthetic.ZeroOrderBackgroundGenerator(
                sequence_length, pos_base_freqs
            ),
            embedders=[embedder]
        )

        # Create the object that generates backgrounds with no motifs
        neg_base_freqs = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
        self.neg_seq_generator = simdna.synthetic.EmbedInABackground(
            backgroundGenerator=simdna.synthetic.ZeroOrderBackgroundGenerator(
                sequence_length, neg_base_freqs
            ),
            embedders=[]
        )

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

    def _get_one_hot_seqs(self, generator, num_to_generate):
        # Fetch all sequences
        seq_set = simdna.synthetic.GenerateSequenceNTimes(
            generator, num_to_generate
        )
        seqs = [s.seq for s in seq_set.generateSequences()]
        
        # Lay all sequences end-to-end and convert to a 1-column Pandas Series
        seqs_col = pd.Series(iter("".join(seqs)))
        
        # Convert each base in the Series to its one-hot encoding
        encs_col = np.array(seqs_col.map(self.one_hot_base_map).values.tolist())

        # Reshape the single column of one-hot encodings into separate sequences
        onehot = encs_col.reshape([num_to_generate, self.sequence_length, 4])
        return onehot

    def _get_ndarray(self, statuses, revcomp=False):
        """
        `statuses` is an N-array of 1s and 0s, denoting positives and negatives.
        Returns a NumPy array of one-hot encoded sequences parallel to
        `statuses` (i.e. wherever there is a 0, a negative sequence is returned,
        and wherever there is a 1, a positive sequence with a motif is
        returned). If `revcomp` is True, also include the reverse complement
        sequences concatenated at the end (i.e. the returned array will be
        twice as long as `statuses`).
        """
        onehot = np.empty((len(statuses), self.sequence_length, 4))
        pos_inds = np.where(statuses == 1)[0]
        neg_inds = np.where(statuses == 0)[0]

        if len(pos_inds):
            pos_onehot = self._get_one_hot_seqs(
                self.pos_seq_generator, len(pos_inds)
            )
            np.put_along_axis(
                onehot, pos_inds[:, None, None], pos_onehot, axis=0
            )

        if len(neg_inds):
            neg_onehot = self._get_one_hot_seqs(
                self.neg_seq_generator, len(neg_inds)
            )
            np.put_along_axis(
                onehot, neg_inds[:, None, None], neg_onehot, axis=0
            )
        
        if revcomp:
            rc = onehot[:, ::-1, ::-1]
            onehot = np.concatenate([onehot, rc])

        return onehot

    def __call__(self, statuses, revcomp=False):
        return self._get_ndarray(statuses, revcomp)


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
