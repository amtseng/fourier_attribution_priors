import simdna.synthetic
import simdna.pwm
import tempfile
import numpy as np
import pandas as pd
import os
import shutil
import pyfaidx
import click
import tqdm

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


def generate_center_embedded_seqs(motif_pfm, num_seqs, seq_length, gc_prob=0.5):
    """
    Generates a set of DNA sequences with the motif embedded in the center
    exactly. For each generated sequence, the motif (or its reverse complement,
    with 50% probability) will be placed exactly in the center of the sequence.
    There will be no other motifs placed.
    Arguments:
        `motif_pwm`: an M x 4 array, which is a motif PFM
        `num_seqs`: number of sequences to generate
        `seq_length`: length of sequences to generate
        `gc_prob`: probability of G/C in the background; defaults to 0.5, which
            is balanced
    Returns a list of `num_seqs` sequences as strings.
    """
    pwm = simdna.pwm.PWM("PWM").addRows(
        motif_pfm
    ).finalise(pseudocountProb=0.00001)
    embedder = simdna.synthetic.RepeatedEmbedder(
        simdna.synthetic.SubstringEmbedder(
            simdna.synthetic.ReverseComplementWrapper(
                substringGenerator=simdna.synthetic.PwmSampler(pwm),
                reverseComplementProb=0.5
            ),
            positionGenerator=simdna.synthetic.InsideCentralBp(len(motif_pfm))
        ),
        quantityGenerator=simdna.synthetic.FixedQuantityGenerator(1)
    )

    assert gc_prob >= 0 and gc_prob <= 1
    at_prob = (1.0 - gc_prob) / 2
    gc_prob = gc_prob / 2
    base_freqs = {
        "A": at_prob, "C": gc_prob, "G": gc_prob, "T": at_prob
    }
    embed_in_bg = simdna.synthetic.EmbedInABackground(
        backgroundGenerator=simdna.synthetic.ZeroOrderBackgroundGenerator(
            seq_length, base_freqs
        ),
        embedders=[embedder]
    )

    with tempfile.NamedTemporaryFile() as tempf:
        seq_set = simdna.synthetic.GenerateSequenceNTimes(embed_in_bg, num_seqs)
        simdna.synthetic.printSequences(
            tempf.name, seq_set, includeEmbeddings=True
        )
        return simdna.synthetic.read_simdata_file(tempf.name).sequences


def create_simulated_fasta(
    base_fasta_path, output_fasta_path, homer_motif_path, peak_bed_paths,
    seq_length, gc_prob=0.5, chrom_set=None
):
    """
    Creates a copy of a Fasta file, with intervals from a set of peaks replaced
    by simulated sequences. Each replaced sequence will have a single motif
    instance centered at the summit.
    Arguments:
        `base_fasta_path`: path to original Fasta to use as a template
        `output_fasta_path`: path to output the new Fasta, with peak regions
            replaced with simulated sequences
        `homer_motif_path`: path to a single motif in HOMER format
        `peak_bed_paths`: list of paths to BED files in NarrowPeak format
        `seq_length`: the length of sequence to replace
        `gc_prob`: probability of having a G or C in the background in the
            generated sequences
        `chrom_set`: if specified, only replace peaks in these chromosomes
    """
    motif_pfm = import_homer_motif(homer_motif_path)

    # Import the set of peaks
    peaks = []
    for peak_bed_path in peak_bed_paths:
        table = pd.read_csv(
            peak_bed_path, sep="\t", header=None,  # Infer compression
            names=[
                "chrom", "peak_start", "peak_end", "name", "score",
                "strand", "signal", "pval", "qval", "summit_offset"
            ]
        )

        # Filter for chromosome
        if chrom_set:
            table = table[table["chrom"].isin(chrom_set)]

        # Add column for actual summit position
        table["summit_pos"] = table["peak_start"] + table["summit_offset"]
        peaks.append(table)
    peaks = pd.concat(peaks)  # Concatenate into single table

    # Get the set of simulated sequences
    print("Generating simulated sequences...")
    sim_seqs = generate_center_embedded_seqs(
        motif_pfm, len(peaks), seq_length, gc_prob=gc_prob
    )

    # To avoid overwriting previous motif instances with the simulated sequences
    # of newer peaks, we perform two passes. In the first pass, we put in the
    # simulated sequences, regardless of overlap. In the second pass, we then
    # fill in the motifs
   
    print("Making a copy of the template...")
    # First, make a copy of the template base Fasta
    outdir = os.path.dirname(output_fasta_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    shutil.copyfile(base_fasta_path, output_fasta_path)

    print("Generating index...")
    # Opening the Fasta file using pyfaidx generates the index
    f = pyfaidx.Fasta(output_fasta_path, mutable=True)
    
    # For each peak, write in the simulated sequence
    i = 0
    for _, row in tqdm.tqdm(
        peaks.iterrows(), desc="Filling in background", total=len(peaks)
    ):
        chrom, summit = row["chrom"], row["summit_pos"]
        sim_seq = sim_seqs[i]
        start = summit - (seq_length // 2)
        # Using an if-statement instead of try/except makes this loop extremely
        # slow for some reason
        try:
            f[chrom][start : start + seq_length] = sim_seq
        except OSError:
            # Didn't fit
            pass
        finally:
            i += 1

    # Location of motif in each simulated sequence
    motif_len = len(motif_pfm)
    motif_pos = (seq_length // 2) - (motif_len // 2)

    # Now write in the motif for each peak
    i = 0
    for _, row in tqdm.tqdm(
        peaks.iterrows(), desc="Filling in motifs", total=len(peaks)
    ):
        chrom, summit = row["chrom"], row["summit_pos"]
        sim_seq = sim_seqs[i]
        motif_instance = sim_seq[motif_pos : motif_pos + motif_len]
        start = summit - (motif_len // 2)
        f[chrom][start : start + motif_len] = motif_instance
        i += 1

    print("Regenerating index...")
    f.close()
    f = pyfaidx.Fasta(output_fasta_path)
    f.close()


@click.command()
@click.option(
    "--base-fasta", "-b", default="/users/amtseng/genomes/hg38.fasta",
    help="Base template Fasta to use"
)
@click.option(
    "--out-fasta", "-o", required=True, help="Where to store new Fasta"
)
@click.option(
    "--motif-path", "-m", required=True,
    help="Path to single motif file in HOMER format"
)
@click.option(
    "--length", "-l", required=True, type=int,
    help="Length of sequence to replace, centered at summits"
)
@click.option(
    "--gc-prob", "-gc", default=0.5, type=float,
    help="Probability of G/C in simulated backgrounds"
)
@click.option(
    "--chrom-set", "-c", default=None,
    help="A comma-separated list of chromosomes to replace peaks for; defaults to all chromosomes"
)
@click.argument("peak_bed_paths", nargs=-1)
def main(
    base_fasta, out_fasta, motif_path, length, gc_prob, chrom_set,
    peak_bed_paths
):
    """
    Creates a copy of a genome Fasta, with peak regions replaced with simulated
    sequences with a motif at the summit.
    """
    if chrom_set:
        chrom_set = chrom_set.split(",")

    create_simulated_fasta(
        base_fasta, out_fasta, motif_path, peak_bed_paths, length,
        gc_prob=gc_prob, chrom_set=chrom_set
    )


if __name__ == "__main__":
    main()
