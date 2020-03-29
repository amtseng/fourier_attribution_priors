#!/usr/bin/env python

# Copied directly from https://github.com/kundajelab/interpret-benchmark/blob/b1a2e40bdb84b21c83bc9ca197762745079b7bfa/scripts/homer2meme
# Authored by Avanti Shrikumar

from __future__ import division, print_function, absolute_import
import argparse
import numpy as np


class HomerPwm(object):

    def __init__(self, letter_prob_matrix, motif_name, best_guess,
                       detection_threshold, logpval, 
                       target_num, target_frac, bg_num, bg_frac,
                       enrichment_pval):
        self.letter_prob_matrix = letter_prob_matrix
        self.motif_name = motif_name
        self.best_guess = best_guess
        self.detection_threshold = detection_threshold
        self.logpval = logpval
        self.target_num = target_num
        self.target_frac = target_frac
        self.bg_num = bg_num
        self.bg_frac = bg_frac
        self.enrichment_pval = enrichment_pval

    @classmethod
    def read_homer_motif(cls, motif_file):
        fh = open(motif_file) 
        letter_prob_matrix = []
        for idx,line in enumerate(fh):
            if (idx==0):
                (logo,motif_name_n_best_guess,
                 detection_threshold, logpval,_,
                 matchdata) = line.rstrip().split("\t") 
                motif_name = motif_name_n_best_guess.split(",")[0]
                best_guess = ",".join(motif_name_n_best_guess.split(",")[1:])
                (target_match_data,bg_match_data,enrichment_pval) =\
                 matchdata.split(",")
                detection_threshold = float(detection_threshold)
                logpval = float(logpval)
                target_frac = float(((target_match_data
                                       .split(":")[1]).split("(")[1])[:-2])/100
                target_num = float((target_match_data
                                  .split(":")[1]).split("(")[0])
                bg_frac = float(((bg_match_data
                                  .split(":")[1]).split("(")[1])[:-2])/100
                bg_num = float((bg_match_data.split(":")[1]).split("(")[0])
                enrichment_pval = float(enrichment_pval[2:])
            else: 
                letter_prob_matrix.append([float(x) for x in line.split("\t")])
        return cls(letter_prob_matrix=np.array(letter_prob_matrix),
            motif_name=motif_name,
            best_guess=best_guess,
            detection_threshold=detection_threshold,
            logpval=logpval, target_num=target_num, target_frac=target_frac,
            bg_num=bg_num, bg_frac=bg_frac,
            enrichment_pval=enrichment_pval) 

    def write_in_meme_format(self, out_file_handle):
        out_file_handle.write("MOTIF "+self.motif_name
                              +" "+self.best_guess+"\n") 
        out_file_handle.write("letter-probability matrix:\n")
        for row in self.letter_prob_matrix:
            out_file_handle.write(" ") 
            out_file_handle.write(("  ".join([str(x) for x in row]))+"\n")
        out_file_handle.write("\n")


def write_meme_motifs(motifs, output_file):
    out_fh = open(output_file, 'w') 
    out_fh.write("MEME version 4\n\n")
    out_fh.write("ALPHABET= ACGT\n\n")
    out_fh.write("strands: + -\n\n")
    out_fh.write("Background letter frequencies\n")
    out_fh.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
    for motif in motifs:
        motif.write_in_meme_format(out_file_handle=out_fh)
    out_fh.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('motif_files', nargs='+') 
    parser.add_argument('--output_file', required=True) 
    parser.add_argument('--max_pval', type=float, default=1) 
    parser.add_argument('--min_target_frac', type=float, default=0.0) 
    args = parser.parse_args()
    motifs = []
    for motif_file in args.motif_files: 
        motif = HomerPwm.read_homer_motif(motif_file)
        if (motif.enrichment_pval < args.max_pval and
             motif.target_frac > args.min_target_frac):
            motifs.append(motif) 
        else:
            print("Skipping "+str(motif.motif_name)
                             +","+str(motif.best_guess)
                  +" with pval "+str(motif.enrichment_pval)
                  +" and min target frac "+str(motif.target_frac))
    write_meme_motifs(motifs=motifs, output_file=args.output_file)
