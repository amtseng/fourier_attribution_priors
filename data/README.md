### `raw/`
Links to `/mnt/lab_data2/amtseng/att_priors/data/raw/`
- Raw data downloaded directly from its source
- `ENCODE_TFChIP/`
	- Contains TF-ChIPseq data fetched from ENCODE
	- `encode_tf_chip_experiments.tsv`
		- A metadata file listing various experiments from ENCODE, filtered for those that are TF-ChIPseq experiments, aligned to hg38, and status released (and not retracted, for example)
		- Downloaded with the following command, on 4 Oct 2019:
			```
			wget -O encode_tf_chip_experiments.tsv "https://www.encodeproject.org/report.tsv?type=Experiment&status=released&assay_slims=DNA+binding&assay_title=TF+ChIP-seq&assembly=GRCh38"
			```
	- `encode_control_chip_experiments.tsv`
		- A metadata file listing various control experiments from ENCODE (i.e. for a particular cell-line, a ChIP-seq experiment with no immunoprecipitation), filtered for those that are aligned to hg38 and have a status of released
		- Downloaded with the following command, on 4 Oct 2019:
			```
			wget -O encode_control_chip_experiments.tsv "https://www.encodeproject.org/report.tsv?type=Experiment&status=released&assay_slims=DNA+binding&assay_title=Control+ChIP-seq&assembly=GRCh38"
			```
	- `encode_tf_chip_files.tsv`
		- A metadata file listing various ENCODE files, filtered for those that are aligned to hg38, status released, and of the relevant output types (i.e. unfiltered alignments, called peaks, and optimal IDR-filtered peaks)
		- Downloaded with the following commands:
			```
			FIRST=1; for otype in "unfiltered alignments" "peaks and background as input for IDR" "optimal IDR thresholded peaks"; do wget -O - "https://www.encodeproject.org/report.tsv?type=File&status=released&assembly=GRCh38&output_type=$otype" | awk -v first=$FIRST '(first == 1) || NR > 2' >> encode_tf_chip_files.tsv; FIRST=2; done
			```
		- The API for the download of the experiment and files metadata is described [here](https://app.swaggerhub.com/apis-docs/encodeproject/api/basic_search/)
	- The files for each TF are downloaded by `download_ENCODE_TFChIP_data.py`

- `ENCODE_DNase/`
	- Contains metadata on DNase-seq data that has already been pre-processed by Anna, and specific downloaded files
	- Unlike TF-ChIPseq runs, most DNase-seq runs have been processed by Anna using the [ENCODE ATAC-seq pipeline](https://github.com/ENCODE-DCC/atac-seq-pipeline/)
	- The set of processed outputs can be found [here](http://mitra.stanford.edu/kundaje/projects/atlas/dnase_processed/)
		- Specifically, under `aggregate_outputs/`, we download the files specified in `bowtie2_bams`, `idr.optimal.narrowPeak`, and `ambiguous.optimal.narrowPeak`
			- Note that the BAMs are the merged replicate BAMs, unfiltered
		- These files are directly downloaded from [here](http://mitra.stanford.edu/kundaje/projects/atlas/dnase_processed/aggregate_outputs/), on 14 Feb 2020
	- The set of experiment IDs in `to_download.tsv` is curated manually
	- The download was performed by `download_ENCODE_DNase_data.py`

- `BPNet_ChIPseq/`
	- Contains the files used to train BPNet ChiP-seq (a profile model)
	- These are files that come from ChIP-seq runs in mouse (mm10) of Oct4, Sox2, and Nanog
	- The set of files used can be found [here](https://github.com/kundajelab/bpnet-manuscript/blob/master/src/bpnet-pipeline/ChIP-seq.dataspec.yml), copied from `/oak/stanford/groups/akundaje/avsec/basepair/data/processed/comparison/data/chip-seq/`
	- Note that because they are all the same cell type, there is a shared control

- `BPNet_ChIPnexus/`
	- Contains the called peaks and 5' count profile BigWigs from ChIP-nexus experiments
	- The set of files used can be found [here](https://github.com/kundajelab/bpnet-manuscript/blob/master/src/bpnet-pipeline/ChIP-nexus.dataspec.yml), copied from `/oak/stanford/groups/akundaje/avsec/basepair/data/processed/comparison/data/chip-nexus/`

- `DNase_bias/`
	- Contains a bias track for DNase, encoding the inherent biases/preferenceses of DNase I
	- Original data created by [this paper](https://academic.oup.com/nar/article/42/19/11865/2902648), and the processed using the same ENCODE ATAC-seq pipeline as above
	- Specifically, this is SRR1565781, which is DNase bias in K562
	- Data was processed by Anna using the same pipeline as above for the ENCODE DNase tracks (the tracks here are also 5' counts)
		- This processed data was copied from `/oak/stanford/groups/akundaje/projects/enzymatic_bias_correction/pipeline_out/atac/SRR1565781/call-bam2ta/shard-0/execution/SRR1565781.1.fastq.trimmed.gz.merged.nodup.no_chrM_MT.tagAlign.sorted.counts.{plus,minus}.bw`

- `DNase_footprints/`
	- [Paper](https://www.biorxiv.org/content/10.1101/2020.01.31.927798v1.full)
	- `Vierstra_et_al_2020_Consensus_footprints_0p99.motifs.bed.gz` and `Vierstra_et_al_2020_Footprints_per_sample.0q01.tar.gz` downloaded from [Zenodo](https://zenodo.org/record/3603549) on 2 Feb 2020
	- `footprints_per_sample/`
		- Un-tarred footprints per sample from `Vierstra_et_al_2020_Footprints_per_sample.0q01.tar.gz`, subsequently compressed using `gzip`

- `gc_content/`
	- Contains the GC content in the genome, for hg38 and mm10
	- These are copied from `/mnt/data/annotations/gc_content_tracks/{hg38,mm10}/gc_{hg38,mm10}_1000bp_flank.bigwig`
	- Each base in the BigWig is the GC content of the 1kb region centered at that base

- `ENCODE_TFChIP_cellline_peaks/`
	- Cell-line specific peaks for specific TFs
	- These are consist of only the optimal IDR peaks, downloaded using `download_ENCODE_TFChIP_cellline_peaks.py`

### `interim/`
Links to `/mnt/lab_data2/amtseng/att_priors/data/interim/`
- Raw data that has been processed to an intermediate form
- `ENCODE_TFChIP/`
	- `binary/`
		- For each TF, there is a subdirectory containing the HDF5s generated by `generate_ENCODE_TFChIP_binary_labels.sh` on the optimal and background peak sets in `raw/`
			- This uses `genomewide_labels` from [seqdataloader](https://github.com/kundajelab/seqdataloader/tree/master/seqdataloader)
			- 2 HDF5s are generated: one contains binary labels for each task over genome bins, and the other contains the coordinates of the peaks underlying each bin
	- `profile/`
		- There is a subdirectory for each TF; each such directory contains a set of BigWigs and peak files for that TF
			- Labels are generated by the alignments and peaks from `raw/`, using `generate_ENCODE_TFChIP_profile_labels.sh`
			- In addition to bedTools, this script also requires bedGraphToBigWig from UCSC Tools, as well as the [hg38 chromosome sizes with chrEBV included](https://github.com/ENCODE-DCC/encValData/blob/master/GRCh38/GRCh38_EBV.chrom.sizes)
			- Note that we start with the _unfiltered_ alignments from ENCODE and filter them ourselves using samtools
				- The filtering process we do is identical, but we keep duplicate reads, because those can be useful for profile prediction

- `ENCODE_DNase/`
	- `binary/`
		- For each TF, there is a subdirectory containing the HDF5s generated by `generate_ENCODE_TFChIP_binary_labels.sh` on the optimal and background peak sets in `raw/`
			- This uses `genomewide_labels` from [seqdataloader](https://github.com/kundajelab/seqdataloader/tree/master/seqdataloader)
			- 2 HDF5s are generated: one contains binary labels for each task over genome bins, and the other contains the coordinates of the peaks underlying each bin
	- `profile/`
		- There is a subdirectory for each cell line; each such directory contains the set of optimal peaks that passed IDR, the set of ambiguous peaks that did not pass the IDR test, and the set of BigWigs for the cell line
			- Labels are generated by the alignments from `raw/`, using `generate_ENCODE_DNase_profile_labels.sh`
			- In addition to bedTools, this script also requires bedGraphToBigWig from UCSC Tools, as well as the [hg38 chromosome sizes with chrEBV included](https://github.com/ENCODE-DCC/encValData/blob/master/GRCh38/GRCh38_EBV.chrom.sizes)
			- Note that we start with the _unfiltered_ alignments from ENCODE and filter them ourselves using samtools
				- The filtering process we do is identical, but we keep duplicate reads, because those can be useful for profile prediction
		- Note that unlike for TF-ChIPseq prediction, there are no matched controls

- `BPNet_ChIPseq/`
	- `binary/`
		- Contains the HDF5s generated by `generate_BPNet_binary_labels.sh` on the optimal IDR peaks in `raw/`
			- Note that no ambiguous peak set is used for BPNet
			- This uses `genomewide_labels` from [seqdataloader](https://github.com/kundajelab/seqdataloader/tree/master/seqdataloader)
			- 2 HDF5s are generated: one contains binary labels for each task over genome bins, and the other contains the coordinates of the peaks underlying each bin
	- Unlike `ENCODE_TFChIP/` and `ENCODE_DNase`, there is no `profile/` subdirectory, because the data came in BigWig form already

- `ENCODE_TFChIP_cellline_peaks/`
	- The same peak files from `raw/`, but consolidated and sorted using Bedtools

### `processed/`
Links to `/mnt/lab_data2/amtseng/att_priors/data/processed/`
- Processed data ready to train models or downstream analysis
- `chrom_splits.json`
	- This defines the chromosome splits for hg38, based on chromosome size, for appropriate training, validation, and test sets

- `ENCODE_TFChIP/`
	- `binary/`
		- Processed data ready for binary models
		- `labels/`
			- Binarized labels for each TF, which is an HDF5 of labels and NumPy arrays of bin labels and peak signal strengths
			- These files are created by `create_binary_bins.py` on the HDF5s in `interim/` (from `generate_ENCODE_TFChIP_binary_labels.sh`)
				- The HDF5 contains the same information as the bin labels HDF5 in `interim/`, but in a better, more efficient form for consumption
				- The NumPy arrays contain the separated bin-level labels and task-specific peak signal values, for ease of data loading and for subsetting bins
		- `config/`
			- Training configuration files like paths to training data in `labels/`, and parameter configurations like number of tasks
	- `profile/`
		- Processed data ready for profile models
		- `labels/`
			- HDF5 containing BigWig tracks for training, and BED files in NarrowPeak format for the peaks
				- The BED files are copied directly from `interim/`
			- Made from BigWigs in `interim/` using `create_ENCODE_TFChIP_profile_hdf5.py/`
		- `config/`
			- Training configuration files like paths to training data in `labels/`, and parameter configurations like number of tasks

- `ENCODE_DNase/`
	- `binary/`
		- Processed data ready for binary models
		- `labels/`
			- Binarized labels for each TF, which is an HDF5 of labels and NumPy arrays of bin labels and peak signal strengths
			- These files are created by `create_binary_bins.py` on the HDF5s in `interim/` (from `generate_BPNet_binary_labels.sh`)
				- The HDF5 contains the same information as the bin labels HDF5 in `interim/`, but in a better, more efficient form for consumption
				- The NumPy arrays contain the separated bin-level labels and task-specific peak signal values, for ease of data loading and for subsetting bins
		- `config/`
			- Training configuration files like paths to training data in `labels/`, and parameter configurations like number of tasks
	- `profile/`
		- Processed data ready for profile models
		- `labels/`
			- HDF5 containing BigWig tracks for training, and BED files in NarrowPeak format for the peaks
				- The BED files are copied directly from `interim/`
			- Made from BigWigs in `interim/` using `create_ENCODE_DNase_profile_hdf5.py/`
				- Also includes a control track from `raw/DNase_bias/`
		- `config/`
			- Training configuration files like paths to training data in `labels/`, and parameter configurations like number of tasks

- `BPNet_ChIPseq/`
	- `binary/`
		- Processed data ready for binary models
		- `labels/`
			- Binarized labels for BPNet data, which is an HDF5 of labels and NumPy arrays of bin labels and peak signal strengths
			- These files are created by `create_binary_bins.py` on the HDF5s in `interim/` (from `generate_BPNet_binary_labels.sh`)
				- The script was used by specifying an options for "-t" and "-a", and overriding all other paths
				- The HDF5 contains the same information as the bin labels HDF5 in `interim/`, but in a better, more efficient form for consumption
				- The NumPy arrays contain the separated bin-level labels and task-specific peak signal values, for ease of data loading and for subsetting bins
		- `config/`
			- Training configuration files like paths to training data in `labels/`, and parameter configurations like number of tasks
	- `profile/`
		- Processed data ready for profile models
		- `labels/`
			- HDF5 containing BigWig tracks for training, and BED files in NarrowPeak format for the peaks
				- The BED files are copied directly from `raw/`
			- Made from BigWigs in `raw/` using `create_BPNet_profile_hdf5.py/`
		- `config/`
			- Training configuration files like paths to training data in `labels/`, and parameter configurations like number of tasks

- `DNase_footprints/`
	- Made from combined footprints from multiple samples of the same cell line, then intersected with the consensus motif footprints
		- `bedtools intersect -a <(zcat footprints_1.bed footprints_2.bed | bedtools sort) -b <(zcat Vierstra_et_al_2020_Consensus_footprints_0p99.motifs.bed.gz | awk '{if ($5 == "") $5 = "."; print $1 "\t" $2 "\t" $3 "\t" $5}') -wa -wb | awk '{print $6 "\t" $7 "\t" $8 "\t" $9 "\t" $1 ":" $2 "-" $3}'
		- Each resulting BED file has the coordinate of the original consensus motif, with the motif cluster matches in column 4, and the original cell line footprint coordinate in column 5
	- `K562_motifmatched.bed.gz`
		- Combined from `h.K562-DS52908.bed.gz`, `K562-DS15363.bed.gz`, `K562-DS16924.bed.gz`
	- `K562_motifmatched_tencol.bed.gz`
		- `K562_motifmatched.bed.gz`, reformatted into ENCODE 10-column NarrowPeak format
			- The first 3 columns remain the same
			- Columns 4 and 5 are combined with "|" and become column 4 (name) in the NarrowPeak file
			- All other columns are made up
		- Created using the following command:
			- `zcat K562_motifmatched.bed.gz | awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "|" $5 "\t.\t.\t.\t-1\t-1\t-1"}' | gzip > K562_tencol.bed.gz`
	- `K562.bed.gz`
		- The set of all footprints, without intersection with motifs
		- Simply a concatenation and sort of `h.K562-DS52908.bed.gz`, `K562-DS15363.bed.gz`, `K562-DS16924.bed.gz`
			- Sorted and merged intervals
	- `K562_tencol.bed.gz`
		- `K562.bed.gz`, converted to 10-column NarrowPeak format
		- Created using the following command:
			- `zcat K562.bed.gz | awk '{print $1 "\t" $2 "\t" $3 "\t.\t.\t.\t.\t-1\t-1\t-1"}' | gzip > K562_tencol.bed.gz`

- `basset/`
	- Model weights downloaded from [here](https://zenodo.org/record/1466068/files/pretrained_model_reloaded_th.pth)
