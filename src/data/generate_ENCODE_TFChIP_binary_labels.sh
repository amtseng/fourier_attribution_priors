set -beEo pipefail

tfname=$1
indir=/users/amtseng/att_priors/data/raw/ENCODE_TFChIP/$tfname/tf_chipseq
outdir=/users/amtseng/att_priors/data/processed/ENCODE_TFChIP/binary/labels/$tfname

bglimit=150000

scriptdir=$(dirname $0)

mkdir -p $outdir/peaks
taskfile=$outdir/labelgen_tasks.tsv

printf "Preparing the task list\n"
printf "" > $taskfile

for expid in `find $indir -name *peaks*.bed.gz -exec basename {} \; | awk -F "_" '{print $1}' | sort -u`
do
	cline=$(find $indir -name $expid\_* -exec basename {} \; | cut -d "_" -f 2 | uniq)

	bgacc=$(ls $indir/$expid\_$cline\_peaks-all* | xargs basename | cut -d "." -f 1 | cut -d "_" -f 4 | uniq)
	optacc=$(ls $indir/$expid\_$cline\_peaks-optimal* | xargs basename | cut -d "." -f 1 | cut -d "_" -f 4 | uniq)

	bgbed=${indir}/${expid}_${cline}_peaks-all_${bgacc}.bed.gz
	bgcanbed=${outdir}/peaks/${expid}_${cline}_peaks-all_${bgacc}.canon.bed.gz
	optbed=${indir}/${expid}_${cline}_peaks-optimal_${optacc}.bed.gz
	optcanbed=${outdir}/peaks/${expid}_${cline}_peaks-optimal_${optacc}.canon.bed.gz
	ambibed=${outdir}/peaks/${expid}_${cline}_ambig.bed

	printf "\tLimiting background set to top $bglimit peaks, and keeping only canonical contigs\n"
	zcat $bgbed | head -n $bglimit | awk '$1 ~ /^(chr1|chr2|chr3|chr4|chr5|chr6|chr7|chr8|chr9|chr10|chr11|chr11|chr12|chr13|chr14|chr15|chr16|chr17|chr18|chr19|chr20|chr21|chr22|chrX|chrY|chrM)$/' | gzip > $bgcanbed
	zcat $optbed | awk '$1 ~ /^(chr1|chr2|chr3|chr4|chr5|chr6|chr7|chr8|chr9|chr10|chr11|chr11|chr12|chr13|chr14|chr15|chr16|chr17|chr18|chr19|chr20|chr21|chr22|chrX|chrY|chrM)$/' | gzip > $optcanbed

	printf "\tGenerating the ambiguous blacklist for $cline\n"
	# Generate the blacklist of ambiguous regions by taking the coordinates in
	# the background that are not in the "optimal" set
	bedtools intersect -a $bgcanbed -b $optcanbed -v > $ambibed
	printf "${expid}_${cline}\t$optcanbed\t\t$ambibed\n" >> $taskfile
done

printf "Running label generator\n"

genomewide_labels --task_list $taskfile \
				  --outf $outdir/labels.hdf5 \
				  --output_type hdf5 \
				  --chrom_sizes /users/amtseng/genomes/hg38.canon.chrom.sizes \
				  --bin_stride 50 \
				  --bin_size 200 \
				  --left_flank 400 \
				  --right_flank 400 \
				  --chrom_threads 10 \
				  --task_threads 4 \
				  --allow_ambiguous \
				  --labeling_approach peak_percent_overlap_with_bin_classification

echo "Converting HDF5 to gzipped BED files..."
python $scriptdir/hdf5_to_bed.py -i $outdir/labels.hdf5 -o $outdir/$tfname\_all_labels.bed.gz -z

echo "Splitting into training and validation..."
zcat $outdir/$tfname\_all_labels.bed.gz | awk '$1 ~ /^(chr|chr2|chr3|chr4|chr5|chr6|chr7|chr9|chr11|chr12|chr13|chr14|chr15|chr16|chr17|chr18|chr19|chr20|chr22|chrX|chrY|chrM)$/' | gzip > $outdir/$tfname\_train_labels.bed.gz
zcat $outdir/$tfname\_all_labels.bed.gz | awk '$1 ~ /^(chr|chr1|chr8|chr21)$/' | gzip > $outdir/$tfname\_val_labels.bed.gz

# Cleanup
rm -rf $outdir/peaks
rm $taskfile
rm $outdir/labels.hdf5
