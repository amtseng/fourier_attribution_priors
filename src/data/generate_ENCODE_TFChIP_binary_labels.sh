tfname=$1
indir=/users/amtseng/att_priors/data/raw/ENCODE_TFChIP/$tfname/tf_chipseq
outdir=/users/amtseng/att_priors/data/interim/ENCODE_TFChIP/binary/$tfname
chromsizes=/users/amtseng/genomes/hg38.canon.chrom.sizes

bglimit=150000

mkdir -p $outdir/peaks
taskfile=$outdir/$tfname\_labelgen_tasks.tsv

printf "Preparing the task list\n"
printf "task\tnarrowPeak\tambig\n" > $taskfile

for expid in `find $indir -name *peaks*.bed.gz -exec basename {} \; | awk -F "_" '{print $1}' | sort -u`
do
	cline=$(find $indir -name $expid\_* -exec basename {} \; | cut -d "_" -f 2 | uniq)

	bgacc=$(ls $indir/$expid\_$cline\_peaks-all* | xargs basename | cut -d "." -f 1 | cut -d "_" -f 4 | uniq)
	optacc=$(ls $indir/$expid\_$cline\_peaks-optimal* | xargs basename | cut -d "." -f 1 | cut -d "_" -f 4 | uniq)

	bgbed=${indir}/${expid}_${cline}_peaks-all_${bgacc}.bed.gz
	bgtopbed=${indir}/${expid}_${cline}_peaks-all_${bgacc}.top.bed.gz
	optbed=${indir}/${expid}_${cline}_peaks-optimal_${optacc}.bed.gz
	ambibed=${outdir}/peaks/${expid}_${cline}_ambig.bed

	printf "\tLimiting background set to top $bglimit peaks\n"
	# Sort in descending order by -log(q-value)
	zcat $bgbed | sort -k9,9rn | head -n $bglimit | gzip > $bgtopbed

	printf "\tGenerating the ambiguous blacklist for $cline\n"
	# Generate the blacklist of ambiguous regions by taking the coordinates in
	# the background that are not in the "optimal" set
	bedtools intersect -a $bgtopbed -b $optbed -v > $ambibed
	printf "${expid}_${cline}\t$optbed\t$ambibed\n" >> $taskfile
done

printf "Running label generator\n"

genomewide_labels --task_list $taskfile \
				  --outf $outdir/$tfname\_genomewide_labels.h5 \
				  --output_type hdf5 \
				  --chrom_sizes $chromsizes \
				  --bin_stride 50 \
				  --bin_size 200 \
				  --left_flank 400 \
				  --right_flank 400 \
				  --chrom_threads 10 \
				  --task_threads 4 \
				  --allow_ambiguous \
				  --save_label_source \
				  --labeling_approach peak_percent_overlap_with_bin_classification

# Cleanup
rm -rf $outdir/peaks
