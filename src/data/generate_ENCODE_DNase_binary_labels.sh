cline=$1
indir=/users/amtseng/att_priors/data/interim/ENCODE_DNase/binary/$cline
outdir=/users/amtseng/att_priors/data/processed/ENCODE_DNase/binary/labels/$cline

mkdir -p $outdir/peaks
taskfile=$outdir/labelgen_tasks.tsv

printf "Preparing the task list\n"
printf "" > $taskfile

for expid in `find $indir -name *.bed.gz -exec basename {} \; | awk -F "_" '{print $2}' | sort -u`
do
	ambibed=${indir}/${cline}_${expid}_ambiguous.bed.gz
	optbed=${indir}/${cline}_${expid}_optimal.bed.gz

	printf "${cline}_${expid}\t$optbed\t\t$ambibed\n" >> $taskfile
done

printf "Running label generator\n"

genomewide_labels --task_list $taskfile \
				  --outf $outdir/$cline\_labels.h5 \
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

# Cleanup
rm -rf $outdir/peaks
rm $taskfile
