cline=$1
indir=/users/amtseng/att_priors/data/raw/ENCODE_DNase/$cline
outdir=/users/amtseng/att_priors/data/interim/ENCODE_DNase/binary/$cline

mkdir -p $outdir
taskfile=$outdir/$cline\_labelgen_tasks.tsv

printf "Preparing the task list\n"
printf "task\tnarrowPeak\tambig\n" > $taskfile

for expid in `find $indir -name *.bed.gz -exec basename {} \; | awk -F "_" '{print $1}' | sort -u`
do
	ambibed=${indir}/${expid}_${cline}_peaks-ambi.bed.gz
	optbed=${indir}/${expid}_${cline}_peaks-idr.bed.gz

	printf "${cline}_${expid}\t$optbed\t$ambibed\n" >> $taskfile
done

printf "Running label generator\n"

genomewide_labels --task_list $taskfile \
				  --outf $outdir/$cline\_genomewide_labels.h5 \
				  --output_type hdf5 \
				  --chrom_sizes /users/amtseng/genomes/hg38.canon.chrom.sizes \
				  --bin_stride 50 \
				  --bin_size 200 \
				  --left_flank 400 \
				  --right_flank 400 \
				  --chrom_threads 10 \
				  --task_threads 4 \
				  --allow_ambiguous \
				  --save_label_source \
				  --labeling_approach peak_percent_overlap_with_bin_classification
