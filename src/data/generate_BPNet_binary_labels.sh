indir=/users/amtseng/att_priors/data/raw/BPNet_ChIPseq
outdir=/users/amtseng/att_priors/data/interim/BPNet_ChIPseq/binary

mkdir -p $outdir
taskfile=$outdir/BPNet_labelgen_tasks.tsv

printf "Preparing the task list\n"
printf "task\tnarrowPeak\n" > $taskfile

for tfname in `find $indir -name *.bed.gz -exec basename {} \; | awk -F "_" '{print $2}' | sort -u`
do
	optbed=${indir}/BPNet_${tfname}_peaks-idr.bed.gz
	printf "${tfname}\t$optbed\n" >> $taskfile
done

printf "Running label generator\n"

genomewide_labels --task_list $taskfile \
				  --outf $outdir/BPNet_genomewide_labels.h5 \
				  --output_type hdf5 \
				  --chrom_sizes /users/amtseng/genomes/mm10.canon.chrom.sizes \
				  --bin_stride 50 \
				  --bin_size 200 \
				  --left_flank 400 \
				  --right_flank 400 \
				  --chrom_threads 10 \
				  --task_threads 4 \
				  --save_label_source \
				  --labeling_approach peak_percent_overlap_with_bin_classification
