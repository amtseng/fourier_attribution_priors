set -beEo pipefail

tfname=$1
tfindir=/users/amtseng/att_priors/data/raw/ENCODE_TFChIP/$tfname/tf_chipseq
contindir=/users/amtseng/att_priors/data/raw/ENCODE_TFChIP/$tfname/control_chipseq
outdir=/users/amtseng/att_priors/data/interim/ENCODE_TFChIP/profile/$tfname

chromsizes=/users/amtseng/genomes/hg38.with_ebv.chrom.sizes

tempdir=$outdir/temp
mkdir -p $tempdir

matchedconts=()

# Iterate through the TF ChIPseq experiments/cell lines
# Focus on those with an alignment, peaks, and a control
expidclines=$(find $tfindir -name *.bam -exec basename {} \; | awk -F "_" '{print $1 "_" $2}' | sort -u)
for expidcline in $expidclines
do
	echo "Processing TF ChIPseq experiment $expidcline ..."
	tfaligns=$(find $tfindir -name $expidcline\_align-unfilt_*)
	tfpeaksopt=$(find $tfindir -name $expidcline\_peaks-optimal_*)
	cline=$(echo $expidcline | cut -d "_" -f 2)
	contaligns=$(find $contindir -name *_$cline\_align-unfilt_*)
	
	if [[ -z $tfaligns ]] || [[ -z $contaligns ]] || [[ -z $tfpeaksopt ]]
	then
		printf "\tDid not find all required alignments, peaks, and control alignments\n"
		continue
	fi

	numtfpeaksopt=$(echo "$tfpeaksopt" | wc -l)
	if [[ $numtfpeaksopt -gt 1 ]]
	then
		printf "\tFound more than one set of optimal peaks\n"
		continue
	fi

	contexpidclines=$(find $contindir -name *_$cline\_align-unfilt_* -exec basename {} \; | awk -F "_" '{print $1 "_" $2}')
	for item in $contexpidclines
	do
		matchedconts+=("$item")
	done

	# 1) Convert TF ChIP-seq alignment BAMs to BigWigs
	# 1.1) Filter BAM alignments for quality and mappability
	printf "\tFiltering BAM files\n"
	for tfalign in $tfaligns
	do
		name=$(basename $tfalign)
		samtools view -F 780 -q 30 -b $tfalign -o $tempdir/$name.filt
	done
	tfalignsfilt=$(find $tempdir -name *.bam.filt)

	# 1.2) Merge filtered replicate BAM files
	printf "\tMerging TF ChIP-seq replicate BAMs\n"
	numtfaligns=$(echo "$tfalignsfilt" | wc -l)
	if [[ $numtfaligns -gt 1 ]]
	then
		samtools merge $tempdir/tfaligns_$expidcline\_merged.bam $tfalignsfilt
	else
		ln -s $tfalignsfilt $tempdir/tfaligns_$expidcline\_merged.bam
	fi
	samtools index $tempdir/tfaligns_$expidcline\_merged.bam

	# 1.3) Split BAM into + and - strands in BedGraphs
	printf "\tSplitting BAM into BedGraphs by strand\n"
	bedtools genomecov -5 -bg -strand + -g $chromsizes -ibam $tempdir/tfaligns_$expidcline\_merged.bam | sort -k1,1 -k2,2n > $tempdir/tfaligns_$expidcline\_pos.bg
	bedtools genomecov -5 -bg -strand - -g $chromsizes -ibam $tempdir/tfaligns_$expidcline\_merged.bam | sort -k1,1 -k2,2n > $tempdir/tfaligns_$expidcline\_neg.bg

	# 1.4) Convert BedGraphs to BigWigs
	printf "\tConverting BedGraphs to BigWigs\n"
	bedGraphToBigWig $tempdir/tfaligns_$expidcline\_pos.bg $chromsizes $outdir/$tfname\_$expidcline\_pos.bw
	bedGraphToBigWig $tempdir/tfaligns_$expidcline\_neg.bg $chromsizes $outdir/$tfname\_$expidcline\_neg.bw

	# 2) Copy over the optimal peaks for the positive training set
	printf "\tGenerating bins of positive-binding peaks\n"
	cp $tfpeaksopt $outdir/$tfname\_$expidcline\_all_peakints.bed.gz

	# Clean up this iteration
	rm -rf $tempdir/*
done

# Iterate through the control ChIPseq experiments/cell lines
# Focus on those with an alignment, peaks, and a control
for expidcline in `printf "%s\n" "${matchedconts[@]}" | sort -u`  # Uniquify set of controls
do
	echo "Processing control ChIPseq experiment $expidcline ..."
	contaligns=$(find $contindir -name $expidcline\_align-unfilt_*)

	# 1) Convert control ChIP-seq alignment BAMs to BigWig
	# 1.1) Filter BAM alignments for quality and mappability
	printf "\tFiltering control BAM files\n"
	for contalign in $contaligns
	do
		name=$(basename $contalign)
		samtools view -F 780 -q 30 -b $contalign -o $tempdir/$name.filt
	done
	contalignsfilt=$(find $tempdir -name *.bam.filt)

	# 1.2) Merge replicate BAM files
	printf "\tMerging control ChIP-seq replicate BAMs\n"
	numcontaligns=$(echo "$contalignsfilt" | wc -l)
	if [[ $numcontaligns -gt 1 ]]
	then
		samtools merge $tempdir/contaligns_$expidclind\_merged.bam $contalignsfilt
	else
		ln -s $contalignsfilt $tempdir/contaligns_$expidclind\_merged.bam
	fi

	# 1.3) Split BAM into + and - strands in BedGraphs
	printf "\tSplitting BAM into BedGraphs by strand\n"
	bedtools genomecov -5 -bg -strand + -g $chromsizes -ibam $tempdir/contaligns_$expidclind\_merged.bam | sort -k1,1 -k2,2n > $tempdir/contaligns_$expidcline\_pos.bg
	bedtools genomecov -5 -bg -strand - -g $chromsizes -ibam $tempdir/contaligns_$expidclind\_merged.bam | sort -k1,1 -k2,2n > $tempdir/contaligns_$expidcline\_neg.bg

	# 1.4) Convert BedGraphs to BigWigs
	printf "\tConverting BedGraphs to BigWigs\n"
	bedGraphToBigWig $tempdir/contaligns_$expidcline\_pos.bg $chromsizes $outdir/control_$expidcline\_pos.bw
	bedGraphToBigWig $tempdir/contaligns_$expidcline\_neg.bg $chromsizes $outdir/control_$expidcline\_neg.bw
	
	# Clean up this iteration
	rm -rf $tempdir/*
done

rm -r $tempdir
