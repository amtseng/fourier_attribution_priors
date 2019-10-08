tfname=$1
tfindir=/users/amtseng/att_priors/data/raw/ENCODE/$tfname/tf_chipseq
contindir=/users/amtseng/att_priors/data/raw/ENCODE/$tfname/control_chipseq
outdir=/users/amtseng/att_priors/data/processed/ENCODE/profile/labels/$tfname

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
	tfaligns=$(find $tfindir -name $expidcline\_align_*)
	tfpeaksopt=$(find $tfindir -name $expidcline\_peaks-optimal_*)
	cline=$(echo $expidcline | cut -d "_" -f 2)
	contaligns=$(find $contindir -name *_$cline\_align_*)
	
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

	contexpidclines=$(find $contindir -name *_$cline\_align_* -exec basename {} \; | awk -F "_" '{print $1 "_" $2}')
	for item in $contexpidclines
	do
		matchedconts+=("$item")
	done

	# 1) Convert TF ChIP-seq alignment BAMs to BigWig
	# 1.1) Merge replicate BAM files
	printf "\tMerging TF ChIP-seq replicate BAMs\n"
	numtfaligns=$(echo "$tfaligns" | wc -l)
	if [[ $numtfaligns -gt 1 ]]
	then
		samtools merge $tempdir/tfaligns_$expidcline\_merged.bam $tfaligns
	else
		ln -s $tfaligns $tempdir/tfaligns_$expidcline\_merged.bam
	fi

	# 1.2) Split BAM into + and - strands in BedGraphs
	printf "\tSplitting BAM into BedGraphs by strand\n"
	bedtools genomecov -5 -bg -strand + -g $chromsizes -ibam $tempdir/tfaligns_$expidcline\_merged.bam | sort -k1,1 -k2,2n > $tempdir/tfaligns_$expidcline\_pos.bg
	bedtools genomecov -5 -bg -strand - -g $chromsizes -ibam $tempdir/tfaligns_$expidcline\_merged.bam | sort -k1,1 -k2,2n > $tempdir/tfaligns_$expidcline\_neg.bg

	# 1.3) Convert BedGraphs to BigWigs
	printf "\tConverting BedGraphs to BigWigs\n"
	bedGraphToBigWig $tempdir/tfaligns_$expidcline\_pos.bg $chromsizes $outdir/$tfname\_$expidcline\_pos.bw
	bedGraphToBigWig $tempdir/tfaligns_$expidcline\_neg.bg $chromsizes $outdir/$tfname\_$expidcline\_neg.bw

	# 2) Generate bins of the positive binding centered around optimal peak summits
	# 2.1) Fetch the peak summits, expand to length 1000 but not past the chromosome edges
	printf "\tGenerating bins of positive-binding peaks\n"
	zcat $tfpeaksopt | awk -F "\t" '{print $1 "\t" $2 + $10 "\t" $2 + $10}' | bedtools slop -g $chromsizes -b 500 | awk '$3 - $2 == 1000' | bedtools sort | gzip > $outdir/$tfname\_$expidcline\_all_peakints.bed.gz

	# 2.2) Split into training and validation
	zcat $outdir/$tfname\_$expidcline\_all_peakints.bed.gz | awk '$1 ~ /^(chr|chr2|chr3|chr4|chr5|chr6|chr7|chr9|chr11|chr12|chr13|chr14|chr15|chr16|chr17|chr18|chr19|chr20|chr22|chrX|chrY|chrM)$/' | gzip > $outdir/$tfname\_$expidcline\_train_peakints.bed.gz
	zcat $outdir/$tfname\_$expidcline\_all_peakints.bed.gz | awk '$1 ~ /^(chr|chr1|chr8|chr21)$/' | gzip > $outdir/$tfname\_$expidcline\_holdout_peakints.bed.gz

	# Clean up this iteration
	rm -rf $tempdir/*
done

# Iterate through the control ChIPseq experiments/cell lines
# Focus on those with an alignment, peaks, and a control
for expidcline in `printf "%s\n" "${matchedconts[@]}" | sort -u`  # Uniquify set of controls
do
	echo "Processing control ChIPseq experiment $expidcline ..."
	contaligns=$(find $contindir -name $expidcline\_align_*)

	# 1) Convert control ChIP-seq alignment BAMs to BigWig
	# 1.1) Merge replicate BAM files
	printf "\tMerging control ChIP-seq replicate BAMs\n"
	numcontaligns=$(echo "$contaligns" | wc -l)
	if [[ $numcontaligns -gt 1 ]]
	then
		samtools merge $tempdir/contaligns_$expidclind\_merged.bam $contaligns
	else
		ln -s $contaligns $tempdir/contaligns_$expidclind\_merged.bam
	fi

	# 1.2) Split BAM into + and - strands in BedGraphs
	printf "\tSplitting BAM into BedGraphs by strand\n"
	bedtools genomecov -5 -bg -strand + -g $chromsizes -ibam $tempdir/contaligns_$expidclind\_merged.bam | sort -k1,1 -k2,2n > $tempdir/contaligns_$expidcline\_pos.bg
	bedtools genomecov -5 -bg -strand - -g $chromsizes -ibam $tempdir/contaligns_$expidclind\_merged.bam | sort -k1,1 -k2,2n > $tempdir/contaligns_$expidcline\_neg.bg

	# 1.3) Convert BedGraphs to BigWigs
	printf "\tConverting BedGraphs to BigWigs\n"
	bedGraphToBigWig $tempdir/contaligns_$expidcline\_pos.bg $chromsizes $outdir/control_$expidcline\_pos.bw
	bedGraphToBigWig $tempdir/contaligns_$expidcline\_neg.bg $chromsizes $outdir/control_$expidcline\_neg.bw
	
	# Clean up this iteration
	rm -rf $tempdir/*
done

rm -r $tempdir
