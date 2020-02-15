set -beEo pipefail

cline=$1
clineindir=/users/amtseng/att_priors/data/raw/ENCODE_DNase/$cline/
outdir=/users/amtseng/att_priors/data/interim/ENCODE_DNase/profile/$cline

chromsizes=/users/amtseng/genomes/hg38.with_ebv.chrom.sizes

tempdir=$outdir/temp
mkdir -p $tempdir

# Iterate through the DNase cell lines/experiments
# Focus on those with an alignment and peaks
expidclines=$(find $clineindir -name *.bam -exec basename {} \; | awk -F "_" '{print $1 "_" $2}' | sort -u)
for expidcline in $expidclines
do
	echo "Processing DNase-seq experiment $expidcline ..."
	idrpeaks=$(find $clineindir -name $expidcline\_peaks-idr.bed.gz)
	aligns=$(find $clineindir -name $expidcline\_merged.bam)
	expid=$(echo $expidcline | cut -d "_" -f 1)
	
	if [[ -z $idrpeaks ]] || [[ -z $aligns ]]
	then
		printf "\tDid not find all required alignments and peaks\n"
		continue
	fi

	numtfpeaksopt=$(echo "$idrpeaks" | wc -l)
	if [[ $numidrpeaks -gt 1 ]]
	then
		printf "\tFound more than one set of optimal IDR peaks\n"
		continue
	fi

	numaligns=$(echo "$aligns" | wc -l)
	if [[ $numaligns -gt 1 ]]
	then
		printf "\tFound more than one set of merged alignments\n"
		continue
	fi

	# 1) Filter the BAM alignments and generate BigWigs
	# 1.1) Filter BAM alignments for quality and mappability
	printf "\tFiltering BAM file\n"
	name=$(basename $aligns)
	filtaligns=$tempdir/$name.filt
	samtools view -F 780 -q 30 -b $aligns -o $filtaligns

	# 1.2) Convert to BedGraph
	printf "\tConverting BAM to BedGraph\n"
	bedtools genomecov -5 -bg -g $chromsizes -ibam $filtaligns | sort -k1,1 -k2,2n > $tempdir/aligns_$expidcline.bg

	# 1.3) Convert BedGraph to BigWig
	printf "\tConverting BedGraph to BigWig\n"
	bedGraphToBigWig $tempdir/aligns_$expidcline.bg $chromsizes $outdir/$cline\_$expid.bw

	# 2) Copy over the optimal peaks for the positive training set
	printf "\tGenerating bins of positive-binding peaks\n"
	cp $idrpeaks $outdir/$cline\_$expid\_all_peakints.bed.gz

	# Clean up this iteration
	rm -rf $tempdir/*
done

rm -r $tempdir
