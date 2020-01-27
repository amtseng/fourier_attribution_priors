tfname=$1
thresh=5

jasparmotifpath=/users/amtseng/att_priors/data/raw/JASPAR/SPI1/SPI1_motifs.motif
peakbeds=/users/amtseng/att_priors/data/processed/ENCODE/profile/labels/$tfname/*.bed.gz
homermotifpath=/users/amtseng/att_priors/data/interim/HOMER/SPI1/SPI1_homer_motifs.motif
motifpredsstem=/users/amtseng/att_priors/data/interim/HOMER/SPI1/SPI1_motif_preds
outtrack=/users/amtseng/att_priors/data/processed/HOMER/SPI1/SPI1_motif_preds.bw

motifconverter=/users/amtseng/test/genomewide_motif_preds/jaspar_to_homer_motif.py
chromsizes=/users/amtseng/genomes/hg38.chrom.sizes
genome=hg38

mkdir -p `dirname $homermotifpath`
mkdir -p `dirname $motifpredsstem`
mkdir -p `dirname $outtrack`

# 1) Convert the motifs downloaded from JASPAR to a HOMER motif file
echo "Converting JASPAR motif file to HOMER motif file"
printf "\tSetting threshold to $thresh\n"
python $motifconverter -t $thresh $jasparmotifpath > $homermotifpath

# 2) Run HOMER motif scanning with these motifs
echo "Scanning motifs through the genome with HOMER"
scanMotifGenomeWide.pl $homermotifpath $genome > $motifpredsstem.tsv -p 10

# 3) Compute the number of non-overlapping peaks in the specified BED files
numpeaks=$(zcat $peakbeds | bedtools sort | bedtools merge | wc -l)
echo "Found $numpeaks non-overlapping peaks in the BEDs"

# 4) For each motif, get the top `numpeaks` instances
echo "Getting top hits for each motif"
motifnames=$(cat $homermotifpath | grep ">" | cut -f 2)
printf "" > $motifpredsstem.top.tsv  # Clear this file
for motifname in $motifnames
do
	printf "\t$motifname\n"
	tmpfile=$(mktemp)
	cat $motifpredsstem.tsv | awk -F "-" -v motifname=$motifname '$1 == motifname' | sort -k6,6rn | head -n $numpeaks > $tmpfile
	numrows=$(cat $tmpfile | wc -l)
	if [[ $numrows -lt $numpeaks ]]
	then
		echo "Warning: $motifname had only $numrows hits; consider reducing the default threshold"
	fi
	cat $tmpfile >> $motifpredsstem.top.tsv
done

echo "Converting motif predictions to BigWig"
# 5) Merge the top predictions into a single BED file
cat $motifpredsstem.top.tsv | cut -f 2,3,4 | bedtools sort | bedtools merge > $motifpredsstem.bed

# 6) Convert this BED file into a BigWig
cat $motifpredsstem.bed | awk '{print $0 "\t" 1}' > $motifpredsstem.bg
bedGraphToBigWig $motifpredsstem.bg $chromsizes $outtrack
