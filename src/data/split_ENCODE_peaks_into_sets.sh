set -beEo pipefail

tfname=$1
peaksdir=/users/amtseng/att_priors/data/interim/ENCODE/profile/$tfname
outdir=/users/amtseng/att_priors/data/processed/ENCODE/profile/labels/$tfname

# Fetch the peaks files
expidclines=$(find $peaksdir -name $tfname_*_all_peakints.bed.gz -exec basename {} \; | awk -F "_" '{print $2 "_" $3}' | sort -u)

mkdir -p $outdir
# Split each one into training and validation
for expidcline in $expidclines
do
	echo $expidcline
	zcat $peaksdir/$tfname\_$expidcline\_all_peakints.bed.gz | awk '$1 ~ /^(chr|chr1|chr8|chr21)$/' | gzip > $outdir/$tfname\_$expidcline\_val_peakints.bed.gz
	zcat $peaksdir/$tfname\_$expidcline\_all_peakints.bed.gz | awk '$1 ~ /^(chr|chr2|chr3|chr4|chr5|chr6|chr7|chr9|chr10|chr11|chr12|chr13|chr14|chr15|chr16|chr17|chr18|chr19|chr20|chr22|chrX)$/' | gzip > $outdir/$tfname\_$expidcline\_train_peakints.bed.gz
done

