tfname=$1
inpath=$2

outdir=$(dirname $inpath)
outtrain=$outdir/$tfname\_train_labels.bed.gz
outval=$outdir/$tfname\_holdout_labels.bed.gz

zcat $inpath | awk '$1 ~ /^(chr|chr2|chr3|chr4|chr5|chr6|chr7|chr9|chr11|chr12|chr13|chr14|chr15|chr16|chr17|chr18|chr19|chr20|chr22|chrX|chrY|chrM)$/' | gzip > $outtrain
zcat $inpath | awk '$1 ~ /^(chr|chr1|chr8|chr21)$/' | gzip > $outval
