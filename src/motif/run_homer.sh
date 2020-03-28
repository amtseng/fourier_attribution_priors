show_help() {
	cat << EOF
Usage: ${0##*/} [OPTIONS] IN_FILE OUT_DIR
Runs HOMER on the input 'IN_FILE' and outputs results in 'OUT_DIR'.
'IN_FILE' may be a Fasta, a BED, or gzipped BED.
HOMER needs to be loaded.
Assumes reference genome of hg38.
EOF
}

POSARGS=""  # Positional arguments
while [ $# -gt 0 ]
do
	case "$1" in
		-h|--help)
			show_help
			exit 0
			;;
		-g|--genome)
			genome=$2
			shift 2
			;;
		-*|--*)
			echo "Unsupported flag error: $1" >&2
			show_help >&2
			exit 1
			;;
		*)
			POSARGS="$POSARGS $1"  # Preserve positional arguments
			shift
	esac
done
eval set -- "$POSARGS"  # Restore positional arguments to expected indices

if [[ -z $2 ]]
then
	show_help
	exit 1
fi

if [[ -z $genome ]]
then
	genome=hg38
fi

infile=$1
outdir=$2

mkdir -p $outdir
if [ ${infile: -6} == ".fasta" ]
then
	findMotifs.pl $infile fasta $outdir -len 12 -p 4
elif [ ${infile: -3} == ".gz" ]
then
	findMotifsGenome.pl <(zcat $infile) $genome $outdir -len 12 -size 200 -p 4
else
	findMotifsGenome.pl $infile $genome $outdir -len 12 -size 200 -p 4
fi
