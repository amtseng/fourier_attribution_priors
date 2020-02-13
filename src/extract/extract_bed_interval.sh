set -beEo pipefail

show_help() {
	cat << EOF
Usage: ${0##*/} BED CHROM START END [-c CENTERPAD] [-p]
Given the path to a BED file and a coordinate, extracts all rows in the BED file
that intersect with the coordinate.
    -c/--centerpad    If given, center the coordinate and pad on each side
                      to this size
    -p/--partial      If given, allow partial matches
    -h/--help         Show this message and quit
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
		-c|--centerpad)
			centerpad=$2
			shift 2
			;;
		-p|--partial)
			partial=1
			shift 1
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

if [[ -z $4 ]]
then
	show_help
	exit 1
fi

bedpath=$1
chrom=$2
start=$3
end=$4

if [[ ! -z $centerpad ]]
then
	# Pad the coordinate to size `centerpad`
	midpoint=$(echo "($start + $end) / 2" | bc)
	padhalf=$(echo "$centerpad / 2" | bc)
	start=$(echo "$midpoint - $padhalf" | bc)
	end=$(echo "$start + $centerpad" | bc)
fi

if [[ ! -z $partial ]]
then
	bedtools intersect -a $bedpath -b <(printf "$chrom\t$start\t$end\n") -wa | awk -v start=$start '{$2 = $2 - start; $3 = $3 - start; print $0}'
else
	bedtools intersect -a $bedpath -b <(printf "$chrom\t$start\t$end\n") -wa -f 1 | awk -v start=$start '{$2 = $2 - start; $3 = $3 - start; print $0}'
fi
