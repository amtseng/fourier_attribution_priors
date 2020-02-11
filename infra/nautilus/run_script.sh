#!/bin/bash

set -beEo pipefail

show_help() {
	cat << EOF
Usage: ${0##*/} -m MODEL_DIR -t TRAIN_BED -v VAL_BED -c CONFIG_JSON 
                -p SRC1 DEST1 -p SRC2 DEST2 [ CONFIGS ]
Copies desired files to a more local location, and kicks off training.
These arguments are directly supplied to 'hyperparam.py', and their descriptions
are identical.
    -h,--help           Display this help and exit
    -m,--modeldir       Location to store the saved model
    -t,--trainbed       Path to the training BED file
    -v,--valbed         Path to the validation BED file
    -c,--configjson     Path to a JSON file containing config options for Sacred
                        to override existing options (optional)
    -p,--copy           Specifies a pair of source and destination to copy files
                        (e.g. genome files, data files, etc.); this can be used
                        to specify the desired files to copy to a local location
    CONFIGS             Any other commandline arguments will be parsed as
                        additional config options for Sacred; these will
                        override existing options in Sacred, as well as any
                        options specified by a JSON file
EOF
}

# Arrays to hold sources and destinations of what to copy
SRCS=()
DESTS=()

POSARGS=""  # Positional arguments
while [ $# -gt 0 ]
do
	case "$1" in
		-h|--help)
			show_help
			exit 0
			;;
		-m|--modeldir)
			modeldir="$2"
			shift 2
			;;
		-t|--trainbed)
			trainbed="$2"
			shift 2
			;;
		-v|--valbed)
			valbed="$2"
			shift 2
			;;
		-c|--configjson)
			configjson="$2"
			shift 2
			;;
		-p|--copy)
			SRCS+=("$2")
			DESTS+=("$3")
			shift 3
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

otherconfig=$@

if [[ ! -z $configjson ]]
then
	configopt="-c $configjson"
else
	configopt=""
fi
if [[ -z $modeldir ]]
then
	echo "Must specify a location to store the model"
	show_help >&2
	exit 1
fi

echo "Copying..."

# Copy needed files to a better location
for ((i=0; i<${#SRCS[@]}; ++i))
do
	src="${SRCS[i]}"
	dest="${DESTS[i]}"
	echo "... $src TO $dest"
	cp $src $dest
done

# Copy source code
echo "... Source code"
cp -r /amtseng_ceph/att_priors/src /att_priors

# Kick-off training
echo "Beginning hyperparameter tuning..."
cd /att_priors/src
export LC_ALL=C.UTF-8  # Needed for Python in the Docker container
export LANG=C.UTF-8
MODEL_DIR=$modeldir python -m model.hyperparam -t $trainbed -v $valbed $configopt $@
