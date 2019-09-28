set -beEuo pipefail

modeldir=$1
trainpath=$2
valpath=$3
configpath=$4

# Kick-off training
cd /amtseng_ceph/src
MODEL_DIR=/amtseng_ceph/trained_models/$modeldir python -m model.hyperparam_search -t $trainpath -v $valpath -c $configpath
