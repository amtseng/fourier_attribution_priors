set -beEuo pipefail

podname=att-priors
cephpath=/amtseng_ceph/att_priors/

echo "Copying src..."
kubectl cp /users/amtseng/att_priors/src $podname:$cephpath/src

# echo "Copying reference genomes..."
# genomesrc=/users/amtseng/genomes/
# genomedest=/amtseng_ceph/genomes/
# kubectl exec $podname -- mkdir -p $genomedest
# kubectl cp $genomesrc/hg38.fasta $podname:$genomedest/hg38.fasta
# kubectl cp $genomesrc/hg38.fasta.fai $podname:$genomedest/hg38.fasta.fai
# kubectl cp $genomesrc/hg19.fasta $podname:$genomedest/hg19.fasta
# kubectl cp $genomesrc/hg19.fasta.fai $podname:$genomedest/hg19.fasta.fai

echo "Copying data..."
datasrc=/mnt/lab_data2/amtseng/att_priors/data/  # Can't use soft-links
datadest=$cephpath/data
kubectl exec $podname -- mkdir -p $datadest
kubectl cp $datasrc/raw $podname:$datadest/
kubectl cp $datasrc/processed $podname:$datadest/

echo "Copying script..."
kubectl cp run_script.sh $podname:/$cephpath/run_script.sh
