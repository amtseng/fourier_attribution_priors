set -beEuo pipefail

podname=att-priors
ceph=/ceph
stem=/users/amtseng

read -p "Is the main 'att-priors' pod running? [Y/n]?" resp
case "$resp" in
	n|N)
		echo "Aborted"
		exit 1
		;;
	*)
		;;
esac

echo "Copying initial scripts..."
kubectl cp $stem/att_priors/infra/nautilus/nautilus_hyperparam.py $podname:$ceph

echo "Copying source code..."
kubectl exec $podname -- mkdir -p $ceph/$stem/att_priors/
kubectl cp $stem/att_priors/src $podname:$ceph/$stem/att_priors/

echo "Copying training data..."
kubectl exec $podname -- mkdir -p $ceph/$stem/att_priors/data/processed
for item in `ls $stem/att_priors/data/processed/`
do
	kubectl cp $stem/att_priors/data/processed/$item $podname:$ceph/$stem/att_priors/data/processed/
done

echo "Copying genomic references..."
kubectl exec $podname -- mkdir -p $ceph/$stem/genomes
kubectl cp $stem/genomes/hg38.canon.chrom.sizes $podname:$ceph/$stem/genomes
kubectl cp $stem/genomes/hg38.fasta $podname:$ceph/$stem/genomes
kubectl cp $stem/genomes/hg38.fasta.fai $podname:$ceph/$stem/genomes
kubectl cp $stem/genomes/mm10.canon.chrom.sizes $podname:$ceph/$stem/genomes
kubectl cp $stem/genomes/mm10.fasta $podname:$ceph/$stem/genomes
kubectl cp $stem/genomes/mm10.fasta.fai $podname:$ceph/$stem/genomes
