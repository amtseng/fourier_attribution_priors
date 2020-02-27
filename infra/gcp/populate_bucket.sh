set -beEuo pipefail

# This script will put data and source code into the bucket for the project
# All absolute paths will be maintained within the bucket

bucket=gs://gbsc-gcp-lab-kundaje-user-amtseng-prj-ap
localstem=/users/amtseng
bucketstem=$bucket/users/amtseng

echo "Copying initial scripts..."
gsutil cp $localstem/att_priors/infra/gcp/gcp_hyperparam.py $bucket

echo "Copying source code..."
gsutil -m cp -r $localstem/att_priors/src $bucketstem/att_priors/src

echo "Copying training data..."
gsutil -m cp -r $localstem/att_priors/data/processed $bucketstem/att_priors/data/processed

echo "Copying genomic references..."
gsutil cp $localstem/genomes/hg38.canon.chrom.sizes $bucketstem/genomes/hg38.canon.chrom.sizes
gsutil cp $localstem/genomes/hg38.fasta $bucketstem/genomes/hg38.fasta
gsutil cp $localstem/genomes/hg38.fasta.fai $bucketstem/genomes/hg38.fasta.fai
gsutil cp $localstem/genomes/mm10.canon.chrom.sizes $bucketstem/genomes/mm10.canon.chrom.sizes
gsutil cp $localstem/genomes/mm10.fasta $bucketstem/genomes/mm10.fasta
gsutil cp $localstem/genomes/mm10.fasta.fai $bucketstem/genomes/mm10.fasta.fai
