set -beEuo pipefail

# This script will put data and source code into the bucket for the project
# It is intended for this bucket to be mounted to `/users/amtseng/` in the
# container

bucketname=gbsc-gcp-lab-kundaje-user-amtseng-prj-ap
mountpoint=/users/amtseng/mounts/$bucketname
mountkey=/users/amtseng/mounts/gbsc-gcp-lab-kundaje-be8de736710b.json

mkdir -p $mountpoint

# Mount the bucket using `gcsfuse`
gcsfuse --key-file $mountkey --implicit-dirs $bucketname $mountpoint

# Copy source code
echo "Copying source code..."
mkdir -p $mountpoint/att_priors
cp -r /users/amtseng/att_priors/src $mountpoint/att_priors

# Copy data
echo "Copying training data..."
mkdir -p $mountpoint/att_priors/data/processed
cp -r /users/amtseng/att_priors/data/processed/* $mountpoint/att_priors/data/processed

# Copy genomic references
echo "Copying genomic references..."
mkdir -p $mountpoint/genomes
cp /users/amtseng/genomes/hg38.chrom.sizes $mountpoint/genomes
cp /users/amtseng/genomes/hg38.fasta $mountpoint/genomes
cp /users/amtseng/genomes/hg38.fasta.fai $mountpoint/genomes
