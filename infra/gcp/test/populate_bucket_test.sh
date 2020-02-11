set -beEuo pipefail

bucketname=gbsc-gcp-lab-kundaje-user-amtseng-prj-ap

echo "Copying test scripts..."
gsutil cp test_base.sh gs://$bucketname/test/
gsutil cp test_gpu.sh gs://$bucketname/test/
gsutil cp test_gpu.py gs://$bucketname/test/
