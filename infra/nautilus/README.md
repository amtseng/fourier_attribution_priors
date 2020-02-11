### Description of files
`main_pod.yaml`
- This creates a pod named `att-priors`, which mounts the permanent Ceph storage system
- This pod can be used for exploration, looking around, and debugging

`load_ceph_data.sh`
- This script uploads the needed data (i.e. training data, source code, and genomic files) to the Ceph storage; it assumes that the `att-priors` pod has been spun up already

`run_script.sh`
- This script will be called by a job to kick off training
- This will copy the needed data from Ceph (which is slow) to a local storage system, and run from there
- An example of how to run this script:
```
	bash run_script.sh \
		-t /att_priors/TEAD4_test_2000.bed.gz \
		-v /att_priors/TEAD4_test_2000.bed.gz \
		-c /amtseng_ceph/att_priors/data/processed/ENCODE/config/TEAD4_config.json \
		-m /amtseng_ceph/att_priors/trained_models/TEST/ \
		-p /amtseng_ceph/genomes/hg38.fasta /att_priors/ \
		-p /amtseng_ceph/genomes/hg38.fasta.fai /att_priors/ \
		-p /amtseng_ceph/att_priors/data/processed/ENCODE/tests/TEAD4_test_2000.bed.gz /att_priors/ \
		dataset.reference_fasta=/att_priors/hg38.fasta \
		train.avg_class_loss=True \
		train.att_prior_loss_weight=1
```

`job_template.yaml`
- An example template of what a job configuration file should look like

`jobs/`
- Filled-in jobs for running

### Docker container
- `Dockerfile` contains the definition of the Docker image for training
- `train_requirements.txt` is a file obtained by manually pruning the output of `pip freeze`, and is used to prepare the Docker image
- The Docker image is built and pushed to `kundajelab/genome-pytorch-sacred:latest` on DockerHub
	- `docker build . -t kundajelab/genome-pytorch-sacred:latest`
	- `docker push kundajelab/genome-pytorch-sacred:latest`

### Information about the Nautilus system and `infra`
- Ceph allows the same volume to be mounted across many pods at once
	- It was setup with a given username (`amtseng`) and password, provided to `kubectl create secret generic ceph-fs-secret`
	- Ceph is slow, so data should be copied to a more local location before running
