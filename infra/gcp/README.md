### Description of files

### Setting up GCP
- Installing `gcloud` and `gsutil`
	```
	curl https://sdk.cloud.google.com | bash
	exec -l $SHELL
	gcloud init
	gcloud auth login
	```
	- Installation prefix is `/users/amtseng/lib/`
- Install the GCP version of kube
	- `gcloud components install kubectl`
	- Verify using `which kubectl` is the GCP version, if another `kubectl` may exist on the system
- Install `gcsfuse`
	- Requires `sudo`
	- Technically, `gcsfuse` is not strictly necessary to move data in/out of a bucket, but it is useful for mounting the bucket for easier access (instead of using `gsutil cp`)
- Get service account key for mounting buckets with `gcsfuse`
	- Go [here](https://console.cloud.google.com/iam-admin/serviceaccounts?project=gbsc-gcp-lab-kundaje)
	- The `gcsfuse3` service account has been created with the proper premissions to work with GCSFuse to mount buckets for the project
	- Created JSON key and downloaded it
- Create Kubernetes cluster on GCP
	- Done through GUI online, created cluster `amtseng`
	- Some specifications selected (outside of defaults):
		- Zone: us-west1-a (us-central is default, but this can be slow)
		- GPU pool:
			- Autoscale nodes, [0, 16]
			- n1-standard-8 (i.e. 8 CPUs, 30 GB RAM) per node
			- 1 Tesla P100 per node
			- 28 GB boot disk per node
			- Storage permissions: read/write
- Connect `kubectl` to the cluster
	- `gcloud container clusters get-credentials amtseng --zone us-west1-a`
	- This draws credentials from the cluster and creates a `kubeconfig` entry so `kubectl` commands can access the cluster
- Install Nvidia GPU drivers to cluster
	- When a cluster has GPU nodes, drivers must be installed to the nodes
	- `kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml`
	- This needs to be done only once for the node pool

### Create bucket for project
- Create bucket for project
	- Created bucket `gs://gbsc-gcp-lab-kundaje-user-amtseng-prj-ap/` on the GUI
	- Regional, with _same region as cluster_, `us-west1` (otherwise the bucket is not accessible to the cluster)
	- Used uniform access control (i.e. same across entire bucket)
	- Default permissions is read/write within the project (i.e. within `gbsc-gcp-lab-kundaje`)
- Mount the bucket and copy data into the bucket
	- `gcsfuse --key-file /users/amtseng/gbsc-gcp-lab-kundaje-be8de736710b.json --implicit-dirs -o allow_other gbsc-gcp-lab-kundaje-user-amtseng-prj-ap ~/mounts/gbsc-gcp-lab-kundaje-user-amtseng-prj-ap`
		- Note the `gs://` prefix is not specified to `gcsfuse`
		- The full path to the keyfile must be specified
	- To unmount, `fusermount -u ~/mounts/gbsc-gcp-lab-kundaje-user-amtseng-prj-ap`
- Populate the bucket with data
	- Done with script `populate_bucket.sh`
	- Warning: this takes on the order of 30-60 minutes
- Once a pod is created through `kubectl` (the `kubectl` that is connected to the cluster), the buckets associated with the project will be visible through `gsutil` automatically

### Creating a Docker image
- The Docker image `kundajelab/genome-pytorch-sacred:gcp` was created (see `..docker/`)
	- In addition to the training requirements, this image has `gcloud` installed, and has created the `/users/amtseng/` directory
- `Dockerfile` contains the definition of the Docker image for training
- `train_requirements.txt` is a file obtained by manually pruning the output of `pip freeze`, and is used to prepare the Docker image
- The Docker image is built and pushed to `kundajelab/genome-pytorch-sacred:gcp` on DockerHub
	- `docker build . -t kundajelab/genome-pytorch-sacred:gcp`
	- `docker push kundajelab/genome-pytorch-sacred:gcp`
- A note about updating images: unless otherwise specified, Kubernetes will use a cached image if the tag has already been pulled

### Test a job
- Test scripts and YAMLs are in the the directory `test/`
- Before running them, make sure that the scripts are copied to the bucket
	- This can be done with `populate_bucket_test.sh`
- `test_base` tests the basic ability to create a pod with the image, and pull data from the bucket
- `test_gpu` tests the ability to run a simple GPU job and write results to the bucket
- Warning: even with autoscale enabled, there must be at least one node running in the cluster for jobs to be submitted
	- `gcloud container clusters resize amtseng --num-nodes 1 --node-pool gpu-pool --zone us-west1-a`
	- Or through the console
