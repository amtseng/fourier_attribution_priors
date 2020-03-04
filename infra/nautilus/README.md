### Description of setup and files

### Setting up Nautilus
- Created a new namespace in Nautilus, `amtseng`
- Configure `kubectl` with Nautilus config
	- Configuration file downloaded from [Nautilus](https://nautilus.optiputer.net/)
	- Need to login using Stanford SSO, then download config (instructions [here](https://ucsd-prp.gitlab.io/userdocs/start/quickstart/))
	- Configuration added to `~/.kube/config`
		- Note that if there are existing configurations (e.g. GCP), the configuration will need to be added
		- To merge several config files, do the following:
			`KUBECONFIG=~/.kube/gcp_config:~/.kube/naut_config kubectl config view --flatten > ~/.kube/config`
		- To get all contexts: `kubectl config get-contexts`
		- To switch to Nautilus context: `kubectl config use-context nautilus`
	- It is okay to use the `kubectl` installed from GCP

### Create storage volume for project
- Create Ceph storage volume for project
	- Unlike Persistent Volumes, Ceph allows the same volume to be mounted across many pods at once
	- Credentials were obtained from the Nautilus admins, and a Ceph volume for the namespace `amtseng` was created using `kubectl create secret -n NAMESPACE generic ceph-fs-secret --from-literal=key=CEPH_KEY`
	- Instructions on Ceph are [here](https://ucsd-prp.gitlab.io/userdocs/storage/ceph/)
	- Note that Ceph is slow, so data should be copied to a more local location before running
- Populate Ceph with data
	- First, we create a pod that acts as a portal to Ceph, defined in `main_pod.yaml`
	- The data population is performed with `populate_ceph.sh` (should take around 20 minutes)
- The volume is structured such that the path to any object in Ceph is identical to the absolute path on the lab cluster (i.e. `/users/amtseng/att_priors/...`)
- When mounting Ceph to a pod, make sure to use the following under `spec`:
	```
	volumes:
	  - name: ceph
	    flexVolume:
	      driver: ceph.rook.io/rook
	      fsType: ceph
	      options:
	        clusterNamespace: rook
	        fsname: nautilusfs
	        path: /amtseng
	        mountUser: amtseng
	        mountSecret: ceph-fs-secret
	```

### Creating a Docker image
- The Docker image `kundajelab/genome-pytorch-sacred:nautilus` was created (see `docker/`)
- `Dockerfile` contains the definition of the Docker image for training
- `train_requirements.txt` is a file obtained by manually pruning the output of `pip freeze`, and is used to prepare the Docker image
- The Docker image is built and pushed to `kundajelab/genome-pytorch-sacred:nautilus` on DockerHub
	- `docker build . -t kundajelab/genome-pytorch-sacred:nautilus`
	- `docker push kundajelab/genome-pytorch-sacred:nautilus`
- A note about updating images: unless otherwise specified, Kubernetes will use a cached image if the tag has already been pulled

### Test a job
- Test scripts and YAMLs are in the the directory `test/`
- Before running them, make sure that the scripts are copied to Ceph
	- This can be done with `populate_ceph_test.sh`
- `test_base` tests the basic ability to create a pod with the image, and pull data from Ceph
- `test_gpu` tests the ability to run a simple GPU job

### Running training jobs on Nautilus
- The Docker image will already create the `/users/amtseng/` directory; this allows easy resharing of configuration files
- The script `nautilus_hyperparam.py` takes in the same arguments as `hyperparam.py`
	- This script will copy over data from the Ceph to the right places, then run `hyperparam.py` with those arguments
- The `populate_ceph.sh` script must be run to populate the bucket with all training data and source code, including `nautilus_hyperparam.py`
- To run the job, a job command spec might look like this:
	```
	apiVersion: batch/v1
	kind: Job
	metadata:
	  name: profile-k562-prior
	spec:
	  template:
	    spec:
	      containers:
	      - name: profile-k562-prior
	        image: kundajelab/genome-pytorch-sacred:nautilus
	        imagePullPolicy: Always
	        resources:
	          requests:
	            cpu: 8
	            memory: 32Gi
	            nvidia.com/gpu: 1
	          limits:
	            cpu: 8
	            memory: 32Gi
	            nvidia.com/gpu: 1
	        command:
	        - /bin/bash
	        - -c
	        args:
	        - cp /ceph/nautilus_hyperparam.py ~;
	          cd ~;
	          MODEL_DIR=/users/amtseng/att_priors/models/trained_models/profile/K562_prior python nautilus_hyperparam.py -t profile -f /users/amtseng/att_priors/data/processed/ENCODE_DNase/profile/config/K562/K562_training_paths.json -s /users/amtseng/att_priors/data/processed/chrom_splits.json -k 1 -c /users/amtseng/att_priors/data/processed/ENCODE_TFChIP/profile/config/K562/K562_config.json train.early_stopping=False -n 30
	        volumeMounts:
	        - mountPath: /ceph
	          name: ceph
	      restartPolicy: Never
	      volumes:
	        - name: ceph
	          flexVolume:
	            driver: ceph.rook.io/rook
	            fsType: ceph
	            options:
	              clusterNamespace: rook
	              fsname: nautilusfs
	              path: /amtseng
	              mountUser: amtseng
	              mountSecret: ceph-fs-secret
	  backoffLimit: 0
	```
