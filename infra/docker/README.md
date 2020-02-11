- `Dockerfile` contains the definition of the Docker image for training
- `train_requirements.txt` is a file obtained by manually pruning the output of `pip freeze`, and is used to prepare the Docker image
- The Docker image is built and pushed to `kundajelab/genome-pytorch-sacred:latest` on DockerHub
	- `docker build . -t kundajelab/genome-pytorch-sacred:latest`
	- `docker push kundajelab/genome-pytorch-sacred:latest`

