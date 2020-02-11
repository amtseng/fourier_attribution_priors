# Use the official Google Cloud image as parent
FROM pytorch/pytorch

# Set the working directory
WORKDIR /scratch

# Copy the set of requirements for training
COPY train_requirements.txt /scratch

# Install any needed packages specified in requirements.txt
RUN pip install -r train_requirements.txt

# Install Google Cloud SDK
RUN apt-get update && apt install -y --allow-unauthenticated wget
RUN cd /opt/ && \
	wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-279.0.0-linux-x86_64.tar.gz && \
	tar xvfz google-cloud-sdk-279.0.0-linux-x86_64.tar.gz && \
	./google-cloud-sdk/install.sh
ENV PATH "$PATH:/opt/google-cloud-sdk/bin/"

# Create /users/amtseng directory
RUN mkdir -p /users/amtseng

# Set environment variables for Python
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
