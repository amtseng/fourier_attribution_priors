# Use the official Google Cloud image as parent
FROM pytorch/pytorch

# Set the working directory
WORKDIR /att_priors

# Copy the set of requirements for training
COPY train_requirements.txt /att_priors

# Install any needed packages specified in requirements.txt
RUN pip install -r train_requirements.txt

# Set environment variables for Python
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
