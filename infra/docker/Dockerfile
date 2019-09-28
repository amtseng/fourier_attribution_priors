# Use the official PyTorch image as parent
FROM pytorch/pytorch

# Set the working directory
WORKDIR /att_priors

# Copy the set of requirements for training
COPY train_requirements.txt /att_priors

# Install any needed packages specified in requirements.txt, and others
RUN pip install -r train_requirements.txt
# RUN apt install -y vim-tiny less
