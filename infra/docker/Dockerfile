# Use the official PyTorch image as parent
FROM pytorch/pytorch

# Set the working directory
WORKDIR /att_priors

# Copy the set of requirements for training
COPY train_requirements.txt /att_priors

# Install any needed packages specified in requirements.txt, and others
RUN pip install -r train_requirements.txt
RUN git clone https://github.com/atseng95/sacred.git
RUN cd sacred && python setup.py install

# Set environment variables for Python
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
