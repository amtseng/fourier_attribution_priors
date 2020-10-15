install-dependencies:
	conda install -c anaconda click scipy numpy pymongo scikit-learn pandas
	conda install -c conda-forge tqdm matplotlib
	pip install sacred seqdataloader tables
	conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
	conda install -c bioconda pyfaidx pybigwig
	conda install h5py

install-interpet:
	mkdir -p ~/lib
	cd ~/lib && git clone https://github.com/atseng95/shap.git && cd shap && pip install -e .
	cd ~/lib && git clone https://github.com/kundajelab/tfmodisco.git && cd tfmodisco && git checkout v0.5.5.5 && pip install -e .
	pip install psutil
