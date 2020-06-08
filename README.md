# Fourier-transform-based attribution priors for genomics deep learning

### Description of files

```
├── Makefile    <- Installation of dependencies
├── data    <- Contains training data
│   ├── interim    <- Intermediate data
│   ├── processed    <- Processed data ready for training
│   ├── raw    <- Raw data, directly downloaded from the source
│   └── README.md    <- Description of data
├── models
│   └── trained_models    <- Trained models
├── notebooks    <- Jupyter notebooks that explore data and plot results
├── README.md    <- This file
├── results    <- Saved results
└── src    <- Source code
    ├── data
    │   ├── create_binary_bins.py    <- Synthesize bin-level (aggregate) labels from task-specific binary labels
    │   ├── create_BPNet_profile_hdf5.py    <- Create profile labels from profile tracks
    │   ├── create_ENCODE_DNase_profile_hdf5.py    <- Create profile labels from profile tracks
    │   ├── create_ENCODE_TFChIP_profile_hdf5.py    <- Create profile labels from profile tracks
    │   ├── download_ENCODE_DNase_data.py    <- Download DNase-seq peaks/BAMs from ENCODE portal
    │   ├── download_ENCODE_TFChIP_cellline_peaks.py    <- Download specific TF's and cell line's TF ChIP-seq peaks/BAMs from ENCODE portal
    │   ├── download_ENCODE_TFChIP_data.py    <- Download specific TF's TF ChIP-seq peaks/BAMs from ENCODE portal
    │   ├── generate_BPNet_binary_labels.sh    <- Generate binary labels for Nanog/Oct4/Sox2 models from peaks
    │   ├── generate_ENCODE_DNase_binary_labels.sh    <- Generate binary labels for DNAse-seq models from peaks
    │   ├── generate_ENCODE_DNase_profile_labels.sh    <- Generate profile tracks for DNase-seq models from read tracks
    │   ├── generate_ENCODE_TFChIP_binary_labels.sh    <- Generate binary labels for TF ChIP-seq models from peaks
    │   └── generate_ENCODE_TFChIP_profile_labels.sh    <- Generate profile tracks for TF ChIP-seqmodels form read tracks
    ├── extract
    │   ├── cluster_gradients.py    <- Helper functions for clustering similar importance score tracks
    │   ├── compute_ism.py    <- Compute _in silico_ mutagenesis scores
    │   ├── compute_predictions.py    <- Compute model predictions and gradients from a trained model
    │   ├── compute_shap.py    <- Compute DeepSHAP importance scores from a trained model
    │   ├── data_loading.py    <- Data loading utilities to easily get model input data for a coordinate/bin
    │   ├── extract_bed_interval.sh    <- Extract a set of BED intervals overlapping a range
    │   ├── __init__.py
    │   ├── make_shap_scores.py    <- Generate DeepSHAP scores over all positive examples
    │   └── run_tfmodisco.py    <- Run TF-MoDISco on DeepSHAP scores to discover motifs
    ├── feature
    │   ├── __init__.py
    │   ├── make_binary_dataset.py    <- Data loading for binary models
    │   ├── make_profile_dataset.py    <- Data loading for profile models
    │   └── util.py    <- Shared data loading utilities
    ├── model
    │   ├── binary_models.py    <- Binary model architecture(s)
    │   ├── binary_performance.py    <- Binary model performance metrics
    │   ├── binary_performance_test.py    <- Tests for binary model performance metric code correctness
    │   ├── hyperparam.py    <- Wrapper for hyperparameter tuning
    │   ├── __init__.py
    │   ├── profile_models.py    <- Profile model architecture(s)
    │   ├── profile_performance.py    <- Profile model performance metrics
    │   ├── profile_performance_test.py    <- Tests for profile model performance metric code correctness
    │   ├── train_binary_model.py    <- Training binary models
    │   ├── train_profile_model.py    <- Training profile models
    │   └── util.py    <- Shared training/model utilities
    ├── motif
    │   ├── generate_simulated_fasta.py    <- Generate a set of synthetic sequences, with embedded motifs
    │   ├── homer2meme.py    <- Convert a HOMER motif file to a MEME motif file
    │   └── run_homer.sh    <- Run HOMER 2
    └── plot
        ├── __init__.py
        └── viz_sequence.py    <- Plot an importance score track
```
