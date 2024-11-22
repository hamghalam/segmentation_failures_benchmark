# Guide to split files with case IDs

To guarantee reproducibility, this folder contains the split files used for all experiments in this paper. For each dataset, there are two files:

- `splits_final.json`: This file defines the training-validation splits (folds). It is produced during preprocessing by nnU-Net.
- `domain_mapping_00.json`: This file lists all test cases and their "domain". It is produced during dataset conversion [(scripts here)](../src/segmentation_failures/data/dataset_conversion/).
