# Requirements
numpy=1.24.3
pandas=1.5.3
matplotlib=3.7.1
seaborn=0.12.2
torch=1.13.1
pytorch-lightning=1.5.6
# Blood Cancer Auxiliary Diagnosis Platform

Welcome to our free-to-use website designed to assist medical staff with the auxiliary diagnosis of blood cancers. This platform harnesses advanced technology to provide a reliable support tool for healthcare professionals.

## Access the Platform
You can access the platform at the following URL: [http://14.29.210.22:8001](http://14.29.210.22:8001).

# Dataset
## Download
The dataset can be downloaded from the link below[1]:
[Download Dataset](https://drive.google.com/drive/folders/1VcmDOdBbG46ILRd99TM2ZsZHBpcMazZ6)

## Usage Instructions
1. Download all files and extract them to the `dataset` folder.
2. Run `./utils/dataset.py` to convert flow cytometry data into tensors.

# How to Run
Use the following command to train the model:
python main.py --mode=train --dataset=CLL/B-ALL

# Acknowledgments
We thank all co-authors of the following articles for providing data.

[1]Edgar E Robles, Ye Jin, Padhraic Smyth, Richard H Scheuermann, Jack D Bui, Huan-You Wang, Jean Oak, Yu Qian, "A Cell-Level Discriminative Neural Network Model for Diagnosis of Blood Cancers", Bioinformatics, Volume 39, Issue 10, October 2023, btad585, [https://doi.org/10.1093/bioinformatics/btad585](https://doi.org/10.1093/bioinformatics/btad585)

# Citation
If your work uses our code or online website, please cite our paper.