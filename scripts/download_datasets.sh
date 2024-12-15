#!/bin/bash

# Adapted from https://github.com/ssundaram21/dreamsim/blob/main/dataset/download_dataset.sh.
mkdir -p /tmlscratch/fcroce/datasets
cd /tmlscratch/fcroce/datasets

# Download NIGHTS dataset
wget -c -O nights.zip https://data.csail.mit.edu/nights/nights.zip

#unzip nights.zip
#rm nights.zip

# THINGS dataset.
wget https://osf.io/n9u4a  # csv file
wget https://files.osf.io/v1/resources/jum2f/providers/osfstorage/5d4ec190265be2001830a77e/?zip=  # images

# KADID-10k IQA Database (https://database.mmsp-kn.de/kadid-10k-database.html).
wget https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip

# LICQA IQA Dataset https://github.com/sherlockyy/LICQA
gdown 1RpwqOE6J-dNa6Hun9wxG7dJEqXiLvtlM

# CUB 200 2011 from https://www.kaggle.com/datasets/coolerextreme/cub-200-2011.
pip install kaggle
kaggle datasets download -d coolerextreme/cub-200-2011

#Spot-the-diff 
wget https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data/resolve/main/Spot-the-Diff.zip

#Birds-to-words
wget https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data/resolve/main/Birds-to-Words.zip

#Flickr-8k https://drive.google.com/file/d/17JYIVKF2qj6DdLSzRzPH6IjVKczHpXSf/view?usp=drive_link
gdown 17JYIVKF2qj6DdLSzRzPH6IjVKczHpXSf

