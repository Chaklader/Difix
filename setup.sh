#!/bin/bash

mkdir ~/datasets
cd ~/datasets

# install gdown to download images from Google Drive
pip install gdown

# download images from Google Drive, unzip and this will create a folder called colmap_workspace
gdown https://drive.google.com/uc?id=1vEqISXwdgryUMWY5kad89HHuqHdtYzo4 -O images.zip 
unzip images.zip 

mkdir colmap_processed