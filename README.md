# Texture feature extractor

### Marco Godi

## Description
A texture feature extractor using code from https://github.com/mcimpoi/deep-fbanks for the "Deep Filter Banks for Texture Recognition, Description, and Segmentation". The interface is easier than the one from the paper (so you can easily extract features and encode them).

## Requirements
Tested on MATLAB 2017, should work on earlier versions.
Install the [deep-banks code](https://github.com/mcimpoi/deep-fbanks)
Change input (dataset.root, deep-fbanks folder) and output paths (features_folder) to suit your machine.
Run main.m to perform all operations

Optional to test on standard dataset:
Download the [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) dataset and change the dataset.root in the main file to point to it (you should modify the folder to put all the classes into a single "images" folder.

## Results
Using the features (with PCA disabled) for the SVM classification as in the original paper yields comparable results.


