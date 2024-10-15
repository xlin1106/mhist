## **MHIST: Histopathology Image Analysis with Deep Learning Models**
Some factors that challenge histopathology research include having high-resolution, variable-size images, high costs of annotation, and unclear annotation guidelines.
We used MHIST, a minimalist histopathology image classification dataset, to train deep-learning models for histopathology image analysis to address these issues.
The aim of this project is to evaluate the performance of various models in recognizing tissue sections across two different classes.

This repository contains the framework for multiple deep-learning models created through Tensorflow and Pytorch.
We will use ResNet18, based on the BMIRDS MHIST guidelines, as our baseline model. 

*For questions about the code, please open an issue on this code repository.*

## **Requirements**
- Python 3.7+
- cudatoolkit=11.3
- Tensorflow
- Pytorch
- Numpy
- pandas
- matplotlib
- scikit-learn
- scikit-image
- Any GPU

It is recommended to create a conda environment to set up all the packages and dependencies in one location.

## **Usage**
To run the program, use the train.py script.
Please take a look at the model.py script to see what models you can choose from.
The names written following "build" are the possible arguments taken.

List of model names to choose from (case sensitive):
- mymodel
- resnet18
- resnet50
- mobilenet
- efficientnet

All models are created with a PyTorch and Tensorflow version. Please specify which version you'd like to use using the flag -t.

If the program does not run, check to make sure that you have a folder called "data" that contains the annotations.csv spreadsheet and another folder called "images" for all the images of the tissues.
