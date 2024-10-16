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

### Dataset
This code follows the dataset of images provided by the [BMIRDS Datasets](https://bmirds.github.io/MHIST/). This dataset should include 3,152 colorectal polyp images and an annotations.csv.

![Images of two different types of colorectal polyp images with their image name/label.](https://cdn.discordapp.com/attachments/948086387096293476/1295789549615845418/Screenshot_2024-10-15_094143.png?ex=670fed8f&is=670e9c0f&hm=6f68b0e5c7ccbf971661b67b4df84fabdfd1d37c4f621486196094b4ff666e29&)

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

If the program does not run, check to make sure that you have a folder called "data" that contains the annotations.csv spreadsheet and another folder called "images" for all the images of the tissues. Also ensure that you have downloaded the model.py script which contains all the function builds for every model and version.

## **Limitations**
- Slight differences between the Tensorflow and PyTorch version of models

## **Future Work**
- Contributions to this repository are welcome.
- If you have issues, please post in the issues section and we will try to address it.

## **Citations**
Jerry Wei, Arief Suriawinata, Bing Ren, Xiaoying Liu, Mikhail Lisovsky, Louis Vaickus, Charles Brown, Michael Baker, Naofumi Tomita, Lorenzo Torresani, Jason Wei, Saeed Hassanpour, “A Petri Dish for Histopathology Image Analysis”, International Conference on Artificial Intelligence in Medicine (AIME), 12721:11-24, 2021.
