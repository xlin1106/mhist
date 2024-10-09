import keras
from keras import layers
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.models import Model
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights

def build_mymodel(input_shape=(224, 224, 3)):
    #create a model based off the mnist model
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(3, kernel_size=(7, 7), activation="relu"),  #downsize the image size
            layers.MaxPooling2D(pool_size=(2, 2)), 
            layers.Conv2D(10, kernel_size=(7, 7), activation="relu"), #downsize again
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(), #flatten into 1-D
            layers.Dropout(0.2), #dropout rate of 20% probability
            layers.Dense(1, activation="sigmoid"), #put it into the 1 element, and use sigmoid to turn it into 0 or 1
        ]
    )

    return model


def build_resnet50(input_shape=(224, 224, 3)):
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling='avg',
        classifier_activation="sigmoid",
        )

    x = base_model.output
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)

    return model


def build_mobilenet(input_shape=(224, 224, 3):
    base_model = MobileNet(
        input_shape=None,
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=1,
        classifier_activation="sigmoid"
    )

    x = base_model.output
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)

    return model


def build_vgg16(input_shape=(224, 224, 3):
    base_model = VGG16(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1,
        classifier_activation="sigmoid",
        name="vgg16",
    )

    x = base_model.output
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)

    return model


def build_resnet18(input_shape=(224, 224, 3)):
    class ModifiedResNet18(nn.Module):
        def __init__(self):
            super(ModifiedResNet18, self).__init__()
            # Load the pre-trained ResNet-18 model
            self.resnet = models.resnet18(weights='IMAGENET1K_V1')
            
            # Modify the final fully connected layer
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, 512)  # Intermediate layer
    
            # Add Dropout and new fully connected layer for binary classification
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(512, 2)  # Adjust for 2-class classification (hp and ssa)
    
        def forward(self, x):
            # Pass input through resnet layers
            x = self.resnet(x)
            # Apply dropout
            x = self.dropout(x)
            # Pass through final fully connected layer for 2-class classification
            x = self.fc(x)
            return x

    model = ModifiedResNet18()
    model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust the final layer for binary classification

    return model