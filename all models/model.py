import keras
from keras import layers
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.efficientnet_v2 import EfficientNetV2S
from keras.layers import Dense
from keras.models import Model
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import MobileNet_V2_Weights
from torchvision.models import EfficientNet_V2_S_Weights

### The following code refers to building all of the Tensorflow models

# Create a model based on the MNIST digits model
def build_mymodel_t(input_shape=(224, 224, 3)):
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

# Create the ResNet18 model
def build_resnet18_t(input_shape=(224, 224, 3)):
    model = keras.Sequential([
        # Initial convolution and max-pooling
        layers.Conv2D(64, kernel_size=(7, 7), strides=2, padding="same", activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'),

        # Residual Block 1
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        # Adding a shortcut (identity mapping) for ResNet structure
        layers.Conv2D(64, kernel_size=(1, 1), padding="same", activation='relu'),

        # Residual Block 2
        layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        # Adding a shortcut (identity mapping) for ResNet structure
        layers.Conv2D(128, kernel_size=(1, 1), strides=2, padding="same", activation='relu'),

        # Residual Block 3
        layers.Conv2D(256, kernel_size=(3, 3), strides=2, padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, kernel_size=(3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        # Adding a shortcut (identity mapping) for ResNet structure
        layers.Conv2D(256, kernel_size=(1, 1), strides=2, padding="same", activation='relu'),

        # Residual Block 4
        layers.Conv2D(512, kernel_size=(3, 3), strides=2, padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(512, kernel_size=(3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        # Adding a shortcut (identity mapping) for ResNet structure
        layers.Conv2D(512, kernel_size=(1, 1), strides=2, padding="same", activation='relu'),

        # Global Average Pooling
        layers.GlobalAveragePooling2D(),

        # Dense and Dropout layers
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),  # Use 'sigmoid' for binary classification
    ])

    return model

# Create the ResNet50 model
def build_resnet50_t(input_shape=(224, 224, 3)):
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

# Create the MobileNet model
def build_mobilenet_t(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(
        input_shape=None,
        alpha=1.0,
        include_top=False,
        weights=None,
        input_tensor=None,
        pooling='avg',
        classes=1,
        classifier_activation="sigmoid"
    )

    x = base_model.output
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)

    return model

# Create the EfficientNet model
def build_efficientnet_t(input_shape=(224, 224, 3)):
    base_model = EfficientNetV2S(
        input_shape=None,
        include_top=False,
        weights=None,
        input_tensor=None,
        pooling='avg',
        classes=1,
        include_preprocessing=True,
        classifier_activation="sigmoid"
    )

    x = base_model.output
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)

    return model

### The following code refers to building all the PyTorch models

# Create the pytorch version for the MNIST digit model
def build_mymodel_p(input_shape=(224,224,3)):
    class MHISTModel(nn.Module):
        def __init__(self):
            super(MHISTModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 10, kernel_size=7, padding=0)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(10, 10, kernel_size=7, padding=0)
            self.dropout = nn.Dropout(0.5)

            # Final fully connected layer - output 1 unit for binary classification
            self.fc1 = None  # Will initialize dynamically based on input size

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))  # Conv -> Relu -> Pooling
            x = self.pool(torch.relu(self.conv2(x)))  # Conv -> Relu -> Pooling
            
            # Calculate the flattened size dynamically
            if self.fc1 is None:
                num_features = x.size(1) * x.size(2) * x.size(3)
                self.fc1 = nn.Linear(num_features, 1).to(x.device)  # One output unit for binary classification

            x = x.view(x.size(0), -1)  # Flatten into 1-D
            x = self.dropout(x)
            x = self.fc1(x)
            return torch.sigmoid(x)  # Sigmoid output for binary classification

    model = MHISTModel()
    return model

# Create the pytorch version of the ResNet18 model
def build_resnet18_p(input_shape=(224, 224, 3)):
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

# Create the pytorch version of the ResNet50 model
def build_resnet50_p(input_shape=(224, 224, 3)):
    class ModifiedResNet50(nn.Module):
        def __init__(self):
            super(ModifiedResNet50, self).__init__()
            # Load the pre-trained ResNet50 model
            self.resnet = models.resnet50(weights='IMAGENET1K_V1')
            
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
            
    model = ModifiedResNet50()
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model

# Create the pytorch version of the MobileNet model
def build_mobilenet_p(input_shape=(224,224,3)):
    class ModifiedMobileNet(nn.Module):
        def __init__(self):
            super(ModifiedMobileNet, self).__init__()
            
            # Load the pre-trained MobileNetV2 model
            self.mobnet = models.mobilenet_v2(weights='IMAGENET1K_V1')
            
            # Modify the final classifier layer
            num_ftrs = self.mobnet.classifier[1].in_features  # Access the in_features of the classifier
            self.mobnet.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 512),  # Intermediate layer with 512 units
                nn.ReLU(),                 # Activation
                nn.Dropout(0.5),           # Dropout layer with p=0.5
                nn.Linear(512, 2)          # Final layer for binary classification (2 classes)
            )

        def forward(self, x):
            # Forward pass through MobileNetV2
            x = self.mobnet(x)
            return x
            
    # Instantiate the model
    model = ModifiedMobileNet()
    return model

# Create the pytorch version of the EfficientNet model
def build_efficientnet_p(input_shape=(224,224,3)):
    class ModifiedEfficientNet(nn.Module):
        def __init__(self):
            super(ModifiedEfficientNet, self).__init__()
            
            # Load the pre-trained EfficientNetV2-S model
            self.effnet = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
            
            # Modify the final classifier layer
            num_ftrs = self.effnet.classifier[1].in_features  # Access the in_features of the classifier
            self.effnet.classifier = nn.Sequential(
                nn.Dropout(0.5),           # Dropout layer with p=0.5
                nn.Linear(num_ftrs, 512),  # Intermediate layer with 512 units
                nn.ReLU(),                 # ReLU activation
                nn.Linear(512, 2)          # Final layer for binary classification (2 classes)
            )

        def forward(self, x):
            # Forward pass through EfficientNetV2
            x = self.effnet(x)
            return x
            
    # Instantiate the model
    model = ModifiedEfficientNet()
    return model
