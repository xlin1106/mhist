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

### The following code refers to building all the PyTorch models

### THIS MYMODEL PYTORCH VERSION IS NOT WORKING
def build_mymodel(input_shape=(224,224,3)):
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


def build_resnet50(input_shape=(224, 224, 3)):
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


def build_mobilenet(input_shape=(224,224,3)):
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


def build_efficientnet(input_shape=(224,224,3)):
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