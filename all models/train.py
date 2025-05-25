import tensorflow as tf
import numpy as np
import os
import pandas as pd
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import ExponentialLR

import model 

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

path_mean = np.array([94.07080238, 82.69394267, 98.84364401])
path_std = np.array([23.95747985, 29.23472607, 20.76279442])

# Create the normalization function to normalize every image input
def normalization(image):
    path_mean = np.array([94.07080238, 82.69394267, 98.84364401])
    path_std = np.array([23.95747985, 29.23472607, 20.76279442])
    return (image - path_mean) / (path_std + 1e-7)

def denormalize(image, mean, std):
    mean = tf.reshape(tf.constant(mean), (1, 1, 1, 3))
    std = tf.reshape(tf.constant(std), (1, 1, 1, 3))
    return image * std + mean

# Create a preprocessing function for augmentation
def custom_preprocessing(image, path_mean, path_std, augmentation=False):
    if augmentation:
        image = tf.image.random_brightness(image, max_delta=0.2)  
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2) 
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.1) 

    image = (image - path_mean) / (path_std + 1e-7)
    return image

# Create a function to get the specific model and version from model.py
def get_model(model_name, add_type):
    try:
        name = model_name + '_' + add_type
        # Dynamically get the corresponding model function from model.py
        model_function = getattr(model, f'build_{name}')
        return model_function(input_shape=(224, 224, 3))
    except AttributeError as e:
        raise ValueError(f"Model {model_name} not found. Error: {str(e)}")

# Main function call
# This is where training and testing the model occurs
def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name to train.')
    parser.add_argument('-e', '--epoch', type=int, default=50, help='Number of epoches to train.')
    parser.add_argument('-t', '--type', type=str, required=True, help='Use tensorflow or pytorch.')
    args = parser.parse_args()

    # Get the specified model
    model_type = args.type
    if model_type == 'tensorflow':
        add_type = 't'
    elif model_type == 'pytorch':
        add_type = 'p'
    model = get_model(args.model, add_type)

    data_dir = './data/'

    """ IMPORTANT: The following lines of code are used for TENSORFLOW models
        Loading the data file using os and glob
        Make sure there is a folder called "data" on your device that contains the annotations spreadsheet and all the images
        The images should be in their own folder called "images" within the "data" folder """
    if model_type == 'tensorflow':
        train_datagen = ImageDataGenerator(
            rotation_range=20,  
            width_shift_range=0.05,  
            height_shift_range=0.05,  
            shear_range=0.05,  
            zoom_range=0.05,  
            horizontal_flip=True,  
            fill_mode='nearest',
            preprocessing_function=normalization
        )
    
        valid_datagen = ImageDataGenerator(
            # rescale=1./255,  
            preprocessing_function=normalization
        )
    
        annotations = pd.read_csv(os.path.join(data_dir, "annotations.csv")) #import the spreadsheet and save it into a variable

        # Split the data into train set and test set
        train_df = annotations[annotations['Partition'] == 'train']
        test_df = annotations[annotations['Partition'] == 'test']

        # Load all the training data images
        train_loader = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory='./data/images',
            x_col="Image Name",
            y_col="Majority Vote Label",
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            shuffle=True
        )

        # Load all the testing data images
        test_loader = valid_datagen.flow_from_dataframe(
            dataframe=test_df,
            directory='./data/images',
            x_col="Image Name",
            y_col="Majority Vote Label",
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            shuffle=False
        )
    
        batch_size = 128
    
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        # Train the model
        history = model.fit(
            x=train_loader,
            y=None,
            batch_size=batch_size,
            epochs=args.epoch,
            verbose="auto",
            callbacks=None,
            validation_split=0.0,
            validation_data=test_loader,
            shuffle=True,
            class_weight={0:0.8, 1:1.2},
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
        )

        # Display the model loss and accuracy
        score = model.evaluate(test_loader, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
    
        testpredictions = []
        testgroundtrues = []
    
        for x,y in test_loader:
            y_pred = model(x)
            testpredictions.append(y_pred)
            testgroundtrues.append(y)
    
        testpredictions = np.concatenate(testpredictions)
        testgroundtrues = np.concatenate(testgroundtrues)
    
        print("The accuracy score for the test loader is: ", accuracy_score(testgroundtrues, testpredictions.round()))
        print("The F1 score for the test loader is: ", f1_score(testgroundtrues, testpredictions.round()), "\n")
    
        conf = confusion_matrix(testgroundtrues, testpredictions.round())
        print("The confusion matrix for the test loader is: \n", conf, "\n")
        print("The sensitivity score of SSA for the test loader is: ", conf[0,0]/(conf[0,0]+conf[0,1]))
        print("The specificity score of HP for the test loader is: ", conf[1,1]/(conf[1,1]+conf[1,0]))
    
        #calculate ROC curve for test loader
    
        fpr, tpr, thresholds = roc_curve(testgroundtrues, testpredictions) 
        roc_auc = auc(fpr, tpr)
        print("The AUC score of the test loader is: ", roc_auc)
    
        # plot the ROC curve
        plt.figure()  
        plt.plot(fpr, tpr, label='Train ROC curve' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--', label='No Training')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
    
        # Save the accuracy plot as a file
        plt.savefig('ROC.png')
        plt.clf()  # Clear the current figure to avoid overlap
    
    
        # Plot accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
    
        # Save the accuracy plot as a file
        plt.savefig('model_accuracy.png')
        plt.clf()  # Clear the current figure to avoid overlap
    
    
        # Plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
    
        # Save the loss plot as a file
        plt.savefig('model_loss.png')
        plt.clf()  # Clear the current figure
        
    elif model_type == 'pytorch':
        class MHISTDataset(Dataset):
            def __init__(self, img_dir, labels_file, transform=None):
                self.img_dir = img_dir
                self.labels = pd.read_csv(labels_file)
                self.transform = transform
        
            def __len__(self):
                return len(self.labels)
        
            def __getitem__(self, idx):
                img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])  # Image names from the first column
                image = Image.open(img_name)
        
                # Map 'hp' and 'ssa' to integer labels
                label = 0 if self.labels.iloc[idx, 1] == 'HP' else 1  # Assuming 'HP' -> 0 and 'SSA' -> 1
        
                if self.transform:
                    image = self.transform(image)
        
                return image, label

        # Define the transformations
        augmentation = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor()
        ])
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # Path to images and the CSV file
        img_dir = './data/images'
        labels_file = './data/annotations.csv'
        
        # Load the CSV file and split based on the 'Partition' column
        annotations = pd.read_csv(labels_file)
        train_data = annotations[annotations['Partition'] == 'train']
        test_data = annotations[annotations['Partition'] == 'test']
        
        # Save the train and test splits into separate CSV files (if needed)
        train_data.to_csv('train_annotations.csv', index=False)
        test_data.to_csv('test_annotations.csv', index=False)
        
        # Create Dataset instances
        train_dataset = MHISTDataset(img_dir=img_dir, labels_file='train_annotations.csv', transform=augmentation)
        test_dataset = MHISTDataset(img_dir=img_dir, labels_file='test_annotations.csv', transform=transform)
        
        # Create DataLoader instances
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, gamma=0.91)  # Learning rate decay factor of 0.91

        num_epochs = 100
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []
        all_labels = []
        all_preds = []   
        
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            training_loss = 0.0
            correct_train = 0
            total_train = 0
        
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            train_loss = training_loss / len(train_loader)
            train_acc = 100 * correct_train / total_train
            
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
            
            # Validation accuracy
            model.eval()  # Set model to evaluation mode
            validation_loss = 0.0
            correct_val = 0
            total_val = 0
        
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
        
                    # Store labels and predictions for ROC curve and AUC
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(outputs[:, 1].cpu().numpy())
        
            val_loss = validation_loss / len(test_loader)
            val_acc = 100 * correct_val / total_val
        
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Validation Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
            
            # Step the learning rate scheduler
            scheduler.step()
            
    print("DONE!!")

if __name__ == '__main__':
    main()
