import numpy as np
import os
import pandas as pd
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import ExponentialLR

import model 

def get_model(model_name):
    try:
        name = model_name
        # Dynamically get the corresponding model function from model.py
        model_function = getattr(model, f'build_{name}')
        return model_function(input_shape=(224, 224, 3))
    except AttributeError as e:
        raise ValueError(f"Model {model_name} not found. Error: {str(e)}")

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name to train.')
    parser.add_argument('-e', '--epoch', type=int, default=50, help='Number of epoches to train.')
    args = parser.parse_args()

    model = get_model(args.model)

    data_dir = './data/'

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {count_parameters(model)}")

        
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
            confidence = self.labels.iloc[idx, 2] / 7.0
        
            if self.transform:
                image = self.transform(image)
        
            return image, label, confidence

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
        
        for inputs, labels, confidences in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss = (loss * confidences).mean() 
                
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
            for inputs, labels, confidences in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = (loss * confidences).mean()
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
            
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

        # Convert probabilities to binary predictions (0 or 1)
        binary_preds = [1 if p > 0.5 else 0 for p in all_preds]

        # Confusion Matrix
        cm = confusion_matrix(all_labels, binary_preds)

        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HP', 'SSA'], yticklabels=['HP', 'SSA'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        # Accuracy
        accuracy = accuracy_score(all_labels, binary_preds)
        print(f"Final Accuracy: {accuracy:.2f}")
            
print("DONE!!")

if __name__ == '__main__':
    main()
