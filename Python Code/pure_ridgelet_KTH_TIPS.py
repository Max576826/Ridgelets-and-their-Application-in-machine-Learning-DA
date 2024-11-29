import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import math
import gc
import time
import seaborn as sns
from sklearn import metrics
from torchsummary import summary
import time  # Make sure to import the time module
high=200
# Ridgelet Layer Definition
class RidgeletLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_kernels=1):
        super(RidgeletLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

        # Initialize parameters for ridgelet functions for all kernels
        self.scales = nn.Parameter(torch.ones(out_channels * num_kernels))
        self.directions = nn.Parameter(torch.linspace(0, math.pi, out_channels * num_kernels))
        self.positions = nn.Parameter(torch.zeros(out_channels * num_kernels))

    def ridgelet_function(self, x1, x2, direction, scale, position):
        transformed_coordinate = (x1 * torch.cos(direction) + x2 * torch.sin(direction) - position) / scale
        ridgelet_value = torch.exp(-transformed_coordinate ** 2 / 2) - 0.5 * torch.exp(-transformed_coordinate ** 2 / 8)
        return ridgelet_value

    def create_ridgelet_kernel(self, device):
        kernel = torch.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size), device=device)
        center = self.kernel_size // 2
        x1, x2 = torch.meshgrid(torch.arange(self.kernel_size) - center, torch.arange(self.kernel_size) - center)
        x1, x2 = x1.float().to(device), x2.float().to(device)

        for out_ch in range(self.out_channels):
            for k in range(self.num_kernels):
                idx = out_ch * self.num_kernels + k
                direction = self.directions[idx]
                scale = self.scales[idx]
                position = self.positions[idx]

                ridgelet_kernel = self.ridgelet_function(x1, x2, direction, scale, position)
                kernel[out_ch, :, :, :] += ridgelet_kernel

        return kernel

    def forward(self, x):
        device = x.device
        kernel = self.create_ridgelet_kernel(device)
        x = F.conv2d(x, kernel, padding=self.kernel_size // 2)
        return x

# Updated Pure Ridgelet Network with Pooling, BatchNorm, and FC layers
class PureRidgeletNetwork(nn.Module):
    def __init__(self, num_classes=10, num_kernels=15, kernel_size=30):
        super(PureRidgeletNetwork, self).__init__()

        # Ridgelet Layer
        self.ridgelet = RidgeletLayer(in_channels=3, out_channels=32, kernel_size=60, num_kernels=15)

        # Batch Normalization Layer
        self.bn = nn.BatchNorm2d(32)

        # ReLU Activation Layer
        self.relu = nn.ReLU()

        # MaxPooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(320000, 256)  # Adjust input size according to the output shape
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Apply Ridgelet Layer
        x = self.ridgelet(x)

        # Apply Batch Normalization
        x = self.bn(x)

        # Apply ReLU Activation
        x = self.relu(x)

        # Apply MaxPooling
        x = self.pool(x)

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the output

        # Apply Fully Connected Layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Custom Dataset Class (KTH_TIPS)
class KTH_TIPS(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.img_paths = []
        self.labels = []

        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.img_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Model Training Function


def train_model(trainloader, device, epochs=25):
    model = PureRidgeletNetwork(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    model.train()
    train_losses = []
    train_accuracies = []
    epoch_times = []  # List to store epoch times

    # Measure the total training time
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_start_time = time.time()  # Start time for the current epoch
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_accuracy = correct_predictions / total_samples
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        scheduler.step(epoch_loss)

        # Measure the end time for the current epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        print(f"Epoch Time: {epoch_duration:.2f} seconds")

    # Measure the total training time
    total_training_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

    return model, train_losses, train_accuracies, epoch_times

# Function to plot confusion matrix (same as previous)
def plot_confusion_matrix(model, dataloader, classes, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cf_matrix = metrics.confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    plt.show()
# Main function to load data and train the model
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((high, high)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # KTH_TIPS Dataset and classes
    root_dir = r"C:\Users\MaxSc\Desktop\dataset\KTH_TIPS"  # Update this with the actual dataset path
    classes = ['aluminium_foil', 'brown_bread', 'corduroy', 'cotton', 'cracker', 'linen', 
               'orange_peel', 'sandpaper', 'sponge', 'styrofoam']

    # Load dataset
    dataset = KTH_TIPS(root_dir=root_dir, classes=classes, transform=transform)

    # Split dataset into training and testing sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train the model
    model, train_losses, train_accuracies = train_model(trainloader, device, epochs = 40)
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot confusion matrix after training
    plot_confusion_matrix(model, trainloader, classes, device)

if __name__ == "__main__":
    main()
