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
import gc
import time
import seaborn as sns
from sklearn import metrics
from torchsummary import summary  # Import torchsummary

high = 100

# Custom Dataset for NWPU_RESISC45
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


# Convolutional Network with two convolution layers
class StandardCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(StandardCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Compute the size of the flattened input to the first fully connected layer
        self.fc1_input_size = 128 * (high // 8) * (high // 8)
        
        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


# Function for model training
def train_model(trainloader, device, epochs=20, accumulation_steps=4):
    model = StandardCNN(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    model.train()
    scaler = GradScaler()

    train_losses = []
    train_accuracies = []
    start_time = time.time()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_start_time = time.time()  # Start time for the epoch
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_accuracy = correct_predictions / total_samples
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        scheduler.step(epoch_loss)  # Use loss for scheduler step
        torch.cuda.empty_cache()
        gc.collect()

        # Measure and print the epoch duration
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Duration: {epoch_duration:.2f} seconds")

    # Measure and print the total training time
    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    return model, train_losses, train_accuracies


# Function to plot confusion matrix
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


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize((high, high)),  # Resize to a fixed size (adjust as needed)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # KTH_TIPS Dataset and classes
    root_dir = r"C:\Users\MaxSc\Desktop\dataset\KTH_TIPS"  # Update this with the actual dataset path
    classes = ['aluminium_foil', 'brown_bread', 'corduroy', 'cotton', 'cracker', 'linen', 
               'orange_peel', 'sandpaper', 'sponge', 'styrofoam']

    # Load dataset
    dataset = KTH_TIPS(root_dir=root_dir, classes=classes, transform=transform)

    # Split dataset into training and testing sets (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Data loaders
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Training the model
    model, train_losses, train_accuracies = train_model(trainloader, device, epochs=20, accumulation_steps=4)
    
    # Visualize model architecture
    print(summary(model, (3, high, high)))

    # Plot training results
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
