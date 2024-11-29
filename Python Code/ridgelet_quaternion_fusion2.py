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
import random
import seaborn as sns
from sklearn import metrics
from torchsummary import summary
from fullyconnectedlayer import createreallayer, connectlayers, createquatlayer, declarefirstlayer, forwardlayerreal, forwardlayerquat
from convolutionallayer import createquatconvlayer, forwardconv, quatmaxpool
from tqdm import tqdm

# Enable CUDA memory configuration and autograd anomaly detection
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.autograd.set_detect_anomaly(True)


high = 128
low = 32
# Custom Dataset
import random

class NWPU_RESISC45(Dataset):
    def __init__(self, root_dir, classes, transform=None, samples_per_class=None):
        """
        Args:
            root_dir (str): Path to the dataset root directory.
            classes (list): List of class names to include.
            transform (callable, optional): Optional transform to be applied on a sample.
            samples_per_class (int, optional): Maximum number of images to use per class. Use all if None.
        """
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.img_paths = []
        self.labels = []

        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            class_files = os.listdir(class_dir)
            
            # Optionally limit the number of images per class
            if samples_per_class is not None:
                class_files = random.sample(class_files, min(samples_per_class, len(class_files)))

            for img_name in class_files:
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

    
class RidgeletLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_kernels=8):
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


# Mixed Resolution CNN with Quaternion Convolution for Low Resolution
class MixedResolutionCNN(nn.Module):
    def __init__(self, num_classes=5, num_kernels=8):
        super(MixedResolutionCNN, self).__init__()
        self.num_classes = num_classes

        # Quaternion convolution for low resolution part
        self.conv1 = createquatconvlayer(number=3, size=3)

        # High-resolution processing with ridgelet layer (unchanged)
        self.high_res_ridgelet = RidgeletLayer(in_channels=3, out_channels=16, kernel_size=19, num_kernels=8)
        self.high_res_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate output size
        self._conv_output_size = self._get_conv_output_size()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(66436, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.num_classes)
    
    def _get_conv_output_size(self):
        device = next(self.parameters()).device
        dummy_low_res = torch.zeros(1, 3, low, low, device=device)  # Low-resolution dummy input
        dummy_high_res = torch.zeros(1, 3, high, high, device=device)  # High-resolution dummy input

        # Low-resolution feature extraction (using quaternion convolution)
        x_low = F.interpolate(dummy_low_res, size=(low, low), mode='bilinear', align_corners=False)
        x_low = forwardconv(x_low, self.conv1)  # Using quaternion convolution
        x_low = quatmaxpool(x_low, size=2)
        x_low = x_low.permute(0, 2, 1, 3, 4)
        #x_low = x_low.reshape(x_low.size(0), -1)
        x_low = x_low.reshape(x_low.size(0), -1)

        # High-resolution feature extraction
        x_high = F.relu(self.high_res_ridgelet(dummy_high_res))
        x_high = self.high_res_pool(x_high)
        x_high = x_high.reshape(x_high.size(0), -1)

        return x_low.size(1) + x_high.size(1)


    def forward(self, x):
        # Low-resolution processing with quaternion convolution
        x_low = F.interpolate(x, size=(low, low), mode='bilinear', align_corners=False)
        convout = forwardconv(x_low, self.conv1)
        convout = quatmaxpool(convout, 3, 3)
        x_low = convout.permute(0, 2, 1, 3, 4)
        #x_low = x_low.view(x_low.size(0), -1)
        x_low = x_low.reshape(x_low.size(0), -1)

        # High-resolution processing (ridgelet layer)
        x_high = F.relu(self.high_res_ridgelet(x))
        x_high = self.high_res_pool(x_high)
        x_high = x_high.view(x_high.size(0), -1)

        # Fusion of high and low resolution
        x_fused = torch.cat((x_low, x_high), dim=1)

        x_fused = F.relu(self.dropout(self.fc1(x_fused)))
        x_fused = F.relu(self.fc2(x_fused))
        x_fused = self.fc3(x_fused)

        return x_fused

def print_model_params(model):
    print("Model parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Name: {name}, Shape: {param.shape}")
# Function for model training
from tqdm import tqdm  # Add this import at the top of your script

def train_model(trainloader, device, epochs=20, accumulation_steps=4):
    model = MixedResolutionCNN(num_classes=5).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
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

        # Add a progress bar using tqdm
        with tqdm(total=len(trainloader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for batch_idx, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

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

                # Update progress bar
                pbar.set_postfix(loss=loss.item(), accuracy=correct_predictions / total_samples)
                pbar.update(1)  # Increment progress bar

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
    torch.backends.cudnn.benchmark = True
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize((high, high)),  # Resize to a fixed size (adjust as needed)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Directory for NWPU-RESISC45 dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, "dataset", "NWPU-RESISC45")  # Update this path to where the dataset is located
    
    # Define the classes to use
    classes = ['airplane', 'railway', 'ship', 'palace', 'bridge']  # Example classes
    #classes = ['airplane', 'railway', 'ship', 'palace', 'bridge', 'island', 'freeway', 'river', 'stadium', 'church']
    
    # Load the dataset
    dataset = NWPU_RESISC45(root_dir=root_dir, classes=classes, transform=transform, samples_per_class=200)
    
    # Create DataLoader
    trainloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    
    # Training the model
    num_kernels = 8  # Adjust the number of kernels here
    model, train_losses, train_accuracies = train_model(trainloader, device, epochs=20, accumulation_steps=4)
    #summary(model, input_size=(3, high, high), device=str(device))
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