import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RidgeletLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(RidgeletLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Learnable parameters for each output channel
        self.scales = nn.Parameter(torch.ones(out_channels))  # Scales
        self.directions = nn.Parameter(torch.linspace(0, math.pi, out_channels))  # Directions
        self.positions = nn.Parameter(torch.zeros(out_channels))  # Positions

    def ridgelet_function(self, x1, x2, direction, scale, position):
        # Calculate transformed coordinate
        transformed_coordinate = (x1 * torch.cos(direction) + x2 * torch.sin(direction) - position) / scale
        # Ridgelet value based on transformed coordinate
        ridgelet_value = torch.exp(-transformed_coordinate ** 2 / 2) - 0.5 * torch.exp(-transformed_coordinate ** 2 / 8)
        return ridgelet_value

    def create_ridgelet_kernel(self, device):
        # Initialize the kernel tensor
        kernel = torch.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size), device=device)

        # Get the grid of x1 and x2 values
        center = self.kernel_size // 2
        x1, x2 = torch.meshgrid(torch.arange(self.kernel_size) - center, torch.arange(self.kernel_size) - center)
        x1, x2 = x1.float().to(device), x2.float().to(device)

        # Loop through each output and input channel to generate the kernel
        for out_ch in range(self.out_channels):
            direction = self.directions[out_ch]  # Direction for each filter
            scale = self.scales[out_ch]  # Scale for each filter
            position = self.positions[out_ch]  # Position for each filter
            
            for in_ch in range(self.in_channels):
                kernel[out_ch, in_ch, :, :] = self.ridgelet_function(x1, x2, direction, scale, position)

        return kernel

    def forward(self, x):
        device = x.device
        # Generate the kernel at each forward pass
        kernel = self.create_ridgelet_kernel(device)
        # Perform convolution
        x = F.conv2d(x, kernel, padding=self.kernel_size // 2)
        return x

# Test Ridgelet Layer
if __name__ == "__main__":
    # Create a RidgeletLayer instance
    ridgelet_layer = RidgeletLayer(in_channels=3, out_channels=6, kernel_size=5)

    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image

    # Perform a forward pass
    output = ridgelet_layer(dummy_input)

    print("Output shape:", output.shape)
    print("Output tensor:", output)
