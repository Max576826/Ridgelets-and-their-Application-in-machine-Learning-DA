from fullyconnectedlayer import createreallayer, connectlayers, createquatlayer, declarefirstlayer, forwardlayerreal, forwardlayerquat
import torch
import numpy as np

# create two quaternion and one real layer
layer1 = createquatlayer(20)
layer2 = createquatlayer(10)
# output of last layer is of size 4 representing 4 classes
layer3 = createreallayer(4)

# declare input size to first layer as 20 pixels
params1 = declarefirstlayer(20, layer1)
# initialize parameters between layers
params12 = connectlayers(layer1, layer2)
params23 = connectlayers(layer2, layer3)

# create a random input of batchsize 10 for each 20 RGB pixels
input = torch.rand(10, 3, 20)

# pass the input through the quaternion layers
out1 = forwardlayerquat(input, params1)
out2 = forwardlayerquat(out1, params12)
# flatten output to fit the real layer
out2flat = torch.transpose(out2, 2, 1).flatten(start_dim=1)
# passing though real layer and get output for the 4 classes for each of the 10 elements of the batch
out = forwardlayerreal(out2flat, params23)
print(out)

# choose class with maximum value as prediction
prediction = torch.argmax(out, 1)
print(prediction)
