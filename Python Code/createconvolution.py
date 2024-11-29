from fullyconnectedlayer import createreallayer, connectlayers, createquatlayer, declarefirstlayer, forwardlayerreal, forwardlayerquat
from convolutionallayer import createquatconvlayer, forwardconv, quatmaxpool
import torch
import numpy as np
import matplotlib.pyplot as plt

# create two convolutions with each 5 filters
# one with convolution filters of size 3x3 and one with size 4x4
conv1 = createquatconvlayer(5, 3)
conv2 = createquatconvlayer(5, 4)

# create a random input of batchsize 10 for each 20x20 RGB pixels
input = torch.rand(10, 3, 20, 20)

# passing the input through the first convolution
convout1 = forwardconv(input, conv1)

# pooling the convolutional output with 2x2 max amplitude pooling
convout1pool = quatmaxpool(convout1, 2, 2)

# passing the input through the second convolution
convout2 = forwardconv(convout1pool, conv2)

# the output of the convolutional layer could now be passed to fully connected layers

# printing examples of what the convolution and pooling does to the random images
plt.subplot(321)
plt.imshow(input[0,:,:,:].permute(1,2,0))
plt.xlabel("random input")
plt.xticks([])
plt.yticks([])
plt.subplot(323)
plt.imshow(convout1[0,0,:,:,:].permute(1,2,0).detach())
plt.xlabel("after 3x3 convolution")
plt.xticks([])
plt.yticks([])
plt.subplot(325)
plt.imshow(convout1pool[0,0,:,:,:].permute(1,2,0).detach())
plt.xlabel("after 2x2 pooling")
plt.xticks([])
plt.yticks([])
plt.subplot(322)
plt.imshow(input[0,:,:,:].permute(1,2,0))
plt.xlabel("random input")
plt.xticks([])
plt.yticks([])
plt.subplot(324)
plt.imshow(convout1[0,1,:,:,:].permute(1,2,0).detach())
plt.xlabel("after different 3x3 convolution")
plt.xticks([])
plt.yticks([])
plt.subplot(326)
plt.imshow(convout1pool[0,1,:,:,:].permute(1,2,0).detach())
plt.xlabel("after 2x2 pooling")
plt.xticks([])
plt.yticks([])
plt.show()
