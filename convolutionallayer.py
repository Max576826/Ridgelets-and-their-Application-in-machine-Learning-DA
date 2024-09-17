import torch
import numpy as np
import gc
import math

# creation of a single convolutional matrix
def createquatconv(size):
    convparams = torch.zeros(size, size, 2)
    for i in range(size):
        for j in range(size):
	    			# initializing the angle theta
            convparams[i, j, 0]=torch.tensor(np.random.uniform(
									-np.pi/2, np.pi/2))
	    			# initializing the scale s
            convparams[i, j, 1]=torch.tensor(np.random.uniform(0.0,2.0))
    return convparams


# creation of an entire layer of similar size convolutions
def createquatconvlayer(number, size):
    convparams = torch.zeros(number, size, size, 2)
		# calling 'createquatconv' for every convolution matrix of the layer
    for i in range(number):
        convparams[i, :, :, :] = createquatconv(size)
    convparams.requires_grad = True
    return convparams


def forwardconv(input, convparams):
    device = input.device
    
	# checking whether it is the first convolutional layer, in this case 
	# the input has no dimension stacking the different 
	# convolutional outputs of the previous layer
    if len(input.size()) == 4:
		# getting the number of convolutions
        number = len(convparams[:, 0, 0, 0])
		# getting the size of the convolutions
        size = len(convparams[0, :, 0, 0])
		# preparation of the output having dimensions for the batchsize, 
		# the number of convolutions
		# and the precalculated size of convolution output matrix 
        output = torch.zeros(len(input[:, 0, 0, 0]), number, 3, 
					len(input[0, 0, :, 0]) - size + 1, 
					len(input[0, 0, 0, :]) - size + 1)
        output = output.to(device)
		# preparing the rotation for every filter matrix and 
		# all corresponding pixels, using the angle theta from convparams
        f1 = 1/3+2/3*torch.cos(convparams[:,:,:,0])
        f2 = 1/3-2/3*torch.cos(convparams[:,:,:,0]-torch.tensor(np.pi)/3)
        f3 = 1/3-2/3*torch.cos(convparams[:,:,:,0]+torch.tensor(np.pi)/3)
        ff = torch.stack([f1,f2,f3,f3,f1,f2,f2,f3,f1], dim=3).unflatten(dim=3, sizes=(3, 3))
        ff = ff.to(device)
		# stacking the scale from convparams for each rotation three times
		# to facilitate multiplication with the tensor later on
        scal = torch.stack([convparams[:, :, :, 1], convparams[:, :, :, 1], 
					convparams[:, :, :, 1]])
        scal = scal.permute(1,0,2,3)
        scal = scal.to(device)
		# preparing the input to fit the tensor multiplication
        input = input.permute(1, 0, 2, 3)

		# looping over the dimensions of the filter
        for p in range(size):
            for l in range(size):
				# looping over the output of the convolution
                for i in range(len(output[0, 0, 0, :, 0])):
                    for j in range(len(output[0, 0, 0, 0, :])):
						# calculating the rotated values
                        rot = torch.matmul(ff[:, p, l, :, :], 
													input[:, :, i + p, j + l]).to(device)
                        rot = rot.permute(2, 0, 1)
						# scaling and adding rotated values to the output
                        output[:,:,:,i,j] = torch.add(output[:,:,:,i,j], 
													scal[:, :, p, l]/(size*size) * rot[:, :, :])

	# if the input is 5-dimensional we already had different convolutions 
	# and need to loop over the according outputs as well
    else:
		# getting the number of different inputs from the last convolution
        inputnum = len(convparams[:, 0, 0, 0])
		# getting the number of convolutions to do in this layer
        filtnum = len(convparams[:, 0, 0, 0])
		# calculating the resulting number of different convolution outputs
        number = inputnum * filtnum
		# getting the size of the convolutions
        size = len(convparams[0, :, 0, 0])
		# preparation of the output having dimensions for the batchsize, 
		# the number of convolutions
		# and the precalculated size of convolution output matrix 
        output = torch.zeros(len(input[:, 0, 0, 0, 0]), number, 3, 
					len(input[0, 0, 0, :, 0]) - size + 1, 
					len(input[0, 0, 0, :, 0]) - size + 1).to(device)
		# preparing the rotation for every filter matrix and 
		# all corresponding pixels, using the angle theta from convparams
        f1 = 1/3+2/3*torch.cos(convparams[:,:,:,0])
        f2 = 1/3-2/3*torch.cos(convparams[:,:,:,0]-torch.tensor(np.pi)/3)
        f3 = 1/3-2/3*torch.cos(convparams[:,:,:,0]+torch.tensor(np.pi)/3)
        ff = torch.stack([f1,f2,f3,f3,f1,f2,f2,f3,f1], dim=3).unflatten(dim=3, sizes=(3, 3)).to(device)
		# stacking the scale from convparams for each rotation three times
	    # to facilitate multiplication with the tensor later on
        scal = torch.stack([convparams[:, :, :, 1], convparams[:, :, :, 1], 
					convparams[:, :, :, 1]]).to(device)
        scal = scal.permute(1,0,2,3)
		# preparing the input to fit the tensor multiplication
        input = input.permute(2, 0, 1, 3, 4)

		# looping over the dimensions of the filter
        for p in range(size):
            for l in range(size):
				# looping over the output of the convolution
                for i in range(len(output[0, 0, 0, :, 0])):
                    for j in range(len(output[0, 0, 0, 0, :])):
						# looping over the different convolution
						# outputs of the last layer
                        for t in range(inputnum):
							# calculating the rotated values
                            rot = torch.matmul(ff[:, p, l, :, :], 
														input[:, :, t, i + p, j + l]).to(device)
                            rot = rot.permute(2,0,1)
							# scaling the values and
							# adding to the output
                            output[:,t*filtnum:(t+1)*filtnum,:,i,j] = torch.add(
															output[:,t*filtnum:(t+1)*filtnum,:,i,j],
															scal[:, :, p, l]/(size*size) * rot[:,:, :]).to(device)
                            
	# manually deleting big tensors that are no longer needed
    del input, number, size, scal, ff, rot, f1, f2, f3
    gc.collect()
    return output


# pooling the quaternion image data after convolution
# selecting the quaternion pixels with the biggest amplitude
import torch
import numpy as np
import gc
import math

# pooling the quaternion image data after convolution
# selecting the quaternion pixels with the biggest amplitude
def quatmaxpool(input, size, stride=1):
    device = input.device
    output = torch.zeros(
        len(input[:, 0, 0, 0, 0]),
        len(input[0, :, 0, 0, 0]),
        3,
        math.floor((len(input[0, 0, 0, :, 0]) - size) / stride) + 1,
        math.floor((len(input[0, 0, 0, :, 0]) - size) / stride) + 1,
        device=device
    )
    amplitudes = torch.zeros(size, size, device=device)
    for k in range(len(input[:, 0, 0, 0, 0])):
        for c in range(len(input[0, :, 0, 0, 0])):
            for i in range(math.floor((len(input[0, 0, 0, :, 0]) - size) / stride) + 1):
                for j in range(math.floor((len(input[0, 0, 0, :, 0]) - size) / stride) + 1):
                    amplitudes = torch.square(input[k, c, :, i * stride:i * stride + size, j * stride:j * stride + size]).sum(axis=0)
                    maxcount = torch.argmax(amplitudes).to(device)
                    
                    # Move maxcount to CPU before using NumPy
                    maxcount_cpu = maxcount.cpu().numpy()
                    
                    x, y = np.unravel_index(maxcount_cpu, (size, size))
                    output[k, c, :, i, j] = input[k, c, :, i * stride + x, j * stride + y]
                    
    del amplitudes, maxcount, x, y
    gc.collect()
    return output

