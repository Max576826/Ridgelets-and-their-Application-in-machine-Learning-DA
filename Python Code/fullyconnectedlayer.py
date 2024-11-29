import torch
import numpy as np
import gc

# prepare the entity quaternion neuron
class QuatNeuron:
    pass

# create a whole layer of quaternion neurons passing the desired width
def createquatlayer(numberneurons):
    layer = []
    for i in range(numberneurons):
        this = QuatNeuron()
        layer.append(this)
    return layer

# prepare the entity real neuron
class RealNeuron:
    pass


# create a whole layer of real neurons passing the desired width
def createreallayer(numberneurons):
    layer = []
    for i in range(numberneurons):
        this = RealNeuron()
        layer.append(this)
    return layer


# connecting two adjacent layers and initializing all necessary parameters
# this version of the function is only useful for two quaternion layers
# quaternion to real connection will be implemented seperately
def connectquatlayers(layer1, layer2):
    params = torch.zeros(len(layer1), len(layer2), 2)
    for i in range(len(layer1)):
        for j in range(len(layer2)):
            # initializing the angle theta
            params[i, j, 0] = torch.tensor(np.random.uniform(
								-np.pi/2, np.pi/2))
            # initializing the scale s
            params[i, j, 1] = torch.tensor(np.random.uniform(
								-np.sqrt(6) / np.sqrt(len(layer1) + len(layer2)),
								+np.sqrt(6) / np.sqrt(len(layer1) + len(layer2))))
    params.requires_grad = True
    return params

# connecting two adjacent layers and initializing all necessary parameters
# this is done with differentiation of the different types of layers
def connectlayers(layer1, layer2):

    if type(layer1[0]) == QuatNeuron and type(layer2[0]) == RealNeuron:
        # parameters between a quaternion and a real layer
        # manually flattening the quaternion output before calling
        # this function is required
        # proceeding as if each quaternion neuron was 3 real neurons
        params = torch.zeros(3 * len(layer1), len(layer2), 2)
        for i in range(3 * len(layer1)):
            for j in range(len(layer2)):
                # initializing the weight w
                params[i, j, 0] = torch.tensor(np.random.uniform(
									-np.sqrt(6) / np.sqrt(3*len(layer1) + len(layer2)),
									+np.sqrt(6) / np.sqrt(3*len(layer1) + len(layer2))))
                # initializing the bias b
                params[i, j, 1] = torch.tensor(0.1)

    elif type(layer1[0]) == RealNeuron and type(layer2[0]) == RealNeuron:
        # parameters between two real layers
        params = torch.zeros(len(layer1), len(layer2), 2)
        for i in range(len(layer1)):
            for j in range(len(layer2)):
                # initializing the weight w
                params[i, j, 0] = torch.tensor(np.random.uniform(
									-np.sqrt(6) / np.sqrt(len(layer1) + len(layer2)),
									+np.sqrt(6) / np.sqrt(len(layer1) + len(layer2))))
                # initializing the bias b
                params[i, j, 1] = torch.tensor(0.1)

    else:
        # parameters between two quaternion layers
        params = torch.zeros(len(layer1), len(layer2), 2)
        for i in range(len(layer1)):
            for j in range(len(layer2)):
                # initializing the angle theta
                params[i, j, 0] = torch.tensor(np.random.uniform(
									-np.pi/2, np.pi/2))
                # initializing the scale s
                params[i, j, 1] = torch.tensor(np.random.uniform(
									-np.sqrt(6) / np.sqrt(len(layer1) + len(layer2)),
									+np.sqrt(6) / np.sqrt(len(layer1) + len(layer2))))
    params.requires_grad = True
    return params


# first layer must be declared seperately to prepare the first parameters
def declarefirstlayer(inputsize, layer):
    params = torch.zeros(inputsize, len(layer), 2)
    for i in range(inputsize):
        for j in range(len(layer)):
            # initializing the angle theta
            params[i, j, 0] = torch.tensor(np.random.uniform(
                                -np.pi / 2, np.pi / 2))
            # initializing the scale s
            params[i, j, 1] = torch.tensor(np.random.uniform(
							    -np.sqrt(6) / np.sqrt(inputsize * len(layer)),
							    +np.sqrt(6) / np.sqrt(inputsize * len(layer))))
    params.requires_grad = True
    return params


# one forward step through a real layer
# not very effective because of nested loops, but but in the runtime
# irrelevant compared to other parts of the network
def forwardlayerreal(input, params):
    output = torch.zeros(len(input[:, 0]), len(params[0, :, 0]))
    # looping over the elements k of the batch
    for k in range(len(input[:, 0])):
        # looping over the neurons of the output layer
        for j in range(len(params[0, :, 0])):
            # looping over the input
            for i in range(len(params[:, 0, 0])):
                # weighing the input with the corresponding weight
                output[k, j] = output[k, j] + (params[i, j, 0]*input[k, i])
            # adding the bias once per neuron
            output[k, j] = output[k, j] + params[0, j, 1]
    return output


# one forward step between two quaternion layers
def forwardlayerquat(input, params):
    # input has the form [k,3,i] with k - element of the batch,
    # i - output of neuron i from the earlier layer, 
    # and respectively the three color values

    # params has the form [i,j,2] and is the parameters for connecting 
    # the earlier layer with i neurons to the later layer with j neurons,
    # assigning theta and scale respectively

    # preparing to have an output of size [k,3,j] with 
    # k - element of the batch, j - output of neuron j of the later layer 
    # (before activation) and respectively the three color values
    output = torch.zeros(len(input[:, 0, 0]), 3, len(params[0, :, 0]))

    # preparation of the rotation for every neuron i from the earlier layer 
    # and output neuron j from the later one
    # using the angle theta from params
    f1 = 1/3 + 2 / 3 * torch.cos(params[:, :, 0])
    f2 = 1/3 - 2 / 3 * torch.cos(params[:, :, 0] - torch.tensor(np.pi) / 3)
    f3 = 1/3 - 2 / 3 * torch.cos(params[:, :, 0] + torch.tensor(np.pi) / 3)
    ff = torch.stack(
         [f1, f2, f3, f3, f1, f2, f2, f3, f1],
         dim=2).unflatten(dim=2, sizes=(3, 3))
    # ff has the size [i,j,3,3] assigning a 3x3 rotation matrix to every 
    # network connection (i,j)

    # stacking the scale from params for each connection (i,j) three times 
    # to facilitate multiplication with the tensor later on
    scal = torch.stack([params[:, :, 1], params[:, :, 1], params[:, :, 1]])
    scal = scal.permute(1, 0, 2)
    # scal has the size [i,j,3]

    # looping over the elements k of the batch
    for k in range(len(input[:, 0, 0])):
        # multiplication of the rotation matrix with the input vector
        # processing all the connections between the layers simultaneously
        rot = torch.matmul(ff[:, :, :, :], input[k, :, :])
        rot = torch.diagonal(rot, offset=0, dim1=0, dim2=3)
        rot = rot.permute(2, 1, 0)

        # scaling the rotated values, summing up the values of each 
        # output neuron j and adding them to the output
        output[k, :, :] = torch.add(output[k, :, :], 
                          torch.sum((scal[:,:,:] * rot[:,:,:]), dim=0))

    # manually deleting big tensors that are no longer needeed
    del ff, scal, rot
    gc.collect()

    return output
