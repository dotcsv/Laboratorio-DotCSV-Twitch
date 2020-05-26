import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torch import nn
from torch.nn import functional as F

POOL_SIZE = 1024
N_CHANNEL = 16
width = height = int(math.sqrt(POOL_SIZE))

sobelX = torch.from_numpy(np.array([[-1, 0,  1], [-2, 0, 2], [-1, 0, 1]]).astype(float)).repeat((16,16,1,1))
sobelY = torch.from_numpy(np.array([[-1,-2, -1], [0,  0, 0], [1,  2, 1]]).astype(float)).repeat((16,16,1,1))
cellId = torch.from_numpy(np.array([[0,  0,  0], [0,  1, 0], [0, 0,  0]]).astype(float)).repeat((16,16,1,1))


filters = [sobelX, sobelY, cellId]

grid = np.zeros((width, height, N_CHANNEL))


def imshowTensor(x, chan_n=3):
    plt.imshow(x[0,:chan_n,...].detach().numpy().transpose(1,2,0))
    plt.show()
    return

class UpdateGrid(torch.nn.Module):

    def __init__(self):
        super(UpdateGrid, self).__init__()
        self.fc1 = nn.Conv2d(N_CHANNEL * len(filters), 128, (1, 1))
        self.fc2 = nn.Conv2d(128, N_CHANNEL, (1, 1))

    def forward(self, x):

        perception = torch.empty((1, len(filters) * N_CHANNEL, width, height))

        for f, filt in enumerate(filters):

            perception[:, (f * N_CHANNEL):((f+1) * N_CHANNEL), :, :] = F.conv2d(x, filt, padding=[1, 1])

        dx = self.fc1(perception)
        dx = F.relu(dx)
        dx = self.fc2(dx)

        # Skip connection + stochastic update.
        randomMask = torch.from_numpy(np.random.randint(0, 2, (1, 1, width, height))).repeat(1,16,1,1)

        x = x + dx * randomMask

        alive = F.conv2d((x[:,3:4,:,:] > 0.1).type(torch.int),
                         torch.from_numpy(np.ones((1,1,3,3)).astype(int)), padding=1)

        alive = (alive > 0.0).type(torch.int)

        alive = alive.repeat(1,16,1,1)

        return x * alive



for step in range(1000):
    if step == 0:
        im = Image.open("macedonIA.png")
        grid[:, :, 0:4] = np.array(im) / 255.
        updateGrid = UpdateGrid()
        result = torch.from_numpy(grid).view((1, width, height, N_CHANNEL)).permute(0, 3, 1, 2)
    else:
        result = updateGrid.forward(result)

    imRes = result[0, :4, :, :].detach().numpy().transpose(1, 2, 0)
    plt.imshow((np.clip(imRes[:, :, :3], 0.0, 1.0) * 255.0).astype(int))
    plt.show()

# im = np.clip((arr * 255).astype(int), 0, 255)
# Image.fromarray(im, mode='RGBA').show()