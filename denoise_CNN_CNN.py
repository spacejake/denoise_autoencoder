
import os
import itertools
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms

import torchvision.utils

from skimage import data, img_as_float, color
from skimage.util import random_noise
from cnn_encoder import CNNEncoder, CNNDecoder

from transforms import RandomNoiseWithGT

output_dir='./out'
batch_size = 32#4
seq_size=16

transform = transforms.Compose([
    RandomNoiseWithGT(),
    transforms.ToTensor()
])

raw_data = dset.MNIST("../autoencoderMNIST/MNIST", download=True, transform=transform)
dloader = torch.utils.data.DataLoader(raw_data, batch_size=batch_size,
                                      shuffle=True, drop_last=True)
in_channel = 1 # Network has same dim for input and output

encoder = CNNEncoder(input_nc=1, )
print(encoder)

decoder = CNNDecoder(input_nc=128)
print(decoder)

encoder.cuda()
decoder.cuda()

crit = nn.MSELoss() #nn.BCEWithLogitsLoss()
crit.cuda()

params = itertools.chain(encoder.parameters(), decoder.parameters())
optimizer = optim.Adam(params, lr=1e-4)#, weight_decay=1e-4)


# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, step_size=3, gamma=0.1)

#exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, threshold=1e-4, mode='min',
#                                             factor=0.1, min_lr=1e-6,verbose=True)
s = 1

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for e in range(100):
    exp_lr_scheduler.step()
    ep_loss = []
    for i, v in enumerate(dloader):
        optimizer.zero_grad()

        corrupt_imgs = (v[0])[:,:,:,:28].cuda()
        gt_imgs = (v[0])[:,:,:,28:].cuda()

        ########
        #Encoder
        ########
        encoder_out = encoder(corrupt_imgs)

        logits = decoder(encoder_out)#, encoder_state)

        #######
        #loss##
        #######
        output = logits
        #output = torch.sigmoid(logits)

        #output = torch.sigmoid(decoder_out)
        #output = threshold(decoder_out)

        loss = crit(output, gt_imgs)
        ep_loss.append(loss.data.cpu().numpy())

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Epoch: {0} | Iter: {1} | LR:{2}".format(e, i, exp_lr_scheduler.get_lr()[0]))
            #print("Epoch: {0} | Iter: {1}".format(e, i))
            print("Loss: {0}".format(loss.data.cpu().numpy()))#[0]))
            print("===========================")


        if i % 500 == 0:
            samples = corrupt_imgs.clone().data.cpu()[:1,:,:,:]
            samples = torch.cat((samples, output.data.cpu()[:1,:,:,:]))
            samples = torch.cat((samples, gt_imgs.clone().data.cpu()[:1,:,:,:]))

            torchvision.utils.save_image(samples,
                                "./out/{0},epoch{1},iter{2}.png".format(s,
                                                                         e,i),
                                         nrow=3)
            s += 1
    avg_loss = sum(ep_loss)/len(ep_loss)
    print("Average epoch loss: ", avg_loss)
    #exp_lr_scheduler.step(avg_loss)
