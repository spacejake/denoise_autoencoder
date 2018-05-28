
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
from cnn_encoder import CNNEncoder, CNNDecoder, NLayerDiscriminator, GANLoss

from transforms import RandomNoiseWithGT

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batch_size * 784

    return BCE + KLD


def backwardG(fake, gt, w_id=1):
    optimizer_G.zero_grad()
    # GAN Loss
    #loss_G = crit_gan(discriminator(fake), True)
    pred_fake = discriminator(fake)
    true = torch.ones(pred_fake.shape).cuda()
    loss_G = crit_gan(pred_fake, true)

    # ID Loss
    loss_ID = critID(fake, gt)
    loss_G_total = loss_G + loss_ID * w_id
    loss_G_total.backward()
    optimizer_G.step()

    return loss_G_total, loss_G, loss_ID


def backwardD(fake, gt):
    # Train Discriminator
    optimizer_D.zero_grad()
    pred_real = discriminator(gt)
    true = torch.ones(pred_real.shape).cuda()
    loss_D_real = crit_gan(pred_real, true)

    # Fake
    pred_fake = discriminator(fake.detach())
    false = torch.ones(pred_fake.shape).cuda()
    loss_D_fake = crit_gan(pred_fake, false)
    # Combined loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    # backward
    loss_D.backward()  # retain_graph=True)
    optimizer_D.step()
    return loss_D

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

encoder = CNNEncoder(input_nc=1, output_nc=1024)
print(encoder)

decoder = CNNDecoder(input_nc=1024)
print(decoder)

discriminator = NLayerDiscriminator(input_nc=1)#, use_sigmoid=True)
print(discriminator)

encoder.cuda()
decoder.cuda()
discriminator.cuda()

crit_gan = nn.MSELoss() #nn.BCEWithLogitsLoss() #GANLoss()
crit_gan.cuda()
critID = nn.MSELoss() #nn.BCEWithLogitsLoss()
critID.cuda()

# IF TRAIN, else MSELoss
params_G = itertools.chain(encoder.parameters(), decoder.parameters())
params_D = itertools.chain(discriminator.parameters())

optimizer_G = optim.Adam(params_G)#, weight_decay=1e-4)
optimizer_D = optim.Adam(params_D)#, weight_decay=1e-4)

optimizers = []
schedulers = []

optimizers.append(optimizer_G)
optimizers.append(optimizer_D)

# Decay LR by a factor of 0.1 every 5 epochs
for optimizer in optimizers:
    schedulers.append(lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1))

s = 1

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for e in range(100):
    for scheduler in schedulers:
        scheduler.step()

    ep_loss = []
    for i, v in enumerate(dloader):

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

        ##############
        #GAN Backprop#
        ##############
        loss_G_total, loss_G, loss_ID = backwardG(output, gt_imgs)
        ep_loss.append(loss_ID.data.cpu().numpy())

        ########################
        #Discriminator Backprop#
        ########################
        loss_D = backwardD(output, gt_imgs)


        if i % 100 == 0:
            print("Epoch: {0} | Iter: {1} | LR:{2}".format(e, i, schedulers[0].get_lr()[0]))
            #print("Epoch: {0} | Iter: {1}".format(e, i))
            print("MSELoss: {0}, GLoss: {1}, DLoss: {2}".format(loss_ID.data.cpu().numpy(),
                                                                loss_G.detach().cpu().numpy(),
                                                                loss_D.detach().cpu().numpy()))#[0]))
            print("===========================")


        if i % 500 == 0:
            samples = corrupt_imgs.clone().data.cpu()[:1,:,:,:]
            samples = torch.cat((samples, output.data.cpu()[:1,:,:,:]))
            samples = torch.cat((samples, gt_imgs.clone().data.cpu()[:1,:,:,:]))

            torchvision.utils.save_image(samples,
                                         "./out/{0},epoch{1},iter{2}.png".format(s,e,i),
                                         nrow=3)
            s += 1

    avg_loss = sum(ep_loss)/len(ep_loss)
    print("Average epoch loss: ", avg_loss)
    #exp_lr_scheduler.step(avg_loss)


