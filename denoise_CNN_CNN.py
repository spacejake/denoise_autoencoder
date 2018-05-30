
import os
import itertools
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

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
from pygit2 import Repository

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Denoise Autoencoder.")
    parser.add_argument("--train", action="store_true", dest="isTrain", default=True,
                        help="Train model, save checkpoints (Default)")
    parser.add_argument("--test", action="store_false", dest="isTrain", default=True,
                        help="Test model, load checkpoints")
    parser.add_argument("--noise", dest="noise_var", default=0.8, type=float,
                        help="Random Noise Variance, Default = 0.8")
    return parser.parse_args()

args = get_arguments()
branch = "default"

try:
    branch = Repository('.').head.shorthand
except:
    pass

isTrain = args.isTrain
if isTrain:
    print("Trainging...")
else:
    print("Testing...")
save_dir = "./{}/chkpnts".format(branch)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# helper saving function that can be used by subclasses
def save_network(network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()

# helper loading function that can be used by subclasses
def load_network(network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    if os.path.isfile(save_path):
        print("Loading model: {}".format(save_path))
        network.load_state_dict(torch.load(save_path))
    else:
        print("Cannot Find Model: {}".format(save_path))
        exit(1)

output_dir='./out'
batch_size = 32#4
seq_size=16

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28**2)).cuda()

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * 28**2

    return BCE + KLD

mode = "train" if isTrain else "test"
output_dir='./{0}/out_{1}'.format(branch,mode)
batch_size = 32#4
seq_size=16


transform = transforms.Compose([
    RandomNoiseWithGT(args.noise_var),
    transforms.ToTensor(),
])

raw_data = dset.MNIST("../autoencoderMNIST/MNIST", train=True, download=True, transform=transform)
raw_test = dset.MNIST("../autoencoderMNIST/MNIST", train=False, download=True, transform=transform)
dloader_train = torch.utils.data.DataLoader(raw_data, batch_size=batch_size,
                                      shuffle=True, drop_last=True)

dloader_test = torch.utils.data.DataLoader(raw_test, batch_size=batch_size,
                                      shuffle=True, drop_last=True)

dloader = dloader_train if isTrain == True else dloader_test

in_channel = 1 # Network has same dim for input and output

encoder = CNNEncoder(input_nc=1, output_nc=784)
print(encoder)

decoder = CNNDecoder(input_nc=784)
print(decoder)

if isTrain is False:
    which_epoch = 'latest'
    load_network(encoder, 'Encoder', which_epoch)
    load_network(decoder, 'Decoder', which_epoch)

encoder.cuda()
decoder.cuda()

crit = loss_function #nn.MSELoss() #nn.BCEWithLogitsLoss()
#crit.cuda()

if isTrain is True:
    params = itertools.chain(encoder.parameters(), decoder.parameters())
    optimizer = optim.Adam(params)#, lr=1e-4)#, weight_decay=1e-4)


    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    #exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, step_size=3, gamma=0.1)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

s = 0

for e in range(100):
    if isTrain:
        exp_lr_scheduler.step()
    ep_loss = []
    for i, v in enumerate(dloader):
        if isTrain:
            optimizer.zero_grad()

        corrupt_imgs = (v[0])[:,:,:,:28].cuda()
        gt_imgs = (v[0])[:,:,:,28:].cuda()

        ########
        #Encoder
        ########
        encoder_out, mu, logvar = encoder(corrupt_imgs)

        logits = decoder(encoder_out)#, encoder_state)

        #######
        #loss##
        #######
        output = logits
        #output = torch.sigmoid(logits)

        #output = torch.sigmoid(decoder_out)
        #output = threshold(decoder_out)

        loss = crit(output, gt_imgs, mu, logvar)

        mseloss = F.mse_loss(output, gt_imgs)
        ep_loss.append(mseloss.data.cpu().numpy())

        if isTrain:
            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            if isTrain:
                print("Epoch: {0} | Iter: {1} | LR:{2}".format(e, i, exp_lr_scheduler.get_lr()[0]))
            #print("Epoch: {0} | Iter: {1}".format(e, i))

            print("Epoch: {0} | Iter: {1} | Loss: {2}".format(e, i, ep_loss[-1]))#[0]))
            print("===========================")


        if (isTrain is False and i%25 == 0) or (isTrain is True and s < 50 and i%10 == 0) or i % 500 == 0:
            samples = corrupt_imgs.clone().data.cpu()[:10,:,:,:]
            samples = torch.cat((samples, output.data.cpu()[:10,:,:,:]))
            samples = torch.cat((samples, gt_imgs.clone().data.cpu()[:10,:,:,:]))
            fname = "{1},epoch{2},iter{3}.png".format(output_dir, s, e, i)
            fname = os.path.join(output_dir, fname)

            torchvision.utils.save_image(samples,
                                         fname,
                                         nrow=10)
            s += 1
    if isTrain:
        save_network(encoder, 'Encoder', 'latest')
        save_network(decoder, 'Decoder', 'latest')
        if e%100 == 0:
            save_network(encoder, 'Encoder', e)
            save_network(decoder, 'Decoder', e)

    avg_loss = sum(ep_loss)/len(ep_loss)
    print("Average epoch loss: ", avg_loss)
    #exp_lr_scheduler.step(avg_loss)
    if isTrain is False:
        break

if isTrain:
    save_network(encoder, 'Encoder', 'latest')
    save_network(decoder, 'Decoder', 'latest')