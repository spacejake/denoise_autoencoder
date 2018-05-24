
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, input_nc, output_nc=128, hidden_in_nc=32, n_layers=6, stride_layer=2):
        super(CNNEncoder, self).__init__()
        norm_layer = nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, hidden_in_nc, kernel_size=kw, stride=1, padding=padw),
            nn.ReLU(True) #nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**(n//stride_layer), 8)
            stride = 2 if n%stride_layer == 0 else 1
            padw=1
            if stride == 2:
                padw=3

            sequence += [
                nn.Conv2d(hidden_in_nc * nf_mult_prev, hidden_in_nc * nf_mult,
                          kernel_size=kw, stride=stride, padding=padw),
                norm_layer(hidden_in_nc * nf_mult),
                nn.ReLU(True) #nn.LeakyReLU(0.2, True)
            ]

        sequence += [nn.Conv2d(hidden_in_nc * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class CNNDecoder(nn.Module):
    def __init__(self, input_nc=128, hidden_out_nc=32, output_nc=1, n_layers=6, stride_layer=2):
        super(CNNDecoder, self).__init__()
        norm_layer = nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.ConvTranspose2d(input_nc, input_nc, kernel_size=kw, stride=1, padding=padw),
            nn.ReLU(True) #nn.LeakyReLU(0.2, True)
        ]

        nf_mult = min(2**((n_layers-1)//stride_layer), 8)
        nf_mult_prev = 1
        for n in reversed(range(1, n_layers)):
            nf_mult_prev = nf_mult
            nf_mult = min(2**(n//stride_layer), 8)
            stride = 2 if n%stride_layer == 0 else 1
            padw = 1
            if stride == 2:
                padw = 3
            sequence += [
                nn.ConvTranspose2d(hidden_out_nc * nf_mult_prev, hidden_out_nc * nf_mult,
                          kernel_size=kw, stride=stride, padding=padw),
                norm_layer(hidden_out_nc * nf_mult),
                nn.ReLU(True) #nn.LeakyReLU(0.2, True)
            ]

        sequence += [nn.ConvTranspose2d(hidden_out_nc * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class DenoiesCNN(nn.Module):
    def __init__(self, in_channel, hidden_channels, use_fc=False, fc_out_channels=None):
        super(DenoiesCNN, self).__init__()
        #layers = []
        #self.layers = nn.ModuleList()
        self.layers = nn.Sequential()
        self.kernel_size =  3
        self.out_channels =  fc_out_channels if fc_out_channels is not None else hidden_channels[-1]
        self.dense_lyr = None

        for idx, hid_chnl in enumerate(hidden_channels):

            conv_lyr = nn.Sequential(              # input shape (in_channel, ?, ?)
                nn.Conv2d(
                    in_channels=in_channel,        # input height
                    out_channels=hid_chnl,         # n_filters
                    kernel_size=self.kernel_size,  # filter size
                    stride=1,                      # filter movement/step
                    padding=self.kernel_size//2    # for same width/height of image after con2d, padding=(kernel_size-1)/2 if stride=1
                ),                                 # output shape (hid_chnl, ?, ?)
                nn.BatchNorm2d(hid_chnl,affine=True),
            )

            if idx < (len(hidden_channels)-1):# or use_fc == True:
                conv_lyr.add_module("activation", nn.ReLU())

            self.layers.add_module("conv_{}".format(idx), conv_lyr)

            # setup input size for next layer
            in_channel = hid_chnl

        #if use_fc == True:
        #    self.output_lyr = nn.Sequential(
        #        Flatten(),
        #        nn.Linear( 28 * 28 * hidden_channels[-1], self.out_channels), # fully connected layer, output 10 classes
        #        nn.BatchNorm1d(self.out_channels)
        #    )
        #    self.layers.add_module("dense_1", dense_lyr)

    def forward(self, input):
        return self.layers(input)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization