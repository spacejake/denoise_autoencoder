
import torch
from torch.autograd import Variable
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, input_nc, output_nc=512, hidden_nc=32, n_layers=6, stride_layer=2):
        super(CNNEncoder, self).__init__()
        norm_layer = nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, hidden_nc, kernel_size=kw, stride=1, padding=padw),
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
                nn.Conv2d(hidden_nc * nf_mult_prev, hidden_nc * nf_mult,
                          kernel_size=kw, stride=stride, padding=padw),
                norm_layer(hidden_nc * nf_mult),
                nn.ReLU(True) #nn.LeakyReLU(0.2, True)
            ]

        #sequence += [nn.Conv2d(hidden_nc * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw)]
        prepare_linear_seq = [
            # Enforce size of H and W for FC RNN Layer
            nn.AdaptiveAvgPool2d((7,7)),
            Flatten(),
        ]

        fc_sequence = [
            #Add Sequence, SHould we split the image into sequences?????
            #Unsqueeze(1),
            #nn.RNN(input_size=32 * 32 * ngf * mult, hidden_size=256, num_layers=1, bidirectional=False),
            nn.Linear(7 * 7 * hidden_nc * nf_mult, output_nc),
            #nn.BatchNorm1d(output_nc)
        ]

        log_var_seq = [
            nn.Linear(7 * 7 * hidden_nc * nf_mult, output_nc)
        ]

        self.model = nn.Sequential(*sequence)
        self.prep_linear_model = nn.Sequential(*prepare_linear_seq)
        self.fc_model = nn.Sequential(*fc_sequence)
        self.log_var_model = nn.Sequential(*log_var_seq)

    def forward(self, input):
        model_out = self.model(input)
        lin_prep = self.prep_linear_model(model_out)
        mu = self.fc_model(lin_prep)
        log_var = self.log_var_model(lin_prep)
        #logits = self.sigmoid(linear)
        return self.reparameterize(mu, log_var)

    def reparameterize(self, mu, log_var):
        # TODO: return mu when testing
        # Sample epsilon from standard normal distribution
        ''' Another way, not as good??
        std = log_var.mul(0.5).exp_().cuda()
        eps = Variable(std.data.new(std.size()).normal_()).cuda()
        return eps.mul(std).add_(mu)
        '''
        eps = torch.randn(mu.size(0), mu.size(1)).cuda()
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        z = mu + eps * torch.exp(log_var / 2.)
        return z


class CNNDecoder(nn.Module):
    def __init__(self, input_nc=512, hidden_nc=32, output_nc=1, n_layers=6, stride_layer=2):
        super(CNNDecoder, self).__init__()

        nf_mult = min(2 ** ((n_layers - 1) // stride_layer), 8)
        nf_mult_prev = 1

        reshape_sequence = [
            nn.Linear(input_nc, 7*7*(hidden_nc*nf_mult)),
            nn.BatchNorm1d(7*7*(hidden_nc*nf_mult)),
            nn.ReLU(True),
            Expand((hidden_nc*nf_mult), 7, 7)
        ]

        kw = 4
        padw = 1
        sequence = [
            nn.ConvTranspose2d(hidden_nc * nf_mult, hidden_nc * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.ReLU(True) #nn.LeakyReLU(0.2, True)
        ]


        for n in reversed(range(1, n_layers)):
            nf_mult_prev = nf_mult
            nf_mult = min(2**(n//stride_layer), 8)
            stride = 2 if n%stride_layer == 0 else 1
            padw = 1
            if stride == 2:
                padw = 3
            sequence += [
                nn.ConvTranspose2d(hidden_nc * nf_mult_prev, hidden_nc * nf_mult,
                                   kernel_size=kw, stride=stride, padding=padw),
                nn.BatchNorm2d(hidden_nc * nf_mult),
                nn.ReLU(True) #nn.LeakyReLU(0.2, True)
            ]

        sequence += [
            nn.ConvTranspose2d(hidden_nc * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw),
            nn.Sigmoid()
        ]

        self.reshape_model = nn.Sequential(*reshape_sequence)
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        reshaped = self.reshape_model(input)
        return self.model(reshaped)

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
        flattened = input.view(input.size(0), -1)
        return flattened

class Expand(nn.Module):
    def __init__(self, hidden_nc, height, width):
        super(Expand,self).__init__()
        self.hidden_nc = hidden_nc
        self.height = height
        self.width = width

    def forward(self, input):
        expanded = input.view(-1, self.hidden_nc, self.height, self.width)
        return expanded

class Unsqueeze(nn.Module):
    def __init__(self, dim=0):
        super(Unsqueeze, self).__init__()
        self.dim=dim

    def forward(self, input):
        return input.unsqueeze(self.dim)

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

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()

        kw = 3
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            stride =1#stride=2 if n%2 == 0 else 1
            #padw=0 if n%2 == 0 else 1

            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=stride, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        #sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        sequence_out = [nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=1)]
        if use_sigmoid:
            sequence_out += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        #self.model_out = nn.Sequential(*sequence_out)

    def forward(self, input):
        out = self.model(input)
        #out = self.model_out(out)
        return out

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)