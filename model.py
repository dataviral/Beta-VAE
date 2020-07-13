import torch as th
from torch import nn 


class BetaVAE(nn.Module):
    
    
    def __init__(self, input_size=64, latent_dim=64, beta=1, input_channels=3):

        super(BetaVAE, self).__init__()

        self.input_size         =   input_size
        self.latent_dim         =   latent_dim
        self.beta               =   beta
        self.input_channels     =   input_channels
        self.kernel_size        =   (2, 2)

        self.__validate()

        self.encoder            =   self.__init_encoder()
        self.decoder            =   self.__init_decoder()
        self.mean, self.logvar     =   self.__init_sampler()


    def __validate(self):
        
        # input_size validation
        input_size = self.input_size
        while input_size > 0: 
            if (input_size % 2) != 0:
                raise ValueError("Input Size must be a power of 2")


    def __init_encoder(self):
        
        modules = [
            nn.Conv2d(self.input_channels, 16, self.kernel_size, stride=2, padding=1),
            nn.Conv2d(16, 32, self.kernel_size, stride=2, padding=1),
            nn.Conv2d(32, 32, self.kernel_size, stride=2, padding=1),
            nn.Conv2d(32, 64, self.kernel_size, stride=2, padding=1),
            nn.Conv2d(64, 64, self.kernel_size)
        ]

        return nn.Sequential(*modules)
    

    def __init_decoder(self):

        modules = [
            nn.ConvTranspose2d(64, 64, self.kernel_size),
            nn.ConvTranspose2d(64, 32, self.kernel_size, stride=2, padding=1),
            nn.ConvTranspose2d(32, 32, self.kernel_size, stride=2, padding=1),
            nn.ConvTranspose2d(32, 16, self.kernel_size, stride=2, padding=1),
            nn.ConvTranspose2d(16, self.input_channels, self.kernel_size, stride=2, padding=1)
        ]

        return nn.Sequential(*modules)
    

    def __init_sampler(self):

        mean = nn.Linear(64, self.latent_dim)
        logvar  = nn.Linear(64, self.latent_dim)

        return mean, logvar


    def __reparameterize(self, mean, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.FloatTensor(std.shape).normal_()
        
        return mean + std * eps
