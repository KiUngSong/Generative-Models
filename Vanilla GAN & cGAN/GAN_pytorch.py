import torch
import numpy as np
from torch import nn
from tqdm.notebook import tqdm


class Generator(nn.Module):
    def __init__(self, img_shape, dim_latent, g_dims=[128,256,512,1024]):
        super(Generator, self).__init__()
        self.dim_latent = int(dim_latent)
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self._blocks = []
        self._blocks += block(self.dim_latent, g_dims[0], normalize=False)
        for i in range(len(g_dims)-1):
            self._blocks += block(g_dims[i], g_dims[i+1])
        self._blocks = np.reshape(self._blocks, -1).tolist()
        self.total_block = nn.Sequential(*self._blocks)

        self.fc = nn.Sequential(
            nn.Linear(g_dims[-1], int(np.prod(img_shape))),
            nn.Tanh())

    def forward(self, x):
        x = self.total_block(x)
        img = self.fc(x)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape, d_dims=[512, 256]):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self._blocks = []
        self._blocks += block(int(np.prod(self.img_shape)), d_dims[0])
        for i in range(len(d_dims)-1):
            self._blocks += block(d_dims[i], d_dims[i+1])
        self.total_block = nn.Sequential(*self._blocks)

        self.fc = nn.Sequential(nn.Linear(d_dims[-1], 1), nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.total_block(x)
        pred = self.fc(x)
        return pred


def Train(epoch, dataloader, device, G, D, optimizer_G, optimizer_D):
    Tensor = torch.FloatTensor
    dim_latent = G.dim_latent
    adversarial_loss = torch.nn.BCELoss()

    for j in tqdm(range(epoch)):
        for _, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)
            
            # Adversarial ground truths
            y_valid = torch.ones(batch_size, 1).to(device)
            y_fake = torch.zeros(batch_size, 1).to(device)
            
            # Configure input
            real_imgs = imgs.type(Tensor).to(device)
            
            # Sample noise as generator input
            z = torch.rand(batch_size, dim_latent).to(device)

            # Generate a batch of images
            gen_imgs = G(z)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(D(real_imgs), y_valid)
            fake_loss = adversarial_loss(D(gen_imgs.detach()), y_fake)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator : 
            optimizer_G.zero_grad()
            g_loss = adversarial_loss(D(gen_imgs), y_valid)
            g_loss.backward()
            optimizer_G.step()

        if (j+1)%5 == 0:
            print(f"Epoch {j+1} / D loss: {d_loss.item():.4f} / G loss: {g_loss.item():.4f}")
        