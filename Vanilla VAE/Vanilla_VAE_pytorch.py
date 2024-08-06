import torch, torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F

from tqdm.notebook import tqdm
import matplotlib


class VAE(nn.Module):
    def __init__(self, img_size, dims=[400,200], dim_latent=20):
        super(VAE, self).__init__()
        self.img_size = int(np.prod(img_size))

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU())
            return layers

        encoder_block = []
        encoder_block += block(self.img_size, dims[0])
        for i in range(len(dims)-1):
            encoder_block += block(dims[i], dims[i+1])
        encoder_block += [nn.Linear(dims[-1], 2 * dim_latent)]
        encoder_block = np.reshape(encoder_block, -1).tolist()
        self.encoder_block = nn.Sequential(*encoder_block)

        dims.reverse()
        decoder_block = []
        decoder_block += block(dim_latent, dims[0])
        for i in range(len(dims)-1):
            decoder_block += block(dims[i], dims[i+1])
        decoder_block += [nn.Linear(dims[-1], self.img_size)]
        decoder_block = np.reshape(decoder_block, -1).tolist()
        self.decoder_block = nn.Sequential(*decoder_block)

    def encoder(self, x):
        x = self.encoder_block(x.view(-1, self.img_size))
        mu, logvar = torch.chunk(x, chunks=2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decoder(self, z):
        z = self.decoder_block(z)
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def Train(epoch, dataloader, device, VAE, img_size, optimizer, verbose):
    fixed_imgs, _ = next(iter(dataloader))
    fixed_imgs = fixed_imgs.to(device)
    temp = fixed_imgs
    temp = np.transpose(torchvision.utils.make_grid(temp.cpu(), padding=2, normalize=True),(1,2,0))
    matplotlib.image.imsave('Input_Images.jpg', temp.numpy())
    
    for j in tqdm(range(epoch)):
        VAE.train()
        for _, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            imgs = imgs.to(device).view(-1, int(np.prod(img_size)))
            # Compute Loss
            recon_imgs, mu, logvar = VAE(imgs)
            # Since reconstruction loss is computed for flattened image, 
            # both reduction with sum & dividend by batch size is desired to consider
            # proper loss scale per each image
            Recon_loss = F.binary_cross_entropy(recon_imgs, imgs, reduction='sum') / batch_size
            KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            loss = Recon_loss + KL_loss
            # Update VAE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (j+1) % verbose == 0:
            print(f"Epoch {j+1} / Loss: {loss.item():.4f}")
            # Evaluate result of VAE
            VAE.eval()
            recon_imgs, _, _ = VAE(fixed_imgs)
            recon_imgs = recon_imgs.view(fixed_imgs.size(0), *img_size)
            recon_imgs = np.transpose(torchvision.utils.make_grid(
                                        recon_imgs.detach().cpu(), padding=2, normalize=True),(1,2,0))
            matplotlib.image.imsave('Reconstructed_Images.jpg', recon_imgs.numpy())
