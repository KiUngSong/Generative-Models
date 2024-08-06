import torch, torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from tqdm.notebook import tqdm
from einops import rearrange
import matplotlib


# Self-Attention block
class Self_Att(nn.Module):
    def __init__(self, in_dim):
        super(Self_Att,self).__init__()        
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size= 1)
        # Define trainable parameter gamma which controls effect of attention
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        # inputs : feature maps with [B, C, W, H] shape
        h = x.size(-1)
        query  = self.query_conv(x)
        query = rearrange(query, 'b c w h -> b (w h) c')
        key =  self.key_conv(x)
        key = rearrange(key, 'b c w h -> b c (w h)')
        # q = k = w * h dimensional
        att = torch.einsum('bqc, bck -> bqk', query, key)
        att = F.softmax(att, dim=-1)
        att = rearrange(att, 'b q k -> b k q')

        value = self.value_conv(x)
        value = rearrange(value, 'b c w h -> b c (w h)')
        # Tensor shape of final output is [B, C, W * H]
        out = torch.einsum('bck, bkq -> bcq', value, att)
        # Reshape tensor to the desired input shape : [B, C, W, H]
        out = rearrange(out, 'b c (w h) -> b c w h', h=h)
        out = self.gamma * out + x

        return out, att


# Define Generator class
class Generator(nn.Module):
    def __init__(self, img_shape, z_dim=128, conv_dim=32):
        assert 2 ** int(np.log2(img_shape[-1])) == img_shape[-1]
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        def block(in_feat, out_feat):
            layers = [spectral_norm(nn.ConvTranspose2d(in_feat, out_feat, kernel_size=3, 
                                                        stride=2, padding=1, output_padding=1))]
            layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.ReLU())
            return layers
        
        repeat_num = int(np.log2(img_shape[-1])) - 2
        dim = conv_dim * (2 ** repeat_num)

        self._blocks = []
        self._blocks += block(z_dim, dim)
        for i in range(repeat_num-2):
            self._blocks += block(dim, dim // 2)
            dim = dim // 2
        self._blocks = np.reshape(self._blocks, -1).tolist()
        self.block1 = nn.Sequential(*self._blocks)
        self.attn1 = Self_Att(dim)

        self._blocks = []
        self._blocks += block(dim, dim // 2)
        self._blocks = np.reshape(self._blocks, -1).tolist()
        self.block2 = nn.Sequential(*self._blocks)
        dim = dim // 2
        self.attn2 = Self_Att(dim)

        self._blocks = [nn.ConvTranspose2d(dim, img_shape[0], 3, 2, 1, 1), nn.Tanh()]
        self.block3 = nn.Sequential(*self._blocks)

    def forward(self, z):
        # Input shape of noise tensor as [B, C, 2, 2]
        img = self.block1(z)
        img, p1 = self.attn1(img)
        img = self.block2(img)
        img, p2 = self.attn2(img)
        img = self.block3(img)

        return img, p1, p2


# Define Discriminator class
class Discriminator(nn.Module):
    def __init__(self, img_shape, dim=32):
        super(Discriminator, self).__init__()

        def block(in_feat, out_feat):
            layers = [spectral_norm(nn.Conv2d(in_feat, out_feat, 4, 2, 1)), nn.LeakyReLU(0.1)]
            return layers

        repeat_num = int(np.log2(img_shape[-1])) - 2

        self._blocks = []
        self._blocks += block(img_shape[0], dim)
        for i in range(repeat_num-2):
            self._blocks += block(dim, 2 * dim)
            dim =  2 * dim
        self._blocks = np.reshape(self._blocks, -1).tolist()
        self.block1 = nn.Sequential(*self._blocks)
        self.attn1 = Self_Att(dim)

        self._blocks = []
        self._blocks += block(dim, 2 * dim)
        self._blocks = np.reshape(self._blocks, -1).tolist()
        self.block2 = nn.Sequential(*self._blocks)
        dim = 2 * dim
        self.attn2 = Self_Att(dim)

        self.block3 = nn.Conv2d(dim, 1, 4)

    def forward(self, x):
        x = self.block1(x)
        x, p1 = self.attn1(x)
        x = self.block2(x)
        x, p2 = self.attn2(x)
        x = self.block3(x)
        # Apply global avg pooling and flatten to match batch size
        # Due to gloabl avg pooling agnostic to image size
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1), p1, p2


def Train(epoch, dataloader, device, z_dim, G, D, optimizer_G, optimizer_D, verbose=5):
    fixed_z = torch.randn(64, z_dim, 2, 2).to(device)
    for j in tqdm(range(epoch)):
        G.train()
        for _, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)

            # Sample noise as generator input
            z = torch.randn(batch_size, z_dim, 2, 2).to(device)
            # Generate a batch of images
            gen_imgs, _, _ = G(z)

            ## Train Discriminator with WGAN-GP loss
            pred_fake, _, _ = D(gen_imgs)
            fake_loss = torch.mean(pred_fake)
            pred_real, _, _ = D(imgs)
            real_loss = - torch.mean(pred_real)

            # Compute GP loss
            alpha = torch.rand(batch_size, 1, 1, 1).expand_as(imgs).to(device)
            imgs_hat = alpha * imgs + (1 - alpha) * gen_imgs
            pred_hat, _, _ = D(imgs_hat)

            grad = torch.autograd.grad(outputs=pred_hat, inputs=imgs_hat,
                                           grad_outputs=torch.ones(pred_hat.size()).to(device),
                                           retain_graph=True, create_graph=True)[0]
            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            gp_loss = torch.mean((grad_l2norm - 1) ** 2)

            # Update discriminator
            optimizer_D.zero_grad()
            d_loss = real_loss + fake_loss + 10 * gp_loss
            d_loss.backward()
            optimizer_D.step()

            ## Train Generator
            z = torch.randn(batch_size, z_dim, 2, 2).to(device)
            pred_fake, _, _ = D(G(z)[0])
            optimizer_G.zero_grad()
            g_loss = - torch.mean(pred_fake)
            g_loss.backward()
            optimizer_G.step()

        if (j+1) % verbose == 0:
            print(f"Epoch {j+1} / D loss: {d_loss.item():.4f} / G loss: {g_loss.item():.4f}")

            G.eval()
            gen_imgs, _, _ = G(fixed_z)
            gen_imgs = np.transpose(torchvision.utils.make_grid(
                                    gen_imgs.detach().cpu(), padding=2, normalize=True),(1,2,0))
            matplotlib.image.imsave('Generated_Images.jpg', gen_imgs.numpy())
