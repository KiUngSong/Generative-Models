import torch, torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from math import pi
from einops import rearrange
import numpy as np
from tqdm import tqdm


class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.register_buffer("init", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, x):
        with torch.no_grad():
            mean = torch.mean(x.clone(), dim=[0,2,3], keepdim=True)
            std = torch.std(x.clone(), dim=[0,2,3], keepdim=True)
            self.scale.data.copy_(1 / (std + 1e-6))
            self.bias.data.copy_(-mean)

    def forward(self, x):
        if self.init.item() == 0:
            self.initialize(x)
            self.init.fill_(1)
        
        _, _, h, w = x.shape
        # Log determinant is h*w*sum(log|s|)
        logdet = h * w * torch.sum(torch.log(torch.abs(self.scale)))
        return self.scale * (x + self.bias), logdet

    def reverse(self, x):
        return x / self.scale - self.bias


# Invertible 1x1 convolution with LU decomposition for faster training
class Inv1x1(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        w_shape = [in_channel, in_channel]
        # Process of sampling random rotation matrix
        w_init, _ = torch.qr(torch.randn(*w_shape))
        P, L, U = torch.lu_unpack(*torch.lu(w_init))
        s = torch.diag(U)

        self.register_buffer("L_mask", torch.tril(torch.ones(w_shape), -1))
        self.register_buffer("eye", torch.eye(*w_shape))
        self.register_buffer("P", P)
        self.register_buffer("s_sign", torch.sign(s))

        self.log_s = nn.Parameter(torch.log(torch.abs(s)))
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)

    def get_weight(self):
        weight = (self.P @ (self.L * self.L_mask + self.eye) 
                    @ (self.U * self.L_mask.t() + torch.diag(self.s_sign * torch.exp(self.log_s))))
        return weight

    def forward(self, x):
        _, _, h, w = x.shape
        weight = self.get_weight()
        weight = rearrange(weight, "w1 w2 -> w1 w2 1 1")
        # Log determinant is h*w*sum(log|s|)
        logdet = h * w * torch.sum(self.log_s)
        return F.conv2d(x, weight), logdet

    def reverse(self, x):
        weight = self.get_weight()
        weight_inv = rearrange(weight.inverse(), "w1 w2 -> w1 w2 1 1")
        return F.conv2d(x, weight_inv)


class ZeroConv(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, 0)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.pad(x, [1, 1, 1, 1], value=1)
        return self.conv(x) * torch.exp(self.scale * self.scale_factor)


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, latent_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, latent_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, 1), nn.ReLU(inplace=True), ZeroConv(latent_dim, in_channel))

    def forward(self, x):
        x_a, x_b = x.chunk(2, 1)
        log_s, bias = self.net(x_b).chunk(2, 1)
        # Followed the openai's implementation which used sigmoid instead of exp(log_s)
        scale = torch.sigmoid(log_s + 2)
        # Log determinant is sum(log|s|)
        logdet = torch.log(scale).view(x.shape[0], -1).sum(-1)
        return torch.cat([scale * (x_a + bias), x_b], 1), logdet

    def reverse(self, output):
        y_a, y_b = output.chunk(2, 1)
        log_s, bias = self.net(y_b).chunk(2, 1)
        scale = torch.sigmoid(log_s + 2)
        return torch.cat([y_a / scale - bias, y_b], 1)


class Flow_Step(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.actnorm = ActNorm(in_channel)
        self.inv1x1 = Inv1x1(in_channel)
        self.coupling = AffineCoupling(in_channel)

    # Forward process transforms x to z
    def forward(self, x):
        x, logdet0 = self.actnorm(x)
        x, logdet1 = self.inv1x1(x)
        z, logdet2 = self.coupling(x)
        return z, logdet0 + logdet1 + logdet2

    # Reverse process transforms z to x
    def reverse(self, z):
        x = self.coupling.reverse(z)
        x = self.inv1x1.reverse(x)
        x = self.actnorm.reverse(x)
        return x


# Latent vector z is computed for each flow block
class Flow_Block(nn.Module):
    """ For each flow block, input or output tensor shape is transformed
        Forward process of x to z : for input, (b, c, h, w) -> (b, 4*c, h/2, w/2)
        Reverse process of z to x : for output, (b, c, h, w) -> (b, c/4, 2h, 2w)
        split: If 'True', latent z is computed by channel-wise splittng """
    def __init__(self, in_channel, n_flow, split=True):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow_Step(in_channel * 4))

        self.split = split
        if split:
            self.prior = ZeroConv(in_channel * 2, in_channel * 4)
        else:
            self.prior = ZeroConv(in_channel * 4, in_channel * 8)

    # Compute the log likelihood of latent z for forward process
    def gaussian_log_p(self, x, mean, log_std):
        return -0.5 * np.log(2 * pi) - log_std - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_std)

    # Forward process transforms x to z
    def forward(self, x):
        x = rearrange(x, "b c (h s1) (w s2) -> b (c s1 s2) h w", s1=2, s2=2)

        logdet_sum = 0
        for flow in self.flows:
            x, logdet = flow(x)
            logdet_sum += logdet

        if self.split:
            x, z = x.chunk(2, 1)
            mean, log_std = self.prior(x).chunk(2, 1)
            log_p = self.gaussian_log_p(z, mean, log_std).view(x.shape[0], -1).sum(1)
        else:
            z = x
            mean, log_std = self.prior(torch.zeros_like(x)).chunk(2, 1)
            log_p = self.gaussian_log_p(x, mean, log_std).view(x.shape[0], -1).sum(1)

        return x, logdet_sum, log_p, z

    # Sample the latent z from the trained gaussian distribution for reverse process
    def gaussian_sample(self, mean, log_std, temperature):
        return mean + torch.exp(log_std) * temperature

    # Reverse process transforms z to x
    def reverse(self, output, temperature):
        if self.split:
            mean, log_std = self.prior(output).chunk(2, 1)
            z = self.gaussian_sample(mean, log_std, temperature)
            x = torch.cat([output, z], 1)
        else:
            mean, log_std = self.prior(torch.zeros_like(output)).chunk(2, 1)
            z = self.gaussian_sample(mean, log_std, temperature)
            x = z

        for flow in self.flows[::-1]:
            x = flow.reverse(x)

        x = rearrange(x, "b (c s1 s2) h w -> b c (h s1) (w s2)", s1=2, s2=2)
        return x


class Glow_model(nn.Module):
    def __init__(self, in_channel, n_flow, n_block):
        super().__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        # "split=True" is used for intermedita flow blocks
        for i in range(n_block - 1):
            self.blocks.append(Flow_Block(n_channel, n_flow))
            n_channel *= 2
        # For the last forward block, z is obtained without splitting
        self.blocks.append(Flow_Block(n_channel, n_flow, split=False))

    def forward(self, x):
        log_p_sum = 0
        logdet_sum = 0

        for block in self.blocks:
            x, logdet, log_p, _ = block(x)
            log_p_sum += log_p
            logdet_sum += logdet

        return log_p_sum, logdet_sum

    def reverse(self, z_list):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                x = block.reverse(z_list[-1], z_list[-1])
            else:
                x = block.reverse(x, z_list[-(i + 1)])
        return x


class Glow():
    def __init__(self, img_size, in_channel, n_flow, n_block, save_path, device, is_load):
        super(Glow, self).__init__()
        self.model = Glow_model(in_channel, n_flow, n_block)
        self.model = self.model.to(device)
        print(f"Number of model parameters : {sum(p.numel() for p in self.model.parameters())}")
        
        self.device = device
        self.save_path = save_path
        self.sample_temperature = 0.7

        # Define shape of latent z for each block
        self.z_shapes = []
        for i in range(n_block - 1):
            img_size //= 2
            in_channel *= 2
            self.z_shapes.append((in_channel, img_size, img_size))
        img_size //= 2
        self.z_shapes.append((in_channel * 4, img_size, img_size))

        if is_load:
            self.model.load_state_dict(torch.load(self.save_path))
            print("Model loaded successfully")

    def dequantize(self, data, n_bits=5):
        n_bins = 2. ** n_bits
        data = data * 255
        if n_bits < 8:
            data = torch.floor(data / 2 ** (8 - n_bits))
        return data / n_bins - .5, n_bins   

    def sample_z(self, num_samples=16):
        z_sample = []
        for z in self.z_shapes:
            z_new = torch.randn(num_samples, *z) * self.sample_temperature
            z_sample.append(z_new.to(self.device))
        return z_sample

    def train(self, dataloader, epoch, lr, verbose=10):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.num_samples = 16
        self.fixed_sample_z = self.sample_z(num_samples=self.num_samples)

        for i in tqdm(range(epoch)):
            train_loss = 0
            for j, imgs in enumerate(dataloader):
                imgs = imgs[0].to(self.device)
                b, c, h, w = imgs.shape
                n_pixel = c * h * w
                imgs, n_bins = self.dequantize(imgs, n_bits=5)

                log_p, logdet = self.model(imgs + torch.randn_like(imgs) / n_bins)
                loss = log_p + logdet - (np.log(n_bins) * n_pixel)
                loss = - (loss / (np.log(2) * n_pixel)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b
                denom = np.log(2) * n_pixel

                if j % 200 == 0:
                    self.check(f'Samples.jpg', loss, log_p/denom, logdet/denom, temp=True)

            if (i+1) % verbose == 0:
                self.check(f'Generated_Images_{i+1}.jpg', train_loss, log_p/denom, logdet/denom, idx=i)
                torch.save(self.model.state_dict(), self.save_path)

    # Save example of test images to check training
    def check(self, img_path, loss_value, log_p, logdet, idx=None, temp=False):
        self.model.eval()
        with torch.no_grad():
            gen_imgs = self.model.reverse(self.fixed_sample_z)
        self.model.train()
        
        torchvision.utils.save_image(gen_imgs.detach().cpu(), img_path, 
                                    normalize=True, nrow=int(np.sqrt(self.num_samples)), range=(-0.5,0.5)) 
        if temp:
            print(f'loss:{loss_value.item():.3f} / log(p):{log_p.mean():.3f} / log(det):{logdet.mean():.3f}')
        else:
            print(f'Epoch: {idx+1} / loss:{loss_value / len(dataloader.dataset):.3f}')

    def generate(self, num_samples=16):
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        with torch.no_grad():
            sample_z = self.sample_z(num_samples=num_samples)
            gen_imgs = self.model.reverse(sample_z)
            torchvision.utils.save_image(gen_imgs.detach().cpu(), f'Generated_Images.jpg', 
                                        normalize=True, nrow=int(np.sqrt(num_samples)), value_range=(-0.5,0.5))


if __name__ == "__main__":
    img_size = 128
    batch_size = 16
    root = '/home/sk851/data/celeba_hq'
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    # data = torchvision.datasets.MNIST('/home/sk851/data', transform=transform, train=True, download=True)
    # data = torchvision.datasets.CIFAR10(root='/home/sk851/data', download=True, transform=transform)
    data = torchvision.datasets.ImageFolder(root, transform=transform)

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    glow = Glow(img_size, in_channel=3, n_flow=32, n_block=4, save_path='./glow.pt', device=device, is_load=False)
    glow.train(dataloader, epoch=100, lr=1e-5, verbose=10)
    # glow.generate()