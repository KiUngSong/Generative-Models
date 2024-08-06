import torch, torchvision
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import glob, os, random, itertools, copy


##################################
###  Define model architecture
##################################

# Function to initialize weight of model
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)

# channel_in : Number of input image channels e.g. 3
# onum_c : Number of output image channels e.g. 3
class Generator(nn.Module):
    def __init__(self, channel_in, channel_out, n_downsample_block=2, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model=[ nn.ReflectionPad2d(3),
                nn.Conv2d(channel_in, 64, 7),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(n_downsample_block):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(n_downsample_block):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, channel_out, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # Since downsampling&upsampling process is symmetrical, agnostic to image size
        return self.model(x)

# num_c : Number of input image channels e.g. 3
class Discriminator(nn.Module):
    def __init__(self, num_c, d_dims=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        # Building blocks
        model = [   nn.Conv2d(num_c, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        for i in range(len(d_dims)-1):
            model += [  nn.Conv2d(d_dims[i], d_dims[i+1], 4, stride=2, padding=1),
                        nn.InstanceNorm2d(d_dims[i+1]), 
                        nn.LeakyReLU(0.2, inplace=True) ]

        # FC layer
        model += [nn.Conv2d(d_dims[-1], 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Apply global avg pooling and flatten to match batch size
        # Due to gloabl avg pooling agnostic to image size
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


# Function for training
def train(n_epochs, dataloader, G_A2B, G_B2A, D_A, D_B, 
            optimizer_G, optimizer_D_A, optimizer_D_B, test_dataloader,
            lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B, device):
    # Store fixed images for testing
    fixed_imgs = copy.deepcopy(next(iter(test_dataloader)))

    # Set Replaybuffer
    fake_A_buffer = ReplayBuffer(device)
    fake_B_buffer = ReplayBuffer(device)
    
    # Defince Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    for i in tqdm(range(n_epochs)):
        for _, batch in enumerate(dataloader):
            batch_size = batch['A'].size(0)
            y_real = torch.ones(batch_size, 1).to(device)
            y_fake = torch.zeros(batch_size, 1).to(device)

            # Set model input
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            ## Train Generators A2B & B2A
            optimizer_G.zero_grad()
            # Identity loss
            same_B = G_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            same_A = G_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0
            # GAN loss
            fake_B = G_A2B(real_A)
            pred_fake = D_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, y_real)
            fake_A = G_B2A(real_B)
            pred_fake = D_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, y_real)
            # Cycle loss
            recovered_A = G_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
            recovered_B = G_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            optimizer_G.step()

            ## Train Discriminator A
            optimizer_D_A.zero_grad()
            # Real loss
            pred_real = D_A(real_A)
            loss_D_real = criterion_GAN(pred_real, y_real)
            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = D_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, y_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            ## Train Discriminator B
            optimizer_D_B.zero_grad()
            # Real loss
            pred_real = D_B(real_B)
            loss_D_real = criterion_GAN(pred_real, y_real)
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = D_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, y_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()
            optimizer_D_B.step()
            
        if (i+1) % 5 == 0:
            print(f'Epoch: {i+1} / loss_G:{loss_G:.4f} / loss_G_identity:{loss_identity_A + loss_identity_B:.4f} / loss_G_GAN:{loss_GAN_A2B + loss_GAN_B2A:.4f} / ', 
                    f'loss_G_cycle:{loss_cycle_ABA + loss_cycle_BAB:.4f} / loss_D:{loss_D_A + loss_D_B:.4f}')

            plt.figure(figsize=(15,15))
            plt.subplot(1,2,1)
            plt.axis("off")
            plt.title("Original Images : summer")
            plt.imshow(np.transpose(torchvision.utils.make_grid(fixed_imgs['A'], nrow=4, padding=1, normalize=True).cpu(),(1,2,0)))

            G_A2B.eval()
            gen_img = G_A2B(fixed_imgs['A'].to(device))
            plt.subplot(1,2,2)
            plt.axis("off")
            plt.title("Tranfered Images : winter")
            plt.imshow(np.transpose(torchvision.utils.make_grid(gen_img.detach().cpu(), nrow=4, padding=1, normalize=True),(1,2,0)))
            plt.savefig('CycleGAN_Results.jpg')
            plt.close()
            G_A2B.train()
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()


##################################
###  Utils for data loading
##################################

class ReplayBuffer():
    def __init__(self, device, max_size=50):
        assert (max_size > 0)
        self.max_size = max_size
        self.buffer = []
        self.device = device

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            # If buffer has empty space, add element to buffer&return
            if len(self.buffer) < self.max_size:
                self.buffer.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.buffer[i].clone())
                    self.buffer[i] = element
                else:
                    to_return.append(element)

        return torch.cat(to_return).to(self.device)

class Dataset_Cycle(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        # Shuffle index of files_B
        item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


####################################
###  Execute Training at terminal
####################################
if __name__ == "__main__":
    batch_size = 4
    img_size = 256
    root = './data/summer2winter_yosemite'

    transform = [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    dataloader = DataLoader(Dataset_Cycle(root, transforms_=transform), batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(Dataset_Cycle(root, transforms_=transform, mode='test'), 
                                    batch_size=16, shuffle=False, num_workers=2)

    # Save train data example
    real = next(iter(dataloader))
    plt.figure(figsize=(15,10))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("A Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real['A'][:4], padding=1, normalize=True).cpu(),(1,2,0)))

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("B Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real['B'][:4], padding=1, normalize=True).cpu(),(1,2,0)))
    plt.savefig('CycleGAN_Train_Examples.jpg')
    plt.close()
    print("Example train images were saved")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:3" if cuda else "cpu")

    # Build generators and discriminators respectively
    G_A2B = Generator(channel_in=3, channel_out=3, n_downsample_block=2, n_residual_blocks=9).to(device)
    G_B2A = Generator(channel_in=3, channel_out=3, n_downsample_block=2, n_residual_blocks=9).to(device)
    D_A = Discriminator(num_c=3, d_dims=[64,128,256,512]).to(device)
    D_B = Discriminator(num_c=3, d_dims=[64,128,256,512]).to(device)

    # Initialize weight
    G_A2B.apply(weights_init_normal)
    G_B2A.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # Define optimizer
    lr = 0.0002
    optimizer_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.9)
    lr_scheduler_D_A = torch.optim.lr_scheduler.StepLR(optimizer_D_A, step_size=10, gamma=0.9)
    lr_scheduler_D_B = torch.optim.lr_scheduler.StepLR(optimizer_D_B, step_size=10, gamma=0.9)

    train(n_epochs=20, dataloader=dataloader, G_A2B=G_A2B, G_B2A=G_B2A, D_A=D_A, D_B=D_B, 
        optimizer_G=optimizer_G, optimizer_D_A=optimizer_D_A, optimizer_D_B=optimizer_D_B, test_dataloader=test_dataloader, 
        lr_scheduler_G=lr_scheduler_G, lr_scheduler_D_A=lr_scheduler_D_A, lr_scheduler_D_B=lr_scheduler_D_B, device=device)