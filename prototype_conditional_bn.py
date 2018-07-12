import torch
from torch import nn
from torch import optim
import numpy as np
from torchvision.utils import save_image
import utils

class CBN1d(nn.Module):
    def __init__(self, in_f, bn_f):
        super(CBN1d, self).__init__()
        self.bn = nn.BatchNorm1d(bn_f, affine=False)
        self.scale = nn.Embedding(in_f, bn_f)
        self.shift = nn.Embedding(in_f, bn_f)
    def forward(self, x, y):
        scale = self.scale(y)
        shift = self.shift(y)
        return self.bn(x)*scale + shift

class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, class_num=10):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.linear = nn.Linear(self.input_dim, 1024)
        self.cbn = CBN1d(class_num, 1024)
        self.relu = nn.ReLU()

        self.fc = nn.Sequential(
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = self.linear(input)
        x = self.cbn(x, label)
        x = self.relu(x)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num
        self.embed = nn.Embedding(class_num, 1024, max_norm=1)

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.psi = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.phi = nn.Sequential(
            nn.Linear(1024, self.output_dim)
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.psi(x)
        y_proj = self.embed(label)
        result = self.phi(x) + torch.sum(y_proj*x, dim=1, keepdim=True)
        return result

from torchvision.datasets import MNIST

mnist = MNIST(root="./data", download=True)

from torch.utils.data import DataLoader, TensorDataset

X_train = mnist.train_data
X_train = X_train.view(60000, 1, 28, 28)

ds = TensorDataset(X_train, mnist.train_labels)
data_loader = DataLoader(ds, batch_size=64, shuffle=True)
data_loader

disc = Discriminator(input_size=28)
gen = Generator(input_size=28)
print(disc)
print(gen)
disc.cuda()
gen.cuda()

optim_g = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5,0.999))
optim_d = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5,0.999))

for epoch in range(100):
    gen.train()
    disc.train()
    d_losses = []
    g_losses = []
    for (X_batch, y_batch) in data_loader:
        X_batch = X_batch.float().cuda()
        X_batch = ((X_batch/255.)-0.5) / 0.5
        y_batch = y_batch.long().cuda()
        bs = X_batch.size(0)
        optim_g.zero_grad()
        optim_d.zero_grad()
        # Do the generator.
        z_batch = torch.randn(bs, 100).cuda()
        y_fake = torch.randint(10, size=(bs,)).long().cuda()
        gz = gen(z_batch, y_fake)
        disc_g = disc(gz, y_fake)
        disc_g_loss = torch.mean((disc_g - 1.)**2)
        disc_g_loss.backward()
        optim_g.step()
        # Do the discriminator.
        optim_d.zero_grad()
        disc_real = disc(X_batch, y_batch)
        disc_fake = disc(gz.detach(), y_fake)
        disc_loss = torch.mean((disc_real - 1.)**2) + \
                    torch.mean((disc_fake - 0.)**2)
        disc_loss.backward()
        optim_d.step()
        # Log losses.
        g_losses.append(disc_g_loss.item())
        d_losses.append( (disc_loss / 2.).item() )
    print( epoch+1, np.mean(g_losses), np.mean(d_losses) )
    save_image(gz*0.5 + 0.5, "epoch_%i.png" % (epoch+1))
    with torch.no_grad():
        gen.eval()
        y_cat = []
        for i in range(10):
            y_cat += [i]*10
        y_cat = torch.from_numpy(np.asarray(y_cat)).long().cuda()
        z_batch = torch.randn(len(y_cat), 100).cuda()
        gz_cat = gen(z_batch, y_cat)
        save_image(gz_cat*0.5 + 0.5, "cat_epoch_%i.png" % (epoch+1))
