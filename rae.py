from __future__ import print_function
import argparse
import math
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torchvision import datasets, transforms
from torchvision.utils import save_image
from matplotlib import pyplot as plt

ALPHA = 1
BATCH_SIZE = 32
ITERATIONS = 10

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)

def snlinear(in_features, out_features):
    linear = nn.Linear(in_features, out_features)
    torch.nn.init.kaiming_normal_ (linear.weight, mode='fan_in', nonlinearity='linear')
    return linear

class RAE(nn.Module):
    def __init__(self):
        super(RAE, self).__init__()

        self.encoder = nn.Sequential(
              nn.Linear(784, 400),
              nn.ReLU(),
              nn.Linear(400, 20),
              nn.ReLU(),
              nn.Linear(20, 2)
        )

        self.decoder = nn.Sequential(
              nn.Linear(2, 20),
              nn.ReLU(),
              nn.Linear(20, 400),
              nn.ReLU(),
              nn.Linear(400, 784)
        )

    def encode(self, x):
        x = x.view(-1, 784)
        return self.encoder(x)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), self.reg(z)

    def reg(self, z):
        l2 = 0
        for p in self.decoder.parameters():
            l2 += p.pow(2).sum()
        return l2 + (0.5 * z.pow(2).sum())


model = RAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, reg = model(data)
        loss = F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction="sum") + ALPHA * reg
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, reg = model(data)
            loss = F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction="sum") + ALPHA * reg
            test_loss += loss
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results_rae/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, ITERATIONS + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 2).to(device)
            sample_x = model.decode(sample)
            sample_x = sample_x.cpu()
            save_image(sample_x.view(64, 1, 28, 28),
                       'results_rae/sample_' + str(epoch) + '.png')
        
    z_list, y_list = [], []
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        z = model.encode(x_test).cpu().detach()
        z_list.append(z), y_list.append(y_test)
    z = torch.cat(z_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy()
    plt.scatter(z[:, 0], z[:, 1], c=y)
    plt.savefig("latent_space.png")
    plt.show()