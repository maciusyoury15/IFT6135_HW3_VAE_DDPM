from __future__ import print_function
import argparse
import torch
import matplotlib.pyplot as plt
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from q1_vae import *

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.epochs = 20

torch.manual_seed(args.seed)

# print("Epochs: ", args.epochs)
# print("Batch size: ", args.batch_size)
# print("CUDA: ", args.cuda)
# print("Seed: ", args.seed)
# print("Log interval: ", args.log_interval, "\n\n")

if args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, logvar):
    batch_size = mu.size(0)

    # Reconstruction loss: Negative log-likelihood under Bernoulli
    recon_loss = -log_likelihood_bernoulli(recon_x, x.view(-1, 784)).sum()

    # KL divergence between q(z|x) = N(mu, sigma^2) and p(z) = N(0, I)
    mu_p = torch.zeros_like(mu)
    logvar_p = torch.zeros_like(logvar)

    kl = kl_gaussian_gaussian_analytic(mu, logvar, mu_p, logvar_p).sum()

    return recon_loss + kl


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    train_loss /= len(train_loader.dataset)
    print(f'====> Training loss (average) after epoch {epoch}: {train_loss:.4f}')
    return train_loss


def validate(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            val_loss += loss.item()

    val_loss /= len(test_loader.dataset)
    print(f"====> Validation loss (average) after epoch {epoch}: {val_loss:.4f}")
    return val_loss


# if __name__ == "__main__":
#     train_losses = []
#     val_losses = []
#
#     for epoch in range(1, args.epochs + 1):
#         train_loss = train(epoch)
#         val_loss = validate(epoch)
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#
#     torch.save(model.state_dict(), 'vae_model.pt')