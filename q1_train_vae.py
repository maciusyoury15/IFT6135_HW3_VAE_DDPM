from __future__ import print_function
import argparse
import torch
import os
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

gdrive_dir = '/content/drive/MyDrive/IFT6135_HW3'
save_dir = os.makedirs(gdrive_dir, 'vae', exist_ok=True)


if args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(f'{save_dir}/data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(f'{save_dir}/data', train=False, transform=transforms.ToTensor()),
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


def plot_losses(train_losses, val_losses, save_dir=save_dir):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))

    # Plot with specified colors
    plt.plot(epochs, train_losses, color='red', marker='o', label='Training Loss')
    plt.plot(epochs, val_losses, color='blue', marker='s', label='Validation Loss')

    # Dashed threshold line
    plt.axhline(y=104, color='red', linestyle='--', label='Target Threshold (104)')

    # Labels and styling
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/vae_loss_plot.png")
    plt.show()


def generate_samples(model, n_samples=16, latent_dim=20):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)  # Sample from N(0, I)
        samples = model.decode(z).cpu()  # Decode to image space
        return samples.view(n_samples, 1, 28, 28)  # For MNIST-like data


def show_samples(samples, nrow=4, title="Generated Samples", save_dir=save_dir):
    n_samples = samples.size(0)
    fig, axs = plt.subplots(nrow, nrow, figsize=(6, 6))
    axs = axs.flatten()

    for i in range(n_samples):
        axs[i].imshow(samples[i].squeeze(), cmap='gray')
        axs[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/vae_visual_samples_plot.png")
    plt.show()

def latent_traversal_grid(model, latent_dim=20, steps=5, epsilon=2.0):
    model.eval()
    with torch.no_grad():
        # 1. Base z ~ N(0, I)
        base_z = torch.randn(1, latent_dim).to(device)

        # 2. Prepare grid of z variants
        traversal_range = torch.linspace(-epsilon, epsilon, steps).to(device)
        images = []

        for i in range(latent_dim):
            row = []
            for shift in traversal_range:
                z_new = base_z.clone()
                z_new[0, i] += shift
                img = model.decode(z_new).view(28, 28).cpu()
                row.append(img)
            images.append(row)

        return images

def show_latent_traversals(images, steps=5, save_dir=save_dir):
    latent_dim = len(images)
    fig, axs = plt.subplots(latent_dim, steps, figsize=(steps, latent_dim))

    for i in range(latent_dim):
        for j in range(steps):
            axs[i, j].imshow(images[i][j], cmap='gray')
            axs[i, j].axis('off')

    plt.suptitle("Latent Traversals (1 row per latent dimension)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/vae_latent_traversals.png")
    plt.show()


def interpolate_latent_vs_data(model, latent_dim=20, steps=11):
    model.eval()
    with torch.no_grad():
        # Sample z0 and z1 from N(0, I)
        z0 = torch.randn(1, latent_dim).to(device)
        z1 = torch.randn(1, latent_dim).to(device)

        # Decode z0 and z1
        x0 = model.decode(z0).view(28, 28).cpu()
        x1 = model.decode(z1).view(28, 28).cpu()

        alphas = torch.linspace(0, 1, steps).to(device)

        # Latent space interpolation
        latent_imgs = []
        for alpha in alphas:
            z_alpha = alpha * z0 + (1 - alpha) * z1
            x_alpha = model.decode(z_alpha).view(28, 28).cpu()
            latent_imgs.append(x_alpha)

        # Data space interpolation
        data_imgs = []
        for alpha in alphas:
            x_hat = alpha * x0 + (1 - alpha) * x1
            data_imgs.append(x_hat)

        return latent_imgs, data_imgs


def plot_interpolations(latent_imgs, data_imgs, save_dir=save_dir):
    steps = len(latent_imgs)
    fig, axs = plt.subplots(2, steps, figsize=(steps, 2))

    for i in range(steps):
        axs[0, i].imshow(latent_imgs[i], cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(data_imgs[i], cmap='gray')
        axs[1, i].axis('off')

    axs[0, 0].set_ylabel("Latent", fontsize=12)
    axs[1, 0].set_ylabel("Data", fontsize=12)
    plt.suptitle(title="Latent vs Data Space Interpolation")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/vae_latent_data_space_plot.png")
    plt.show()


if __name__ == "__main__":
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        val_loss = validate(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    torch.save(model.state_dict(), 'vae_model.pt')

    # Plot training and validation losses
    plot_losses(train_losses, val_losses)

    samples = generate_samples(model, n_samples=16)
    show_samples(samples)

    images = latent_traversal_grid(model, latent_dim=20, steps=5, epsilon=2.5)
    show_latent_traversals(images)