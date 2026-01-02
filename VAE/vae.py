import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Hyperparameters
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 12
BATCH_SIZE = 64
LR_RATE = 3e-4  # Karpathy Constant


# VAE Model Definition
class VAE(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        # Encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)

        # Optimization: Standard VAEs output log-variance for numerical stability
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)  # This acts as log-variance

        # Decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()
        self.training = True

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu, logvar = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, logvar

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def reparameterization(self, mu, logvar):
        # Optimization: Conversion from log-variance to standard deviation
        std = torch.exp(0.5 * logvar)

        # Sampling epsilon for latent space with distribution from Gamma(3,2)
        # Note: Standard VAEs typically use torch.randn_like(std) for Gaussian priors
        gamma_distribution = torch.distributions.Gamma(3.0, 2.0)
        epsilon = gamma_distribution.sample(std.shape).to(device)

        z = mu + std * epsilon
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z_reparametrized = self.reparameterization(mu, logvar)
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, logvar


# Metrics and Loss Function

def loss_fn(x, x_hat, mean, logvar):
    """
    Optimization: Added proper argument passing to avoid global variable leaks.
    Captures ELBO components: Reconstruction Loss and KL Divergence.
    """
    # Reconstruction Loss: Measures how well the VAE reconstructs the input
    recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')

    # KL Divergence: Measures how much the learned distribution deviates from the prior
    # Calculation assumes a Gaussian prior N(0, I)
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return recon_loss, kl_div


# Loading The Dataset

mnist_transform = transforms.Compose([transforms.ToTensor()])
dataset = MNIST(root="dataset/", train=True, transform=mnist_transform, download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VAE(INPUT_DIM, H_DIM, Z_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)

# Training with Metric Tracking

# Metric storage for paper implementation plotting
history = {
    'total_loss': [],
    'recon_loss': [],
    'kl_div': []
}

for epoch in range(NUM_EPOCHS):
    overall_loss = 0
    overall_recon = 0
    overall_kl = 0

    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        x = x.view(x.shape[0], INPUT_DIM).to(device)

        # Forward pass
        x_recon, mu, logvar = model(x)
        x_recon = torch.clamp(x_recon, 1e-7, 1 - 1e-7)  # Optimization: Avoid log(0) in BCE

        # Backprop with tracked metrics
        recon_loss, kl_div = loss_fn(x, x_recon, mu, logvar)
        loss = recon_loss + kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        overall_loss += loss.item()
        overall_recon += recon_loss.item()
        overall_kl += kl_div.item()

        loop.set_postfix(loss=loss.item())

    # Calculate averages for the epoch
    avg_loss = overall_loss / (len(train_loader) * BATCH_SIZE)
    history['total_loss'].append(avg_loss)
    history['recon_loss'].append(overall_recon / (len(train_loader) * BATCH_SIZE))
    history['kl_div'].append(overall_kl / (len(train_loader) * BATCH_SIZE))

    print(f"\tEpoch {epoch + 1} | Avg Loss: {avg_loss:.4f} | Recon: {overall_recon:.2f} | KL: {overall_kl:.2f}")


# Inference and Generation

def inference(digit, num_examples=1):
    model.eval()  # Set to evaluation mode
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10: break

    with torch.no_grad():
        mu, logvar = model.encode(images[digit].view(1, 784).to(device))
        for example in range(num_examples):
            z = model.reparameterization(mu, logvar)
            out = model.decode(z).cpu().view(-1, 1, 28, 28)
            save_image(out, f"generated_{digit}_ex{example}.png")


for idx in range(10):
    inference(idx, num_examples=1)