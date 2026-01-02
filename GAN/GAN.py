# Importing Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
device = "cuda" if torch.cuda.is_available() else "cpu"


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # In GANs Leaky ReLU is often times a better choice
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    # Taking input as dimension of latent noise (or just noise)
    def __init__(self, latent_dim, img_dim):
        super().__init__()
        # Tanh used here cuz we want to normalize the output to [-1, 1]
        self.gen = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


# Optimization: Proper Weight Initialization (Xavier) for paper stability
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


# Hyperparameters [GANs are very sensitive to hyperparameters]
lr = 3e-4  # Karpathy Constant, Works best with Adam generally
latent_dim = 64
img_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 12

disc = Discriminator(img_dim).to(device)
gen = Generator(latent_dim, img_dim).to(device)
initialize_weights(disc)
initialize_weights(gen)

# Extracting from Nr Dn (Normal Distribution)
fixed_noise = torch.randn(batch_size, latent_dim).to(device)

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()  # Follows the form of the GAN equation

# Outputs the fake imgs that the Gr has generated
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        curr_batch_size = real.shape[0]

        # Train Dr: max log(D(real)) + log(1 - D(G(z)))
        noise = torch.randn(curr_batch_size, latent_dim).to(device)
        fake = gen(noise)

        # This is what the discriminator outputs on the real parts
        # Optimization: Label smoothing (0.9) improves stability
        disc_real = disc(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real) * 0.9)

        # This is what the discriminator outputs on the fake parts
        # Optimization: .detach() saves memory compared to retain_graph=True
        disc_fake = disc(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        disc.zero_grad()
        # Optimization note: Using detach() above avoids the need for retain_graph=True,
        # which saves computation time and memory.
        loss_disc.backward()
        opt_disc.step()

        # Train Gr min log(1- D(G(z))) <-> max log (D(G(z)) [Cuz of saturating grads]
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] Batch {batch_idx}/{len(loader)} "
                f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake_display = gen(fixed_noise).reshape(-1, 1, 28, 28)
                real_display = real.reshape(-1, 1, 28, 28)
                img_grid_real = torchvision.utils.make_grid(real_display, normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake_display, normalize=True)

                writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("MNIST Real Images", img_grid_real, global_step=step)
                step += 1