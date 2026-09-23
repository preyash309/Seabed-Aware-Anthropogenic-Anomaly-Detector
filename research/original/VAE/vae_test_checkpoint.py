import torch
import torch.nn as nn
from pathlib import Path


CHECKPOINT = Path(
    r"E:\SIH\SIH_Results\vae_normal_seabed\checkpoints\best.pt"
)

LATENT_DIM = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ConvVAE(nn.Module):

    def __init__(self, latent_dim=128):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            512 * 8 * 8,
            latent_dim
        )

        self.fc_decode = nn.Linear(
            latent_dim,
            512 * 8 * 8
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):

        h = self.encoder(x)

        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        logvar = torch.clamp(
            logvar,
            -10,
            10
        )

        return mu, logvar

    def decode(self, z):

        h = self.fc_decode(z)

        h = h.view(
            -1,
            512,
            8,
            8
        )

        return self.decoder(h)

    def forward(self, x):

        mu, logvar = self.encode(x)

        reconstruction = self.decode(mu)

        return reconstruction, mu, logvar


print("=" * 70)
print("VAE CHECKPOINT COMPATIBILITY TEST")
print("=" * 70)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model = ConvVAE(
    latent_dim=LATENT_DIM
)

model.load_state_dict(
    state_dict,
    strict=True
)

model.to(DEVICE)
model.eval()

print()
print("SUCCESS!")
print("Checkpoint matches evaluation architecture exactly.")

x = torch.rand(
    2,
    1,
    256,
    256,
    device=DEVICE
)

with torch.no_grad():

    recon, mu, logvar = model(x)

print()
print("Input shape :", tuple(x.shape))
print("Output shape:", tuple(recon.shape))
print("Mu shape    :", tuple(mu.shape))
print("Logvar shape:", tuple(logvar.shape))

print()
print("Input finite :", torch.isfinite(x).all().item())
print("Output finite:", torch.isfinite(recon).all().item())

print()
print("=" * 70)
print("CHECK PASSED")
print("=" * 70)