import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(
        self, in_channels: int, latent_dim: int, hidden_dims: list = None, **kwargs
    ) -> None:

        super(self).__init__()

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
            result = self.encoder(x)
            result = torch.flatten(result, start_dim=1)

            mu = self.fc_mu(result)
            logvar = self.fc_var(result)

            return [mu, logvar]

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            result = self.decoder_input(z)
            result = result.view(-1, 512, 2, 2)
            result = self.decoder(result)
            result = self.final_layer(result)
            return result

        def reparameterize(
            self, mu: torch.Tensor, logvar: torch.Tensor
        ) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu

        def forward(self, x: torch.Tensor, **kwargs) -> list[torch.Tensor]:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return [self.decode(z), x, mu, logvar]

        def loss(self, *args, **kwargs) -> dict:
            reconstruction = args[0]
            og_input = args[1]
            mu = args[2]
            logvar = args[3]

            kld_weight = kwargs["M_N"]  # account for minibatch samples from dataset
            reconstruction_loss = F.mse_loss(reconstruction, og_input)

            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kld_loss = kld_loss.mean(dim=0)

            loss = reconstruction_loss + kld_weight * kld_loss
            return {
                "loss": loss,
                "Reconstruction_Loss": reconstruction_loss,
                "KLD": -kld_loss,
            }

        def sample(
            self, num_samples: int, current_device: int, **kwargs
        ) -> torch.Tensor:
            z = torch.randn(num_samples, self.latent_dim)
            z = z.to(current_device)
            samples = self.decode(z)
            return samples

        def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            return self.forward(x)[0]
