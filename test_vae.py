
import torch
from model import VAE
from train import loss_function


def test_encoder_decoder_architecture():
    model = VAE(latent_dim=2)

    images = torch.rand(4, 1, 28, 28)
    reconstruction, mu, logvar = model(images)

    assert reconstruction.shape == (4, 1, 28, 28)
    assert mu.shape == (4, 2)
    assert logvar.shape == (4, 2)


def test_latent_sampling():
    model = VAE(latent_dim=2)

    images = torch.rand(3, 1, 28, 28)
    mu, logvar = model.encode(images)

    z = model.reparameterize(mu, logvar)

    assert z.shape == (3, 2)
    assert torch.isfinite(z).all()


def test_reconstruction_loss():
    images = torch.rand(3, 1, 28, 28)

    reconstruction = torch.rand_like(images)

    loss = torch.nn.functional.binary_cross_entropy(
        reconstruction,
        images,
        reduction="sum"
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_kl_divergence():
    mu = torch.zeros(3, 2)
    logvar = torch.zeros(3, 2)

    kl_divergence = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    # When mu = 0 and logvar = 0, KL divergence is zero.
    assert torch.isclose(
        kl_divergence,
        torch.tensor(0.0)
    )


def test_total_loss():
    model = VAE(latent_dim=2)
    images = torch.rand(3, 1, 28, 28)

    reconstruction, mu, logvar = model(images)

    total, reconstruction_loss, kl = loss_function(
        reconstruction, images, mu, logvar
    )

    assert torch.isfinite(total)
    assert torch.allclose(
        total, reconstruction_loss + kl
    )
