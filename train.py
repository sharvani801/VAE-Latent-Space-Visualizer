
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VAE

BATCH_SIZE = 128
EPOCHS = 5
LATENT_DIM = 2
LEARNING_RATE = 0.001


def loss_function(reconstruction, images, mu, logvar):
    # Reconstruction loss
    reconstruction_loss = torch.nn.functional.binary_cross_entropy(
        reconstruction, images, reduction="sum"
    )

    # KL divergence
    kl_divergence = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = reconstruction_loss + kl_divergence

    return total_loss, reconstruction_loss, kl_divergence


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Training on:", device)

    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = VAE(latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        model.train()

        total_sum = 0
        recon_sum = 0
        kl_sum = 0

        for images, labels in train_loader:
            images = images.to(device)

            optimizer.zero_grad()

            reconstruction, mu, logvar = model(images)

            loss, recon_loss, kl_loss = loss_function(
                reconstruction, images, mu, logvar
            )

            loss.backward()
            optimizer.step()

            total_sum += loss.item()
            recon_sum += recon_loss.item()
            kl_sum += kl_loss.item()

        n = len(train_data)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Total Loss: {total_sum / n:.2f} | "
            f"Reconstruction Loss: {recon_sum / n:.2f} | "
            f"KL Divergence: {kl_sum / n:.2f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": LATENT_DIM
        },
        "models/vae_mnist.pth"
    )

    print("Training completed!")
    print("Model saved in models/vae_mnist.pth")


if __name__ == "__main__":
    main()
