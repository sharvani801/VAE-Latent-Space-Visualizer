
import os
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import datasets, transforms
from model import VAE

MODEL_PATH = "models/vae_mnist.pth"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.set_page_config(
    page_title="VAE Latent Space Visualizer",
    layout="wide"
)

st.title("VAE Latent Space Visualizer")
st.write(
    "Upload a handwritten digit to encode it into latent space "
    "and reconstruct it using a Variational Autoencoder."
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE
    )

    latent_dim = checkpoint.get("latent_dim", 2)
    model = VAE(latent_dim=latent_dim).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


model = load_model()

if model is None:
    st.warning(
        "Trained model not found. Run python train.py first."
    )
    st.stop()


def prepare_image(pil_image):
    image = ImageOps.grayscale(pil_image)
    image = ImageOps.fit(image, (28, 28))

    array = np.asarray(image).astype(np.float32) / 255.0

    # MNIST uses bright digits on a dark background.
    if array.mean() > 0.5:
        array = 1.0 - array

    tensor = torch.from_numpy(array)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


upload = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

if upload is not None:
    original_image = Image.open(upload).convert("RGB")
    input_tensor = prepare_image(original_image).to(DEVICE)

    with torch.no_grad():
        mu, logvar = model.encode(input_tensor)

        # Use the mean for a stable reconstruction.
        latent_vector = mu

        reconstruction = model.decode(latent_vector)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        fig, ax = plt.subplots()
        ax.imshow(
            input_tensor[0, 0].cpu().numpy(),
            cmap="gray"
        )
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Reconstructed Image")
        fig, ax = plt.subplots()
        ax.imshow(
            reconstruction[0, 0].cpu().numpy(),
            cmap="gray",
            vmin=0,
            vmax=1
        )
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Latent Space Visualization")

    z1 = mu[0, 0].item()
    z2 = mu[0, 1].item()

    fig, ax = plt.subplots()

    ax.scatter(z1, z2, s=120, label="Uploaded image")
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.set_title("Encoded Image Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.write(f"Latent coordinates: ({z1:.3f}, {z2:.3f})")

st.divider()

st.subheader("Explore MNIST Latent Space")

if st.button("Generate Latent Space Plot"):
    with st.spinner("Encoding MNIST images..."):
        dataset = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )

        count = 2000

        images = torch.stack(
            [dataset[i][0] for i in range(count)]
        ).to(DEVICE)

        labels = np.array(
            [dataset[i][1] for i in range(count)]
        )

        with torch.no_grad():
            mu, logvar = model.encode(images)

        coordinates = mu.cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=labels,
            cmap="tab10",
            s=8,
            alpha=0.7
        )

        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")
        ax.set_title("MNIST Latent Space")

        fig.colorbar(
            scatter,
            ax=ax,
            label="Digit Label"
        )

        st.pyplot(fig)
        plt.close(fig)

st.divider()

st.subheader("Model Information")
st.write("Architecture: Encoder → Latent Space → Decoder")
st.write("Latent dimension: 2")
st.write("Loss: Reconstruction Loss + KL Divergence")
st.write(f"Running on: {DEVICE}")
