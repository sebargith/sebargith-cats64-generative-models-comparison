# training/generate_dcgan_samples.py
from pathlib import Path

import torch
from torchvision.utils import save_image

from models.dcgan import GeneratorDCGAN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    z_dim = 100
    num_samples = 64  # para un grid, luego puedes generar miles

    # Cargar el generador entrenado
    netG = GeneratorDCGAN(z_dim=z_dim).to(device)
    state_dict = torch.load("outputs/models/dcgan_G.pth", map_location=device)
    netG.load_state_dict(state_dict)
    netG.eval()

    out_dir = Path("outputs/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, 1, 1, device=device)
        fake = netG(z)
        fake = (fake + 1) / 2  # de [-1,1] a [0,1]

        save_image(
            fake,
            out_dir / "dcgan_final_grid.png",
            nrow=8,
        )

    print("Guardado grid en outputs/samples/dcgan_final_grid.png")


if __name__ == "__main__":
    main()
