# training/train_dcgan.py
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from utils.dataset import CatDataset64
from models.dcgan import GeneratorDCGAN, DiscriminatorDCGAN


def main():
    # Hiperparámetros básicos
    z_dim = 100
    batch_size = 256
    num_epochs = 50       # para prueba; luego puedes subir
    lr = 2e-4
    betas = (0.5, 0.999)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset & DataLoader
    train_ds = CatDataset64("data/splits/train.txt", train=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Modelos
    netG = GeneratorDCGAN(z_dim=z_dim).to(device)
    netD = DiscriminatorDCGAN().to(device)

    # Inicialización tipo DCGAN
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Optimizadores y pérdidas
    criterion = nn.BCEWithLogitsLoss()
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=betas)
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=betas)

    # Labels fijos para visualización
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    out_dir = Path("outputs/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Historial de pérdidas
    lossD_history = []
    lossG_history = []
    step_history = []

    global_step = 0

    for epoch in range(num_epochs):
        for real in train_loader:
            real = real.to(device)
            cur_bs = real.size(0)

            ### Entrenar Discriminador ###
            netD.zero_grad()

            # Real
            label_real = torch.ones(cur_bs, device=device)
            out_real = netD(real)
            loss_D_real = criterion(out_real, label_real)

            # Fake
            noise = torch.randn(cur_bs, z_dim, 1, 1, device=device)
            fake = netG(noise)
            label_fake = torch.zeros(cur_bs, device=device)
            out_fake = netD(fake.detach())
            loss_D_fake = criterion(out_fake, label_fake)

            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optD.step()

            ### Entrenar Generador ###
            netG.zero_grad()
            # Etiqueta real para que el generador engañe al D
            label_gen = torch.ones(cur_bs, device=device)
            out_fake_for_G = netD(fake)
            loss_G = criterion(out_fake_for_G, label_gen)
            loss_G.backward()
            optG.step()

            # Registrar pérdidas
            lossD_history.append(loss_D.item())
            lossG_history.append(loss_G.item())
            step_history.append(global_step)

            if global_step % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step {global_step} "
                    f"Loss_D: {loss_D.item():.4f} "
                    f"Loss_G: {loss_G.item():.4f}"
                )

            if global_step % 500 == 0:
                with torch.no_grad():
                    fake_samples = netG(fixed_noise).detach().cpu()
                    # Desnormalizar de [-1,1] a [0,1] para guardar
                    fake_samples = (fake_samples + 1) / 2
                    save_image(
                        fake_samples,
                        out_dir / f"dcgan_epoch{epoch+1}_step{global_step}.png",
                        nrow=8,
                    )

            global_step += 1

    # Guardar modelos finales
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    torch.save(netG.state_dict(), "outputs/models/dcgan_G.pth")
    torch.save(netD.state_dict(), "outputs/models/dcgan_D.pth")

    # Guardar curva de pérdidas
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(step_history, lossD_history, label="Loss D")
    plt.plot(step_history, lossG_history, label="Loss G")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Evolución de Loss DCGAN (D vs G)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "dcgan_loss_curves.png", dpi=150)
    plt.close()

    print("Curva de loss guardada en outputs/plots/dcgan_loss_curves.png")
    print("Entrenamiento finalizado y modelos guardados.")


if __name__ == "__main__":
    main()
