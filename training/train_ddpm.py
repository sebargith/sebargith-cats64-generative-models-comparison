# training/train_ddpm.py
from pathlib import Path
import time  # para medir tiempos

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.dataset import CatDataset64
from models.ddpm_unet import UNetDDPM


def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """
    Schedule lineal de betas entre beta_start y beta_end.
    """
    return torch.linspace(beta_start, beta_end, T)


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    n_samples: int,
    img_size: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    alphas_cumprod_prev: torch.Tensor,
    posterior_variance: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Muestreo DDPM clásico usando todos los pasos T.
    Retorna imágenes en [-1, 1] de tamaño (n_samples, 3, img_size, img_size).
    """
    model.eval()
    T = betas.shape[0]

    # x_T ~ N(0, I)
    x = torch.randn(n_samples, 3, img_size, img_size, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

        # ε_θ(x_t, t)
        eps_theta = model(x, t_batch)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_prev = alphas_cumprod_prev[t]

        # media del posterior p(x_{t-1} | x_t, ε_θ)
        # μ_t = 1/sqrt(α_t) * ( x_t - (β_t / sqrt(1 - ᾱ_t)) * ε_θ )
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
        )

        if t > 0:
            noise = torch.randn_like(x)
            var = posterior_variance[t]
            x = mean + torch.sqrt(var) * noise
        else:
            x = mean

    x = x.clamp(-1.0, 1.0)
    model.train()
    return x


def main():
    # -------------------
    # Hiperparámetros
    # -------------------
    T = 1000                   # número de pasos de difusión
    beta_start = 1e-4
    beta_end = 2e-2

    batch_size = 128
    num_epochs = 50
    lr = 2e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------------------
    # Dataset / DataLoader
    # -------------------
    train_ds = CatDataset64("data/splits/train.txt", train=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # -------------------
    # Modelo DDPM (U-Net)
    # -------------------
    model = UNetDDPM(img_channels=3, base_channels=64, time_emb_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------
    # Pre-cálculo del schedule
    # -------------------
    betas = make_beta_schedule(T, beta_start, beta_end).to(device)  # (T,)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Para muestreo posterior
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0], device=device), alphas_cumprod[:-1]],
        dim=0,
    )
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    # -------------------
    # Carpetas de salida
    # -------------------
    models_dir = Path("outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = Path("outputs/samples_ddpm")
    samples_dir.mkdir(parents=True, exist_ok=True)

    # -------------------
    # Historial de pérdidas y tiempos
    # -------------------
    loss_history = []
    step_history = []
    epoch_times = []  # tiempo por época
    global_step = 0

    model.train()

    # Timer global de entrenamiento
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_start = time.perf_counter()

    for epoch in range(num_epochs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()

        for x0 in train_loader:
            x0 = x0.to(device)  # (N,3,64,64), en [-1,1]

            # t ~ Uniform(0, T-1) por batch
            t = torch.randint(0, T, (x0.size(0),), device=device).long()

            # ruido epsilon ~ N(0,I)
            noise = torch.randn_like(x0)

            # x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε
            sqrt_alpha_bar_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

            # predicción del ruido
            pred_noise = model(x_t, t)

            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            loss_history.append(loss.item())
            step_history.append(global_step)
            if global_step % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step {global_step} "
                    f"Loss: {loss.item():.4f}"
                )

            global_step += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        steps_this_epoch = len(train_loader)
        sec_per_step_epoch = epoch_time / max(steps_this_epoch, 1)

        print(
            f"--> Epoch {epoch+1}/{num_epochs} terminada en "
            f"{epoch_time:.2f} s "
            f"({sec_per_step_epoch:.4f} s/step aprox.)"
        )

        # Guardar checkpoint en *cada* época
        torch.save(
            model.state_dict(),
            models_dir / f"ddpm_unet_epoch{epoch+1}.pth"
        )

        # Generar samples cada 5 épocas
        if (epoch + 1) % 5 == 0:
            n_samples = 64
            img_size = 64

            samples = sample_ddpm(
                model=model,
                n_samples=n_samples,
                img_size=img_size,
                betas=betas,
                alphas=alphas,
                alphas_cumprod=alphas_cumprod,
                alphas_cumprod_prev=alphas_cumprod_prev,
                posterior_variance=posterior_variance,
                device=device,
            )

            # pasar de [-1,1] a [0,1]
            samples = (samples + 1) / 2.0

            save_image(
                samples,
                samples_dir / f"ddpm_epoch{epoch+1}.png",
                nrow=8,
            )
            print(f"Samples DDPM guardados en {samples_dir / f'ddpm_epoch{epoch+1}.png'}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_train_time = time.perf_counter() - train_start
    sec_per_step_total = total_train_time / max(global_step, 1)

    print(f"=== Entrenamiento DDPM finalizado en {total_train_time:.2f} s ===")
    print(f"Pasos totales: {global_step}, ~{sec_per_step_total:.4f} s/step de media")

    # Guardar modelo final
    torch.save(model.state_dict(), models_dir / "ddpm_unet_final.pth")

    # Guardar curva de pérdida
    plt.figure()
    plt.plot(step_history, loss_history, label="DDPM train loss")
    plt.xlabel("Step")
    plt.ylabel("MSE loss (ε vs ε̂)")
    plt.title("Entrenamiento DDPM – pérdida de predicción de ruido")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "ddpm_loss_curve.png", dpi=150)
    plt.close()

    # Guardar gráfico de tiempo por época
    epochs = list(range(1, num_epochs + 1))
    plt.figure()
    plt.plot(epochs, epoch_times, marker="o")
    plt.xlabel("Época")
    plt.ylabel("Tiempo por época (s)")
    plt.title("DDPM – evolución del tiempo por época")
    plt.tight_layout()
    plt.savefig(plots_dir / "ddpm_epoch_times.png", dpi=150)
    plt.close()

    print(f"Modelo final guardado en: {models_dir / 'ddpm_unet_final.pth'}")
    print(f"Curva de loss guardada en: {plots_dir / 'ddpm_loss_curve.png'}")
    print(f"Gráfico de tiempos por época guardado en: {plots_dir / 'ddpm_epoch_times.png'}")


if __name__ == "__main__":
    main()
