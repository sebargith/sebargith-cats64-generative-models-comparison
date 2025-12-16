# training/train_ddpm.py
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from utils.dataset import CatDataset64
from models.ddpm_unet import UNetDDPM


def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    # Schedule lineal de betas entre beta_start y beta_end
    return torch.linspace(beta_start, beta_end, T)


def main():
    # -------------------
    # Hiperparámetros
    # -------------------
    T = 1000
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
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # alpha_bar_t
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # -------------------
    # Entrenamiento (MVP)
    # -------------------
    global_step = 0
    model.train()

    for epoch in range(num_epochs):
        for x0 in train_loader:
            x0 = x0.to(device)  # (N,3,64,64) en [-1,1]

            # t ~ Uniform(0, T-1)
            t = torch.randint(0, T, (x0.size(0),), device=device).long()

            # epsilon ~ N(0, I)
            noise = torch.randn_like(x0)

            # x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*epsilon
            sqrt_alpha_bar_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

            # predicción del ruido
            pred_noise = model(x_t, t)

            # MSE( epsilon, epsilon_hat )
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step {global_step} "
                    f"Loss: {loss.item():.4f}"
                )

            global_step += 1

    print("Entrenamiento DDPM (MVP) finalizado.")


if __name__ == "__main__":
    main()
