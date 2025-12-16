# training/sample_ddpm.py
from pathlib import Path

import torch
from torchvision.utils import save_image

from models.ddpm_unet import UNetDDPM


def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    return torch.linspace(beta_start, beta_end, T)


@torch.no_grad()
def sample_ddpm(
    model,
    n_samples,
    img_size,
    betas,
    alphas,
    alphas_cumprod,
    alphas_cumprod_prev,
    posterior_variance,
    device,
):
    model.eval()
    T = betas.shape[0]

    x = torch.randn(n_samples, 3, img_size, img_size, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

        eps_theta = model(x, t_batch)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_prev = alphas_cumprod_prev[t]

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
    return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # mismos valores que en el entrenamiento
    T = 1000
    beta_start = 1e-4
    beta_end = 2e-2
    img_size = 64

    betas = make_beta_schedule(T, beta_start, beta_end).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0
    )
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    # cargar modelo final
    model = UNetDDPM(img_channels=3, base_channels=64, time_emb_dim=256).to(device)
    state_dict = torch.load("outputs/models/ddpm_unet_final.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Checkpoint cargado: outputs/models/ddpm_unet_final.pth")

    # generar 64 muestras y guardarlas en un grid 8x8
    n_samples = 64
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

    samples = (samples + 1) / 2.0  # [-1,1] -> [0,1]

    out_dir = Path("outputs/samples_ddpm")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ddpm_final_grid.png"
    save_image(samples, out_path, nrow=8)
    print(f"Grid final guardado en: {out_path}")


if __name__ == "__main__":
    main()
