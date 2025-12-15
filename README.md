# Cats64 – DCGAN (PyTorch)

Repositorio de proyecto de **Generative Deep Learning** para entrenar un **DCGAN** y generar imágenes de **gatos 64×64** en PyTorch.

## Qué incluye?:
- `models/dcgan.py`: `GeneratorDCGAN` y `DiscriminatorDCGAN` (64×64).
- `utils/dataset.py`: `CatDataset64` con `Resize(64,64)` y `Normalize([0.5]*3, [0.5]*3)` → rango `[-1,1]`.
- `utils/prepare_splits.py`: crea `data/splits/train.txt` y `data/splits/val.txt` desde imágenes en `data/raw/`.
- `training/train_dcgan.py`: entrena DCGAN, guarda pesos, samples y curvas de loss.
- `training/generate_dcgan_samples.py`: carga el generador entrenado y genera un grid final.

## Requisitos
- Python 3.10+
- PyTorch + torchvision (CUDA opcional, recomendado)
- `pillow`, `matplotlib`

Para instalar requisitos:
```bash
conda create -n cats-gdl python=3.10 -y
conda activate cats-gdl
# instala PyTorch según tu CUDA (o CPU)
pip install pillow matplotlib
```

## Dataset

Coloca las imágenes (jpg/png) en:

data/raw/

Fuente:
https://www.kaggle.com/datasets/crawford/cat-dataset ó
https://av9.dev/cat-dataset/


## Generar splits
```bash
python -m utils.prepare_splits
```

## Entrenar DCGAN: 

```bash
python -m training.train_dcgan
```
## Outputs esperados

**Pesos:**

- outputs/models/dcgan_G.pth

- outputs/models/dcgan_D.pth

**Samples durante el entrenamiento:**

- outputs/samples/dcgan_epoch{E}_step{S}.png

**Curva de pérdidas:**

- outputs/plots/dcgan_loss_curves.png

## Generar un sample final

Con el modelo ya entrenado:
```bash
python -m training.generate_dcgan_samples
```
