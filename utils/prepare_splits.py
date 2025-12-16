# utils/prepare_splits.py
import random
from pathlib import Path

def main():
    random_seed = 42
    random.seed(random_seed)

    # carpeta donde quedaron las imágenes
    raw_root = Path("data/raw")
    images_root = raw_root

    exts = {".jpg", ".jpeg", ".png"}
    all_imgs = [p for p in images_root.rglob("*") if p.suffix.lower() in exts]

    print(f"Encontradas {len(all_imgs)} imágenes.")

    if len(all_imgs) == 0:
        print("No se encontraron imágenes. Revisa el path en images_root.")
        return

    all_imgs = sorted(all_imgs)
    random.shuffle(all_imgs)

    n_total = len(all_imgs)
    n_train = int(0.8 * n_total)

    train_paths = all_imgs[:n_train]
    val_paths = all_imgs[n_train:]

    Path("data/splits").mkdir(parents=True, exist_ok=True)

    with open("data/splits/train.txt", "w", encoding="utf-8") as f:
        for p in train_paths:
            f.write(str(p.as_posix()) + "\n")

    with open("data/splits/val.txt", "w", encoding="utf-8") as f:
        for p in val_paths:
            f.write(str(p.as_posix()) + "\n")

    print(f"Train: {len(train_paths)} imágenes.")
    print(f"Val:   {len(val_paths)} imágenes.")

if __name__ == "__main__":
    main()
