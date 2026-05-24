"""
train_vit.py — Fine-tuning de DeiT-tiny (Vision Transformer) en GTSRB.

Modelo : facebook/deit-tiny-patch16-224
Tiempo : ~25 min por epoca en CPU
Precision esperada: ~90-94% con 2 epocas

Uso:
    python train_vit.py                  # 2 epocas, configuracion por defecto
    python train_vit.py --epochs 3       # mas epocas, mas precision
    python train_vit.py --threads 4      # limitar uso de CPU (default: 4)
    python train_vit.py --no-cache       # reentrenar aunque exista el modelo
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from tqdm import tqdm


# ─── Configuracion ───────────────────────────────────────────────────────────

MODEL_NAME  = "facebook/deit-tiny-patch16-224"   # 5.7M params, rapido en CPU
MODEL_SAVE  = "vit_gtsrb.pth"
DATA_DIR    = "./data"
NUM_CLASSES = 43
IMAGE_SIZE  = 224


# ─── Argumentos ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=2,    help="Epocas de entrenamiento (default: 2 ~ 50 min)")
    p.add_argument("--batch-size", type=int,   default=32,   help="Batch size (default: 32)")
    p.add_argument("--lr",         type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    p.add_argument("--threads",    type=int,   default=4,    help="Hilos de CPU a usar (default: 4)")
    p.add_argument("--no-cache",   action="store_true",      help="Reentrenar aunque exista modelo guardado")
    return p.parse_args()


# ─── Dataset ─────────────────────────────────────────────────────────────────

def get_loaders(batch_size):
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    print("Cargando dataset GTSRB (descarga automatica si no existe)...")
    full  = datasets.GTSRB(DATA_DIR, split="train", download=True, transform=train_tf)
    test  = datasets.GTSRB(DATA_DIR, split="test",  download=True, transform=val_tf)

    val_n   = int(len(full) * 0.1)
    train_n = len(full) - val_n
    train_set, val_set = random_split(full, [train_n, val_n],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test,      batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, len(train_set), len(val_set), len(test)


# ─── Modelo ──────────────────────────────────────────────────────────────────

def build_model():
    print(f"Descargando/cargando modelo: {MODEL_NAME}")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    return model


# ─── Un paso de entrenamiento ────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    start = time.time()

    bar = tqdm(
        loader,
        desc=f"  Epoca {epoch}/{total_epochs} [Train]",
        unit="batch",
        ncols=80,
        colour="green",
    )

    for imgs, labels in bar:
        outputs = model(pixel_values=imgs).logits
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs          = imgs.size(0)
        total_loss += loss.item() * bs
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += bs

        # Actualizar barra en tiempo real
        bar.set_postfix(
            loss=f"{total_loss/total:.3f}",
            acc=f"{100*correct/total:.1f}%",
        )

    elapsed = time.time() - start
    return total_loss / total, 100 * correct / total, elapsed


# ─── Validacion ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, label="Val"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    bar = tqdm(loader, desc=f"  {label:5s}        ", unit="batch", ncols=80, colour="cyan")
    for imgs, labels in bar:
        out  = model(pixel_values=imgs).logits
        loss = criterion(out, labels)
        bs   = imgs.size(0)
        total_loss += loss.item() * bs
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += bs

    return total_loss / total, 100 * correct / total


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Limitar hilos de CPU para no saturarlo
    torch.set_num_threads(args.threads)
    print(f"Usando {args.threads} hilo(s) de CPU.")

    if os.path.exists(MODEL_SAVE) and not args.no_cache:
        print(f"\nModelo ya existe: '{MODEL_SAVE}'")
        print("Usa --no-cache para reentrenar.")
        return

    # Calcular tiempo estimado
    batches_per_epoch = 24000 // args.batch_size  # aprox
    secs_per_batch    = 2.0
    mins_per_epoch    = batches_per_epoch * secs_per_batch / 60
    total_mins        = mins_per_epoch * args.epochs

    print("\n" + "=" * 55)
    print("  GTSRB — DeiT-tiny  Fine-Tuning")
    print("=" * 55)
    print(f"  Modelo  : {MODEL_NAME}")
    print(f"  Epocas  : {args.epochs}")
    print(f"  Batch   : {args.batch_size}")
    print(f"  LR      : {args.lr}")
    print(f"  Hilos   : {args.threads}")
    print(f"  Tiempo estimado : ~{total_mins:.0f} min ({total_mins/60:.1f} h)")
    print("=" * 55)

    # Dataset
    train_ldr, val_ldr, test_ldr, n_train, n_val, n_test = get_loaders(args.batch_size)
    print(f"\n  Train: {n_train:,}  |  Val: {n_val:,}  |  Test: {n_test:,}\n")

    # Modelo, criterio, optimizador
    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'─'*55}")

        tr_loss, tr_acc, elapsed = train_epoch(model, train_ldr, optimizer, criterion, epoch, args.epochs)
        vl_loss, vl_acc          = evaluate(model, val_ldr, criterion, "Val")
        scheduler.step()

        print(f"\n  Resultado epoca {epoch}:")
        print(f"    Train  ->  Loss: {tr_loss:.4f}  |  Acc: {tr_acc:.1f}%  ({elapsed/60:.1f} min)")
        print(f"    Val    ->  Loss: {vl_loss:.4f}  |  Acc: {vl_acc:.1f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"    -> Modelo guardado  (mejor val acc: {best_val_acc:.1f}%)")

    # Test final
    print(f"\n{'='*55}")
    print("  Evaluacion final en TEST")
    print(f"{'='*55}")
    model.load_state_dict(torch.load(MODEL_SAVE, map_location="cpu"))
    ts_loss, ts_acc = evaluate(model, test_ldr, criterion, "Test")
    print(f"\n  Precision en TEST : {ts_acc:.2f}%")
    print(f"  Modelo guardado   : {MODEL_SAVE}")
    print(f"{'='*55}")
    print("\nListo. Ahora puedes correr:  python app.py")


if __name__ == "__main__":
    main()
