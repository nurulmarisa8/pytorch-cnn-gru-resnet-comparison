"""
========================================================
Tugas Individu - Mata Kuliah Deep Learning
========================================================
Nama  : Nurul Marisa Clara Waldi
NIM   : H071241015
Kelas : Deep Learning

Deskripsi:
    Script ini mengimplementasikan dan membandingkan 3 arsitektur
    deep learning pada dataset Fashion-MNIST:
      1. Custom CNN (Convolutional Neural Network)
      2. Bidirectional GRU (Recurrent Neural Network)
      3. Transfer Learning ResNet-18

Output:
    - outputs/model_comparison.png  : Grafik Accuracy & Loss per epoch
    - outputs/cm_Custom_CNN.png     : Confusion Matrix CNN
    - outputs/cm_BiGRU_RNN.png      : Confusion Matrix BiGRU
    - outputs/cm_ResNet18_Transfer.png : Confusion Matrix ResNet-18
    - outputs/metrics_summary.csv   : Tabel ringkasan metrik evaluasi
========================================================
"""

import os
import time
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

# =============================================================
# KONFIGURASI GLOBAL
# =============================================================
SEED        = 42
BATCH_SIZE  = 64
EPOCHS      = 5
LR          = 0.001
N_TRAIN     = 15000
N_TEST      = 3000
DATA_DIR    = "./data"
OUTPUT_DIR  = "./outputs"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# DETEKSI PERANGKAT (MPS / CUDA / CPU)
# =============================================================
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        label  = "Apple Silicon MPS"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        label  = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        label  = "CPU"
    print(f"[INFO] Menggunakan perangkat: {label}")
    return device


DEVICE = get_device()

# =============================================================
# DATASET & DATALOADER
# =============================================================
def get_dataloaders():
    """Memuat Fashion-MNIST dan membuat subset 15.000 / 3.000."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    full_train = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    full_test = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    # Subset deterministik
    rng        = np.random.RandomState(SEED)
    train_idx  = rng.choice(len(full_train), N_TRAIN, replace=False)
    test_idx   = rng.choice(len(full_test),  N_TEST,  replace=False)

    train_set = Subset(full_train, train_idx)
    test_set  = Subset(full_test,  test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"[INFO] Train samples: {len(train_set)} | Test samples: {len(test_set)}")
    return train_loader, test_loader


# =============================================================
# ARSITEKTUR MODEL
# =============================================================

# ---- 1. Custom CNN ----
class CustomCNN(nn.Module):
    """
    Custom Convolutional Neural Network untuk klasifikasi Fashion-MNIST.

    Arsitektur:
        Block 1: Conv2d(1,32) -> BatchNorm -> ReLU -> MaxPool
        Block 2: Conv2d(32,64) -> BatchNorm -> ReLU -> MaxPool
        FC     : Flatten -> Linear(128) -> ReLU -> Dropout(0.3) -> Linear(10)
    """
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 28x28 -> 14x14
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


# ---- 2. Bidirectional GRU ----
class ImageBiGRU(nn.Module):
    """
    Bidirectional GRU untuk klasifikasi gambar Fashion-MNIST.

    Gambar 28×28 diperlakukan sebagai sekuens 28 time steps
    dengan vektor fitur ukuran 28 per baris.

    Arsitektur:
        GRU(input=28, hidden=128, num_layers=2, bidirectional=True, dropout=0.2)
        -> FC(256 -> 10)
    """
    def __init__(self, input_size=28, hidden_size=128, num_layers=2, num_classes=10):
        super(ImageBiGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 karena bidirectional

    def forward(self, x):
        # x: (batch, 1, 28, 28) -> (batch, 28, 28)
        x = x.squeeze(1)
        out, _ = self.gru(x)
        # Ambil output time step terakhir
        out = self.fc(out[:, -1, :])
        return out


# ---- 3. Transfer Learning ResNet-18 ----
class ResNet18_Transfer(nn.Module):
    """
    Transfer Learning berbasis ResNet-18 (pretrained ImageNet).

    Modifikasi:
        - conv1  : diubah menjadi 1-channel (grayscale)
        - fc     : diganti Dropout(0.3) + Linear(512 -> 10)
    """
    def __init__(self, num_classes=10):
        super(ResNet18_Transfer, self).__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Sesuaikan conv1 untuk input 1-channel
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Ganti fully connected layer
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


# =============================================================
# FUNGSI TRAINING & EVALUASI
# =============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Melatih model selama satu epoch, mengembalikan (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluasi model, mengembalikan (loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += images.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / total,
        100.0 * correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


def run_training(model, model_name, train_loader, test_loader, device):
    """
    Menjalankan training + evaluasi untuk satu model selama EPOCHS epoch.

    Returns:
        history (dict): loss/acc per epoch
        final_preds, final_labels (np.ndarray): prediksi epoch terakhir
        train_time (float): total waktu training (detik)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    print(f"\n{'='*60}")
    print(f"  TRAINING: {model_name}")
    print(f"{'='*60}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
    print("-" * 55)

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc     = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, preds, labels = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"{epoch:>6} {tr_loss:>12.4f} {tr_acc:>9.2f}% {val_loss:>10.4f} {val_acc:>8.2f}%")

    train_time = time.time() - start_time

    # Classification Report
    print(f"\n[Classification Report - {model_name}]")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    return history, preds, labels, train_time


# =============================================================
# VISUALISASI
# =============================================================

def plot_model_comparison(histories, model_names):
    """Menyimpan subplot Accuracy & Loss ketiga model ke satu file."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Perbandingan Model - Fashion-MNIST", fontsize=14, fontweight="bold")

    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    epochs_range = range(1, EPOCHS + 1)

    # Validation Accuracy
    ax = axes[0]
    for hist, name, color in zip(histories, model_names, colors):
        ax.plot(epochs_range, hist["val_acc"], marker="o", label=name, color=color, linewidth=2)
    ax.set_title("Validation Accuracy (%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation Loss
    ax = axes[1]
    for hist, name, color in zip(histories, model_names, colors):
        ax.plot(epochs_range, hist["val_loss"], marker="s", label=name, color=color, linewidth=2)
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Disimpan: {out_path}")


def plot_confusion_matrix(labels, preds, model_name, file_key):
    """Menyimpan heatmap confusion matrix."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Prediksi", fontsize=11)
    ax.set_ylabel("Label Aktual", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"cm_{file_key}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Disimpan: {out_path}")


def save_metrics_csv(results):
    """Menyimpan tabel ringkasan ke CSV."""
    out_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    fieldnames = ["Model", "Train Acc (%)", "Val Acc (%)", "Val Loss", "Macro F1-Score", "Waktu Training (s)"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Disimpan: {out_path}")


# =============================================================
# MAIN
# =============================================================

def main():
    print("\n" + "=" * 60)
    print("  TUGAS DEEP LEARNING - FASHION-MNIST")
    print("  Nama: Nurul Marisa Clara Waldi | NIM: H071241015")
    print("=" * 60)
    print(f"  Konfigurasi: Epochs={EPOCHS}, Batch={BATCH_SIZE}, LR={LR}")
    print(f"  Train samples={N_TRAIN}, Test samples={N_TEST}")
    print("=" * 60)

    train_loader, test_loader = get_dataloaders()

    # Daftar model yang akan diuji
    models_config = [
        (CustomCNN(),           "Custom CNN",          "Custom_CNN"),
        (ImageBiGRU(),         "BiGRU RNN",            "BiGRU_RNN"),
        (ResNet18_Transfer(),  "ResNet18 Transfer",    "ResNet18_Transfer"),
    ]

    histories   = []
    model_names = []
    csv_rows    = []

    for model, display_name, file_key in models_config:
        history, preds, labels, train_time = run_training(
            model, display_name, train_loader, test_loader, DEVICE
        )

        f1 = f1_score(labels, preds, average="macro")

        histories.append(history)
        model_names.append(display_name)

        csv_rows.append({
            "Model":              display_name,
            "Train Acc (%)":      f"{history['train_acc'][-1]:.2f}",
            "Val Acc (%)":        f"{history['val_acc'][-1]:.2f}",
            "Val Loss":           f"{history['val_loss'][-1]:.4f}",
            "Macro F1-Score":     f"{f1:.4f}",
            "Waktu Training (s)": f"{train_time:.1f}",
        })

        plot_confusion_matrix(labels, preds, display_name, file_key)

    # Grafik perbandingan semua model
    plot_model_comparison(histories, model_names)

    # Simpan CSV
    save_metrics_csv(csv_rows)

    # Ringkasan final
    print("\n" + "=" * 60)
    print("  RINGKASAN HASIL EVALUASI")
    print("=" * 60)
    df = pd.DataFrame(csv_rows)
    print(df.to_string(index=False))
    print("=" * 60)
    print("\n[SELESAI] Semua output berhasil disimpan di folder outputs/")


if __name__ == "__main__":
    main()
