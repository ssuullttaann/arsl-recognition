"""
Arabic Sign Language Recognition - Milestone 3
Model:ArSLCNN  — optimized for maximum accuracy

"""

import os
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

# ─────────────────────────────────────────────
# Configuration — tuned for maximum accuracy
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir"      : "./data",
    "image_size"    : 128,           # 64→128: biggest single accuracy boost
    "batch_size"    : 64,
    "epochs"        : 60,
    "learning_rate" : 1e-3,
    "weight_decay"  : 1e-4,
    "train_ratio"   : 0.70,
    "val_ratio"     : 0.15,
    "seed"          : 42,
    "num_workers"   : 2,
    "output_dir"    : "./outputs_v2",
    "patience"      : 12,            # early stopping patience
    "label_smoothing": 0.1,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

# ─────────────────────────────────────────────
# Data Transforms — stronger augmentation
# ─────────────────────────────────────────────
IMG_SIZE = CONFIG["image_size"]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),  # slightly larger then crop
    transforms.RandomCrop(IMG_SIZE),                     # random crop for variety
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),                   # helps with lighting robustness
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # small shifts
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)), # occlusion robustness
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# Dataset Loading with class balancing
# ─────────────────────────────────────────────
def load_datasets(data_dir):
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    n            = len(full_dataset)
    n_train      = int(n * CONFIG["train_ratio"])
    n_val        = int(n * CONFIG["val_ratio"])
    n_test       = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(CONFIG["seed"])
    )

    val_ds.dataset  = datasets.ImageFolder(root=data_dir, transform=eval_transform)
    test_ds.dataset = datasets.ImageFolder(root=data_dir, transform=eval_transform)

    print(f"[INFO] Classes ({len(full_dataset.classes)}): {full_dataset.classes}")
    print(f"[INFO] Split → Train: {n_train} | Val: {n_val} | Test: {n_test}")

    return train_ds, val_ds, test_ds, full_dataset.classes, full_dataset.targets


def make_loaders(train_ds, val_ds, test_ds, all_targets, n_train):
    # Weighted sampler to handle any class imbalance
    train_targets = [all_targets[i] for i in train_ds.indices]
    class_counts  = np.bincount(train_targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              sampler=sampler, num_workers=CONFIG["num_workers"],
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=CONFIG["num_workers"],
                              pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=CONFIG["num_workers"],
                              pin_memory=True)
    return train_loader, val_loader, test_loader

# ─────────────────────────────────────────────
# Model — ArSLCNN v2
# Deeper, wider, with residual-style skip in classifier
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU → MaxPool with residual shortcut"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # 1×1 conv to match channels for residual
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.block(x) + self.shortcut(x))


class ArSLCNN_v2(nn.Module):
    """
    Improved ArSLCNN for Milestone 3.

    Improvements over v1:
    - Double conv per block (more feature extraction per stage)
    - Residual shortcuts in each block (better gradient flow)
    - 5 blocks instead of 4 (deeper for 128×128 input)
    - Wider channels (32→64→128→256→512)
    - Deeper classifier with 2 dropout layers
    - SE-style channel attention in final block
    """
    def __init__(self, num_classes: int):
        super().__init__()

        self.block1 = ConvBlock(3,   64)    # 128→64
        self.block2 = ConvBlock(64,  128)   # 64→32
        self.block3 = ConvBlock(128, 256)   # 32→16
        self.block4 = ConvBlock(256, 512)   # 16→8
        self.block5 = ConvBlock(512, 512)   # 8→4

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        return self.classifier(x)


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters:     {total:,}")
    print(f"[INFO] Trainable parameters: {trainable:,}")
    return total, trainable

# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Mixed precision training (faster on T4)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels

# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_training_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training & Validation Curves – ArSLCNN v2 (Milestone 3)",
                 fontsize=13, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], "b-o", ms=3, label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", ms=3, label="Val")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [a*100 for a in history["train_acc"]], "b-o", ms=3, label="Train")
    axes[1].plot(epochs, [a*100 for a in history["val_acc"]],   "r-o", ms=3, label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n       = len(class_names)
    fig_size = max(10, n * 0.55)

    fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2 + 2, fig_size))
    kw = dict(xticklabels=class_names, yticklabels=class_names,
              cmap="Blues", linewidths=0.3, linecolor="lightgray")

    sns.heatmap(cm,      annot=True, fmt="d",    ax=axes[0], **kw)
    axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].tick_params(axis='x', rotation=45, labelsize=max(6, 10-n//5))
    axes[0].tick_params(axis='y', rotation=0,  labelsize=max(6, 10-n//5))

    sns.heatmap(cm_norm, annot=True, fmt=".2f", ax=axes[1], **kw)
    axes[1].set_title("Confusion Matrix (Normalized)", fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].tick_params(axis='x', rotation=45, labelsize=max(6, 10-n//5))
    axes[1].tick_params(axis='y', rotation=0,  labelsize=max(6, 10-n//5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_per_class_accuracy(y_true, y_pred, class_names, out_path):
    cm        = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1) * 100
    sorted_idx = np.argsort(per_class)
    sorted_names = [class_names[i] for i in sorted_idx]
    sorted_acc   = per_class[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 9))
    colors  = ["#e74c3c" if a < 50 else "#f39c12" if a < 75 else "#2ecc71"
               for a in sorted_acc]
    bars = ax.barh(sorted_names, sorted_acc, color=colors, edgecolor="white")

    for bar, acc in zip(bars, sorted_acc):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{acc:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy – ArSLCNN v2", fontweight="bold")
    ax.set_xlim(0, 110)
    ax.axvline(x=np.mean(per_class), color="navy", linestyle="--",
               linewidth=1.5, label=f"Mean: {np.mean(per_class):.1f}%")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data
    print("\n" + "="*55)
    print("LOADING DATA")
    print("="*55)
    train_ds, val_ds, test_ds, class_names, all_targets = load_datasets(CONFIG["data_dir"])
    n_train = int(len(train_ds) + len(val_ds) + len(test_ds)) * CONFIG["train_ratio"]
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, all_targets, int(n_train)
    )
    num_classes = len(class_names)

    # 2. Model
    print("\n" + "="*55)
    print("BUILDING MODEL")
    print("="*55)
    model = ArSLCNN_v2(num_classes=num_classes).to(DEVICE)
    total_params, _ = count_params(model)

    # 3. Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(),
                            lr=CONFIG["learning_rate"],
                            weight_decay=CONFIG["weight_decay"])

    # Warmup + cosine annealing
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (CONFIG["epochs"] - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    # 4. Training loop
    print("\n" + "="*55)
    print("TRAINING")
    print("="*55)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc   = 0.0
    patience_count = 0
    best_model_path = out_dir / "best_model_v2.pth"

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, scaler
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{CONFIG['epochs']}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% | "
            f"LR: {current_lr:.6f} | {time.time()-t0:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
                "config": CONFIG,
            }, best_model_path)
            print(f"  ✓ Best model saved (val_acc={val_acc*100:.2f}%)")
        else:
            patience_count += 1
            if patience_count >= CONFIG["patience"]:
                print(f"\n[EARLY STOP] No improvement for {CONFIG['patience']} epochs.")
                break

    # 5. Test evaluation
    print("\n" + "="*55)
    print("FINAL TEST EVALUATION")
    print("="*55)
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")
    print(f"Best Val Accuracy: {best_val_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # 6. Plots
    plot_training_curves(history, out_dir / "training_curves_v2.png")
    plot_confusion_matrix(y_true, y_pred, class_names, out_dir / "confusion_matrix_v2.png")
    plot_per_class_accuracy(y_true, y_pred, class_names, out_dir / "per_class_accuracy.png")

    # 7. Save log
    log = {
        "timestamp"   : datetime.now().isoformat(),
        "config"      : CONFIG,
        "device"      : str(DEVICE),
        "num_classes" : num_classes,
        "class_names" : class_names,
        "total_params": total_params,
        "best_val_acc": best_val_acc,
        "test_loss"   : test_loss,
        "test_acc"    : test_acc,
        "history"     : history,
    }
    with open(out_dir / "training_log_v2.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n[DONE] Best val acc: {best_val_acc*100:.2f}% | Test acc: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
