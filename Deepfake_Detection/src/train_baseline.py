import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import timm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# =========================
# Dataset (placeholder)
# =========================
class FFDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None):
        # csv: image_path, label (0/1)
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["label"])

        from PIL import Image
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# =========================
# Train / Val Loop
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    all_loss = []
    all_labels = []
    all_probs = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)  # (B, 2)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        all_loss.append(loss.item())
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    return np.mean(all_loss), acc, f1, auc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    all_loss = []
    all_labels = []
    all_probs = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        all_loss.append(loss.item())
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    return np.mean(all_loss), acc, f1, auc


# =========================
# Main
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./outputs_deepfake_baseline")
    parser.add_argument("--model_name", type=str, default="convnextv2_tiny.fcmae_ft_in1k")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.outdir, "tb"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    from torchvision import transforms
    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # dataset / loader
    train_ds = FFDataset(args.train_csv, transform=train_tf)
    val_ds = FFDataset(args.val_csv, transform=val_tf)

    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)

    # model
    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=2
    )
    model.to(device)

    # loss / optim
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -1.0
    best_ckpt = os.path.join(args.outdir, "best_auc.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, tr_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        va_loss, va_acc, va_f1, va_auc = validate(
            model, val_loader, criterion, device
        )

        print(
            f"[Ep {epoch:03d}] "
            f"Train L={tr_loss:.4f} A={tr_acc:.4f} F1={tr_f1:.4f} AUROC={tr_auc:.4f} | "
            f"Val L={va_loss:.4f} A={va_acc:.4f} F1={va_f1:.4f} AUROC={va_auc:.4f}"
        )

        # TensorBoard logging
        writer.add_scalar("Train/Loss", tr_loss, epoch)
        writer.add_scalar("Train/Acc", tr_acc, epoch)
        writer.add_scalar("Train/F1", tr_f1, epoch)
        writer.add_scalar("Train/AUROC", tr_auc, epoch)

        writer.add_scalar("Val/Loss", va_loss, epoch)
        writer.add_scalar("Val/Acc", va_acc, epoch)
        writer.add_scalar("Val/F1", va_f1, epoch)
        writer.add_scalar("Val/AUROC", va_auc, epoch)

        # best ckpt by AUROC
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_auroc": va_auc,
                },
                best_ckpt,
            )
            print(f"  ✅ Saved best AUROC model: {best_ckpt} (AUROC={va_auc:.4f})")

    writer.close()
    print(f"[Done] Best Val AUROC = {best_auc:.4f}")


if __name__ == "__main__":
    main()
