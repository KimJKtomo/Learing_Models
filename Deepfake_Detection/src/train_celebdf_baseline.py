import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torchvision import transforms


class CelebDFDataset(Dataset):
    """
    CSV 기반 Celeb-DF 이미지 Dataset
    CSV 컬럼:
      - image_path: 프로젝트 루트 기준 상대 경로
      - label: 0 (real), 1 (fake)
    """
    def __init__(self, csv_path: Path, project_root: Path, transform=None):
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform

        if "image_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain 'image_path' and 'label' columns.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = row["image_path"]
        label = int(row["label"])

        img_path = self.project_root / rel_path
        # 안전을 위해 절대 경로로 변환
        img_path = img_path.resolve()

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {img_path} ({e})")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    all_loss = []
    all_labels = []
    all_probs = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
        auc = float("nan")

    return float(np.mean(all_loss)), acc, f1, auc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    all_loss = []
    all_labels = []
    all_probs = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
        auc = float("nan")

    return float(np.mean(all_loss)), acc, f1, auc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Celeb-DF (v2) ConvNeXt Baseline Trainer"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=None,
        help="Deepfake_Detection 프로젝트 루트 경로. "
             "기본값: 이 스크립트 기준 상위 상위 디렉토리",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="train.csv 경로 (project_root 기준 또는 절대경로)",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="val.csv 경로",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/celebdf_convnext_baseline",
        help="결과(모델, 로그) 저장 디렉토리",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="convnextv2_tiny.fcmae_ft_in1k",
        help="timm 모델 이름",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="입력 이미지 크기",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="학습 epoch 수",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="학습률",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader num_workers",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # project_root 설정
    if args.project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    else:
        project_root = Path(args.project_root).resolve()

    outdir = project_root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    tb_dir = outdir / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # CSV 경로 resolve
    train_csv_path = Path(args.train_csv)
    if not train_csv_path.is_absolute():
        train_csv_path = project_root / train_csv_path

    val_csv_path = Path(args.val_csv)
    if not val_csv_path.is_absolute():
        val_csv_path = project_root / val_csv_path

    # Dataset / DataLoader
    train_ds = CelebDFDataset(
        csv_path=train_csv_path,
        project_root=project_root,
        transform=train_tf,
    )
    val_ds = CelebDFDataset(
        csv_path=val_csv_path,
        project_root=project_root,
        transform=val_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Val samples  : {len(val_ds)}")

    # 모델 생성
    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=2,
    )
    model.to(device)

    # Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -1.0
    best_ckpt = outdir / "best_auc.pt"

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

        # best ckpt by Val AUROC
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_auroc": va_auc,
                    "model_name": args.model_name,
                    "img_size": args.img_size,
                },
                best_ckpt,
            )
            print(f"  [SAVE] Best AUROC model updated: {best_ckpt} (AUROC={va_auc:.4f})")

    writer.close()
    print(f"[DONE] Training finished. Best Val AUROC = {best_auc:.4f}")


if __name__ == "__main__":
    main()
