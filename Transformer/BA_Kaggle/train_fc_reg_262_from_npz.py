
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Same architecture as fc_prediction_whole262.py (input dim 262)
class FCRegressor(nn.Module):
    def __init__(self, in_features: int, gender: str):
        super().__init__()
        if gender == 'f':
            feature_list = [384]
            dropout_list = [0.04008342975594972]
        else:
            feature_list = [284]
            dropout_list = [0.08293688710677454]

        layers = []
        for feat, drop in zip(feature_list, dropout_list):
            layers += [nn.Linear(in_features, feat), nn.BatchNorm1d(feat), nn.ReLU(), nn.Dropout(drop)]
            in_features = feat
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(self.layers(x)).squeeze(1)


class NpzDataset(Dataset):
    def __init__(self, npz_path: str, gender: str):
        d = np.load(npz_path, allow_pickle=True)
        self.X = d["X"].astype(np.float32)      # (N,262)
        self.y = d["y"].astype(np.float32)      # (N,)
        self.g = d["gender"].astype(str)        # (N,)
        self.keep = np.where(self.g == gender)[0]

    def __len__(self):
        return int(len(self.keep))

    def __getitem__(self, idx):
        i = int(self.keep[idx])
        return self.X[i], self.y[i]


def train_one_gender(npz_train, npz_val, out_dir, gender, device,
                     lr, epochs, batch_size, weight_decay, patience):
    os.makedirs(out_dir, exist_ok=True)

    ds_tr = NpzDataset(npz_train, gender)
    ds_va = NpzDataset(npz_val, gender)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = FCRegressor(262, gender).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # MAE directly
    loss_fn = nn.L1Loss()

    best = 1e18
    bad = 0
    best_path = os.path.join(out_dir, f"ba_reg_{gender}_best.pt")

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for X, y in dl_tr:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(X)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * X.size(0)
        tr_loss /= max(1, len(ds_tr))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X, y in dl_va:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(X)
                loss = loss_fn(pred, y)
                va_loss += float(loss.item()) * X.size(0)
        va_loss /= max(1, len(ds_va))

        print(f"[{gender}] ep {ep:03d}/{epochs} | train_MAE={tr_loss:.4f} | val_MAE={va_loss:.4f}")

        if va_loss < best - 1e-6:
            best = va_loss
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if patience > 0 and bad >= patience:
                print(f"[{gender}] early stop (best val_MAE={best:.4f})")
                break

    return best_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True, type=str, help="npz with X(262), y(months), gender('m'/'f')")
    ap.add_argument("--val_npz", required=True, type=str)
    ap.add_argument("--out_dir", default="ckpts/fc_ckpt_v2/regression", type=str)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--epochs", default=50, type=int)
    ap.add_argument("--batch_size", default=256, type=int)
    ap.add_argument("--weight_decay", default=1e-4, type=float)
    ap.add_argument("--patience", default=10, type=int)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    for gender, sub in [("f", "female"), ("m", "male")]:
        out = os.path.join(args.out_dir, sub)
        p = train_one_gender(args.train_npz, args.val_npz, out, gender, device,
                             args.lr, args.epochs, args.batch_size, args.weight_decay, args.patience)
        print("[SAVED]", p)

if __name__ == "__main__":
    main()
