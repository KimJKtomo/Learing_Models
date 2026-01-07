
"""
train_Wrist_convnext2_mil_0102.py

ConvNeXtV2-Large + MIL (mixed pos bag + hard-negative neg bag) training script
- PyCharm Run friendly: no CLI args required (all configs in CFG below)
- Label mapping: label_txt_path = YOLO_LABELS_DIR / (filestem(image_path) + ".txt")
- Pre-scan report: class distribution, pos-miss, neg-FP, missing labels, duplicate filestems
- MIL aggregation: top-k mean pooling (reduces "one patch dominates" collapse)
- OOM controls: AMP + gradient accumulation + (optional) gradient checkpointing
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import random
import warnings
import re
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image, ImageFilter

try:
    import timm
except Exception as e:
    timm = None
    raise RuntimeError("timm is required. Please install timm in your environment.") from e


# =========================
# Config (edit here)
# =========================
@dataclass
class CFG:
    # CSVs (AP / LAT)
    TRAIN_CSV_AP: str = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/WristFX_0730/classification/age_train_tmp_AP.csv"
    VAL_CSV_AP:   str = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/WristFX_0730/classification/age_val_tmp_AP.csv"

    TRAIN_CSV_LAT: str = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/WristFX_0730/classification/age_train_tmp_Lat.csv"
    VAL_CSV_LAT:   str = "/mnt/data/KimJG/SwinT/mediaiOA_swinT_cls-main_original/20250623_Single/WristFX_0730/classification/age_val_tmp_Lat.csv"

    # Labels
    YOLO_LABELS_DIR: str = "/mnt/data/KimJG/ELBOW_test/Kaggle_dataset/folder_structure/yolov5/labels"
    FRACTURE_CLASS_ID: int = 3

    # YOLO meta: text class id (to avoid sampling patches dominated by overlays)
    TEXT_CLASS_ID: int = 8
    # Text handling
    # - v5: avoid sampling patch centers inside text bbox
    # - v6: additionally mask text regions in the input image to reduce shortcut learning
    #   modes: "none" | "zero" | "blur"
    TEXT_MASK_MODE: str = "blur"
    TEXT_MASK_APPLY_TO: str = "all"   # "train" or "all"
    TEXT_MASK_BLUR_RADIUS: float = 5.0
    # Sampling optimization
    # If True, when "avoiding" text, use a capped rejection sampling and fall back safely.
    ALLOWED_SAMPLING: bool = True
    MAX_RESAMPLE_TRIES: int = 50

    # Stability
    GRAD_CLIP_NORM: float = 1.0
    SKIP_NONFINITE_LOSS: bool = True
    AMP_INIT_SCALE: float = 1024.0
    AMP_GROWTH_INTERVAL: int = 2000


    # Age-group training (8 models = 2 views × 4 age groups)
    USE_AGE_GROUPS: bool = True
    AGE_COL_CANDIDATES = ["age", "Age", "AGE", "patient_age", "age_years", "AgeYears", "age_yr"]
    DROP_AGE_NAN: bool = True  # if True, rows with missing age are excluded (reported in scan)

    # Output
    SAVE_DIR: str = "./out_mil_0102_v3"

    # Model
    TIMM_BACKBONE: str = "convnextv2_large.fcmae_ft_in22k_in1k"
    IMG_SIZE: int = 384

    # MIL bag
    K_INSTANCES: int = 8
    POS_RATIO: float = 0.30          # fraction of instances drawn from fracture bbox area for positive bags
    HARDNEG_RATIO: float = 0.50      # fraction of instances drawn from YOLO FP bbox for negative bags if available
    HARDNEG_OTHER_CLASSES: tuple[int, ...] = (0,1,2,4,5,6,7)  # use these as hardneg candidates (exclude 3=fracture, 8=text)
    AVOID_CLASSES: tuple[int, ...] = (8,)                         # avoid sampling patch centers on these classes
    TOPK_POOL_K: int = 3             # top-k mean pooling
    BG_REJECT_IOU_TH: float = 0.05   # reject background patches that overlap too much with fracture bbox (approx)

    # Training
    SEED: int = 1229
    EPOCHS: int = 30
    LR: float = 1e-5
    WEIGHT_DECAY: float = 1e-4
    BATCH_SIZE: int = 2
    ACCUM_STEPS: int = 4
    NUM_WORKERS: int = 4

    # Loss controls
    USE_FOCAL: bool = False
    FOCAL_GAMMA: float = 2.0
    NEG_SUPPRESS: bool = True        # add penalty for high instance probs in negative bags
    NEG_MARGIN: float = 0.20
    NEG_LAMBDA: float = 2.0

    # Class imbalance handling (DDP-safe)
    USE_POS_WEIGHT: bool = True
    POS_WEIGHT_CLAMP: tuple[float, float] = (0.25, 4.0)  # clamp neg/pos into this range

    # Decision threshold tuning on validation (DDP-safe, rank0 computes, broadcast to all ranks)
    TUNE_THRESHOLD: bool = True

    # Threshold search range clamp (prevents extreme thresholds)
    THRESH_MIN: float = 0.20          # search lower bound in [0,1]
    THRESH_MAX: float = 0.80          # search upper bound in [0,1]
    THRESH_GRID_SIZE: int = 401       # number of thresholds in [THRESH_MIN, THRESH_MAX]

    # Threshold mode
    # - "youden": maximize (recall + specificity - 1)
    # - "f1": maximize F1
    # - "spec_at_rec": maximize specificity subject to recall >= THRESH_REC_TARGET
    # - "rec_at_spec": maximize recall subject to specificity >= THRESH_SPEC_TARGET
    THRESH_MODE: str = "spec_at_rec"


    # Auto threshold policy (view/age aware)
    AUTO_THR_POLICY: bool = True
    # Default: age0/1/2 -> spec_at_rec with rec_target=0.90, age3 -> rec_at_spec with spec_target=0.95
    AUTO_THR_REC_TARGET: float = 0.90
    AUTO_THR_SPEC_TARGET: float = 0.95
    AUTO_THR_RANGE_POSHEAVY: tuple = (0.10, 0.90)
    AUTO_THR_RANGE_AGE3: tuple = (0.05, 0.95)
    # Constraints / targets
    THRESH_REC_TARGET: float = 0.95   # used in spec_at_rec
    THRESH_SPEC_TARGET: float = 0.95  # used in rec_at_spec

    # Optional additional hard constraints applied in all modes
    THRESH_MIN_SPEC: float = 0.00
    THRESH_MIN_REC: float = 0.00

    # Memory / speed
    AMP: bool = True
    # Force ON: reduces activation memory at cost of compute (recommended when keeping backbone large)
    USE_CHECKPOINTING: bool = True
    CHANNELS_LAST: bool = True
    TF32: bool = True

    # ===== OOM Mitigation (keep batch / K_INSTANCES) =====
    # Strategy: freeze backbone early -> unfreeze last stage later (reduces backward graph + optimizer states)
    FREEZE_BACKBONE: bool = True
    FREEZE_EPOCHS: int = 2           # epochs with full backbone frozen (head-only training)
    UNFREEZE_STAGE4_EPOCH: int = 3   # epoch to unfreeze last stage (stage4). 1-indexed.
    UNFREEZE_STAGE3_EPOCH: int = 999 # optional: unfreeze stage3 later if needed

    # Optimizer memory reduction
    # If True, build optimizer only on trainable params (critical for freeze schedule)
    OPTIM_TRAINABLE_ONLY: bool = True
    # Optional: use bitsandbytes 8-bit AdamW if installed (saves optimizer state VRAM)
    USE_8BIT_ADAMW: bool = False

    # Logging
    PRINT_EVERY: int = 50
    SAVE_BEST_BY: str = "val_f1"     # "val_f1" or "val_auc" etc.

    # MLFLOW
    MLFLOW: bool = True
    MLFLOW_URI: str = "http://127.0.0.1:5000"   # set to your tracking server, or leave as-is
    MLFLOW_EXP: str = "Wrist_MIL_ConvNeXtV2_1229"


CFG = CFG()


# =========================
# Age group
# =========================
def AGE_GROUP_FN(age_value) -> int:
    """Map age (years) -> age_group {0,1,2,3} using the project's rule."""
    try:
        a = float(age_value)
    except Exception:
        return -1
    if a != a:  # NaN
        return -1
    if a < 5:
        return 0
    elif a < 10:
        return 1
    elif a < 15:
        return 2
    else:
        return 3


def detect_age_col(df):
    """Detect age column name from candidates; returns None if not found."""
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    # prefer exact/known candidates
    for cand in getattr(CFG, "AGE_COL_CANDIDATES", []):
        if cand in cols:
            return cand
        lc = str(cand).lower()
        if lc in lower_map:
            return lower_map[lc]
    # fallback: any column containing 'age'
    for c in cols:
        if "age" in c.lower():
            return c
    return None


def add_age_group_column(df, age_col):
    df = df.copy()
    df["age_group"] = df[age_col].apply(AGE_GROUP_FN)
    return df


# =========================
# Utils
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Distributed (DDP) helpers
# =========================
def is_distributed() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and int(os.environ.get("WORLD_SIZE", "1")) > 1

def get_dist_info():
    if not is_distributed():
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local_rank

def ddp_setup():
    """Initialize torch.distributed if launched with torchrun."""
    if not is_distributed():
        return
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")
    if torch.distributed.is_initialized():
        return
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    # Ensure each process uses its own GPU
    _, _, local_rank = get_dist_info()
    torch.cuda.set_device(local_rank)

def ddp_cleanup():
    if is_distributed() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

def is_main_process() -> bool:
    rank, _, _ = get_dist_info()
    return rank == 0

def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


def try_import_mlflow():
    if not CFG.MLFLOW:
        return None
    try:
        import mlflow
        mlflow.set_tracking_uri(CFG.MLFLOW_URI)
        mlflow.set_experiment(CFG.MLFLOW_EXP)
        return mlflow
    except Exception as e:
        print(f"[WARN] MLflow enabled but import/config failed: {e}")
        return None


def find_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


# =========================
# YOLO label parsing
# =========================
def parse_yolo_txt(txt_path: Path) -> list[tuple[int, float, float, float, float]]:
    """
    Returns list of (cls, xc, yc, w, h) in normalized coords
    """
    boxes = []
    try:
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                # clamp to sane range
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)
                boxes.append((cls, xc, yc, w, h))
    except Exception:
        return []
    return boxes


def approx_iou_center(box_a, box_b) -> float:
    """
    Approximate IoU using normalized boxes defined by (xc,yc,w,h)
    """
    _, ax, ay, aw, ah = box_a
    _, bx, by, bw, bh = box_b
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-12
    return float(inter / union)


# =========================
# Scan report
# =========================

def safe_int(v, default: int = 0) -> int:
    """Robust int conversion for CSV values (handles NaN/None/empty/float strings)."""
    try:
        if v is None:
            return default
        # pandas may give numpy.nan
        try:
            import math
            if isinstance(v, float) and math.isnan(v):
                return default
        except Exception:
            pass
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            return default
        # allow float-like numbers such as '1.0'
        return int(float(s))
    except Exception:
        return default


def scan_dataset(csv_path: str, labels_dir: Path, out_dir: Path, tag: str) -> dict:
    df = pd.read_csv(csv_path)
    img_col = find_first_existing_col(df, ["image_path", "img_path", "path", "filepath", "file_path"])
    gt_col  = find_first_existing_col(df, ["fracture_visible", "gt", "label", "target", "y"])
    if img_col is None or gt_col is None:
        raise ValueError(f"[SCAN] Missing required columns in {csv_path}. Found columns: {list(df.columns)}")

    total = len(df)
    class_counter = Counter()

    image_exists = 0
    label_exists = 0
    empty_label = 0

    pos_total = 0
    pos_has_fracture_bbox = 0
    pos_fracture_bbox_counts = []

    neg_total = 0
    neg_fp = 0
    neg_fp_bbox_counts = []

    neg_has_any = 0
    neg_has_text = 0
    neg_has_other = 0
    neg_other_bbox_counts = []
    neg_text_bbox_counts = []

    # filestem duplication check
    stem_to_paths = defaultdict(set)

    for _, row in df.iterrows():
        img_path = Path(str(row[img_col]))
        stem = img_path.stem
        stem_to_paths[stem].add(str(img_path))

        if img_path.exists():
            image_exists += 1

        lbl_path = labels_dir / f"{stem}.txt"
        if lbl_path.exists():
            label_exists += 1
            boxes = parse_yolo_txt(lbl_path)
            if len(boxes) == 0:
                empty_label += 1
            for (c, *_rest) in boxes:
                class_counter[c] += 1

            frac_boxes = [b for b in boxes if b[0] == CFG.FRACTURE_CLASS_ID]
            text_boxes = [b for b in boxes if b[0] == CFG.TEXT_CLASS_ID]
            other_boxes = [b for b in boxes if (b[0] != CFG.FRACTURE_CLASS_ID and b[0] != CFG.TEXT_CLASS_ID)]
        else:
            boxes = []
            frac_boxes = []
            text_boxes = []
            other_boxes = []

        gt = safe_int(row.get(gt_col, 0))
        if gt == 1:
            pos_total += 1
            if len(frac_boxes) > 0:
                pos_has_fracture_bbox += 1
                pos_fracture_bbox_counts.append(len(frac_boxes))
            else:
                pos_fracture_bbox_counts.append(0)
        else:
            neg_total += 1
            if len(frac_boxes) > 0:
                neg_fp += 1
                neg_fp_bbox_counts.append(len(frac_boxes))
            else:
                neg_fp_bbox_counts.append(0)

            if len(boxes) > 0:
                neg_has_any += 1
            if len(text_boxes) > 0:
                neg_has_text += 1
                neg_text_bbox_counts.append(len(text_boxes))
            if len(other_boxes) > 0:
                neg_has_other += 1
                neg_other_bbox_counts.append(len(other_boxes))

    dup_stems = {s: list(p) for s, p in stem_to_paths.items() if len(p) > 1}

    report = {
        "tag": tag,
        "csv_path": csv_path,
        "n_rows": total,
        "image_exists_rate": image_exists / max(1, total),
        "label_exists_rate": label_exists / max(1, total),
        "empty_label_rate": empty_label / max(1, label_exists),
        "class_counts_all": dict(sorted(class_counter.items(), key=lambda x: x[0])),
        "fracture_class_id": CFG.FRACTURE_CLASS_ID,
        "pos": {
            "n": pos_total,
            "has_fracture_bbox_rate": pos_has_fracture_bbox / max(1, pos_total),
            "fracture_bbox_count_mean": float(np.mean(pos_fracture_bbox_counts)) if pos_total else 0.0,
            "fracture_bbox_count_p95": float(np.percentile(pos_fracture_bbox_counts, 95)) if pos_total else 0.0,
            "miss_rate": 1.0 - (pos_has_fracture_bbox / max(1, pos_total)),
        },
        "neg": {
            "n": neg_total,
            "yolo_fp_rate": neg_fp / max(1, neg_total),
            "fp_bbox_count_mean": float(np.mean(neg_fp_bbox_counts)) if neg_total else 0.0,
            "fp_bbox_count_p95": float(np.percentile(neg_fp_bbox_counts, 95)) if neg_total else 0.0,
            "has_any_box_rate": neg_has_any / max(1, neg_total),
            "has_text_box_rate": neg_has_text / max(1, neg_total),
            "has_other_box_rate": neg_has_other / max(1, neg_total),
            "text_bbox_count_mean": float(np.mean(neg_text_bbox_counts)) if len(neg_text_bbox_counts) else 0.0,
            "other_bbox_count_mean": float(np.mean(neg_other_bbox_counts)) if len(neg_other_bbox_counts) else 0.0,
        },
        "dup_filstems": {
            "n_dup_stems": len(dup_stems),
            "examples": {k: v[:3] for k, v in list(dup_stems.items())[:20]},
        },
        "notes": [
            "label path is resolved as YOLO_LABELS_DIR/(filestem(image_path)+.txt)",
            "class_counts_all includes all class ids in txt; training uses only fracture_class_id",
        ],
    }

    ensure_dir(out_dir)
    (out_dir / f"scan_report_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # human-readable txt
    lines = []
    lines.append(f"[SCAN] {tag}")
    lines.append(f"  csv: {csv_path}")
    lines.append(f"  rows: {total}")
    lines.append(f"  image_exists_rate: {report['image_exists_rate']:.4f}")
    lines.append(f"  label_exists_rate: {report['label_exists_rate']:.4f}")
    lines.append(f"  empty_label_rate: {report['empty_label_rate']:.4f}")
    lines.append(f"  fracture_class_id: {CFG.FRACTURE_CLASS_ID}")
    lines.append(f"  pos_n: {pos_total} | pos_has_fracture_bbox_rate: {report['pos']['has_fracture_bbox_rate']:.4f} | pos_miss_rate: {report['pos']['miss_rate']:.4f}")
    lines.append(f"  neg_n: {neg_total} | neg_yolo_fp_rate: {report['neg']['yolo_fp_rate']:.4f} | neg_has_other_box_rate: {report['neg']['has_other_box_rate']:.4f} | neg_has_text_box_rate: {report['neg']['has_text_box_rate']:.4f}")
    lines.append(f"  dup_filstems: {len(dup_stems)}")
    lines.append(f"  class_counts_all (top 20): {list(report['class_counts_all'].items())[:20]}")
    (out_dir / f"scan_report_{tag}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # print summary
    print("\n".join(lines))
    if report["neg"]["yolo_fp_rate"] > 0.05:
        print(f"[WARN] High neg_yolo_fp_rate={report['neg']['yolo_fp_rate']:.4f}. Hard-negative sampling is important.")
    if report["pos"]["miss_rate"] > 0.20:
        print(f"[WARN] High pos_miss_rate={report['pos']['miss_rate']:.4f}. Many positive images have no class-{CFG.FRACTURE_CLASS_ID} bbox; pos sampling may fallback often.")

    return report


# =========================
# Dataset (MIL)
# =========================
# =========================
# Dataset (MIL)
# =========================
class WristMILDataset(Dataset):
    def __init__(self, csv_path: str, labels_dir: str, img_size: int, k_instances: int,
                 pos_ratio: float, hardneg_ratio: float, fracture_cls: int,
                 split_tag: str = "train"):
        self.df = pd.read_csv(csv_path)
        self.img_col = find_first_existing_col(self.df, ["image_path", "img_path", "path", "filepath", "file_path"])
        self.gt_col  = find_first_existing_col(self.df, ["fracture_visible", "gt", "label", "target", "y"])
        if self.img_col is None or self.gt_col is None:
            raise ValueError(f"[DATASET] Missing required columns in {csv_path}. Found columns: {list(self.df.columns)}")

        self.labels_dir = Path(labels_dir)
        self.img_size = int(img_size)
        self.k = int(k_instances)
        self.pos_ratio = float(pos_ratio)
        self.hardneg_ratio = float(hardneg_ratio)
        self.fracture_cls = int(fracture_cls)
        self.split_tag = str(split_tag)

        # minimal augmentation
        self.do_aug = (self.split_tag == "train")

    def __len__(self):
        return len(self.df)

    def _load_rgb(self, path: Path) -> Image.Image:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _apply_text_mask(self, img: Image.Image,
                         avoid_boxes: list[tuple[int, float, float, float, float]]) -> Image.Image:
        """Mask text/overlay regions based on YOLO normalized boxes."""
        mode = str(CFG.TEXT_MASK_MODE).lower().strip()
        if mode == "none" or not avoid_boxes:
            return img

        w, h = img.size

        if mode == "zero":
            arr = np.array(img)
            for cls_id, xc, yc, bw, bh in avoid_boxes:
                x1 = int((xc - bw / 2) * w)
                x2 = int((xc + bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                y2 = int((yc + bh / 2) * h)
                x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
                y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
                if x2 > x1 and y2 > y1:
                    arr[y1:y2, x1:x2, :] = 0
            return Image.fromarray(arr)

        if mode == "blur":
            out = img.copy()
            radius = float(getattr(CFG, "TEXT_MASK_BLUR_RADIUS", 3.0))
            for cls_id, xc, yc, bw, bh in avoid_boxes:
                x1 = int((xc - bw / 2) * w)
                x2 = int((xc + bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                y2 = int((yc + bh / 2) * h)
                x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
                y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
                if x2 > x1 and y2 > y1:
                    region = out.crop((x1, y1, x2, y2))
                    region = region.filter(ImageFilter.GaussianBlur(radius=radius))
                    out.paste(region, (x1, y1))
            return out

        return img

    def _random_center(self) -> tuple[float, float]:
        return random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)

    def _avoid_center(self, cx: float, cy: float,
                      avoid_boxes: list[tuple[int, float, float, float, float]],
                      th: float = 0.10) -> bool:
        """True if a patch centered at (cx,cy) overlaps avoid boxes above threshold (approx IoU)."""
        if not avoid_boxes:
            return False
        probe = (-1, cx, cy, 0.30, 0.30)
        for ab in avoid_boxes:
            if approx_iou_center(probe, ab) > th:
                return True
        return False

    def _random_center_avoiding(self, avoid_boxes: list, max_tries: int | None = None) -> tuple[float, float]:
        if max_tries is None:
            max_tries = int(getattr(CFG, "MAX_RESAMPLE_TRIES", 80))
        for _ in range(max_tries):
            cx, cy = self._random_center()
            if not self._avoid_center(cx, cy, avoid_boxes):
                return cx, cy
        return self._random_center()

    def _bbox_to_center(self, b) -> tuple[float, float]:
        _, xc, yc, w, h = b
        jx = (random.random() - 0.5) * w * 0.5
        jy = (random.random() - 0.5) * h * 0.5
        cx = min(max(xc + jx, 0.0), 1.0)
        cy = min(max(yc + jy, 0.0), 1.0)
        return cx, cy

    def _sample_bg_center(self, frac_boxes: list, avoid_boxes: list, max_tries: int = 80) -> tuple[float, float]:
        """Sample background patch center avoiding fracture boxes and avoid_boxes."""
        for _ in range(max_tries):
            cx, cy = self._random_center()
            bg_box = (-1, cx, cy, 0.30, 0.30)
            ok = True
            for fb in frac_boxes:
                if approx_iou_center(bg_box, fb) > CFG.BG_REJECT_IOU_TH:
                    ok = False
                    break
            if ok and avoid_boxes:
                for ab in avoid_boxes:
                    if approx_iou_center(bg_box, ab) > 0.10:
                        ok = False
                        break
            if ok:
                return cx, cy
        return self._random_center()

    def _crop_patch(self, img: Image.Image, cx: float, cy: float) -> Image.Image:
        w, h = img.size
        size = self.img_size
        px = int(cx * w)
        py = int(cy * h)

        x1 = px - size // 2
        y1 = py - size // 2
        x2 = x1 + size
        y2 = y1 + size

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)

        if pad_l or pad_t or pad_r or pad_b:
            ten = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            ten = F.pad(ten, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0.0)
            px += pad_l
            py += pad_t
            x1 = px - size // 2
            y1 = py - size // 2
            x2 = x1 + size
            y2 = y1 + size
            patch = ten[:, y1:y2, x1:x2]
            patch = (patch * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(patch)

        return img.crop((x1, y1, x2, y2))

    def _augment(self, patch: Image.Image) -> Image.Image:
        if not self.do_aug:
            return patch
        if random.random() < 0.5:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.1:
            patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
        return patch

    def _to_tensor(self, patch: Image.Image) -> torch.Tensor:
        arr = np.asarray(patch).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(str(row[self.img_col]))
        gt = safe_int(row.get(self.gt_col, 0))
        stem = img_path.stem
        lbl_path = self.labels_dir / f"{stem}.txt"

        boxes = parse_yolo_txt(lbl_path) if lbl_path.exists() else []
        avoid_boxes = [b for b in boxes if b[0] in CFG.AVOID_CLASSES]  # e.g., text overlays
        frac_boxes = [b for b in boxes if b[0] == self.fracture_cls]
        other_boxes = [b for b in boxes if (b[0] != self.fracture_cls and b[0] not in CFG.AVOID_CLASSES)]

        img = self._load_rgb(img_path)

        # Optional: mask text overlays to reduce shortcut learning
        if str(CFG.TEXT_MASK_MODE).lower().strip() != "none":
            apply_to = str(getattr(CFG, "TEXT_MASK_APPLY_TO", "all")).lower().strip()
            if apply_to == "all" or (apply_to == "train" and self.split_tag == "train"):
                img = self._apply_text_mask(img, avoid_boxes)

        # Determine sampling counts
        k_pos = int(round(self.k * self.pos_ratio))
        k_pos = max(1, k_pos) if gt == 1 else 0
        k_bg = self.k - k_pos if gt == 1 else self.k

        k_hard = int(round(self.k * self.hardneg_ratio)) if gt == 0 else 0
        k_hard = min(k_hard, self.k)
        k_rand = self.k - k_hard

        centers: list[tuple[float, float]] = []

        if gt == 1:
            # positive bag: some from fracture boxes, rest from background
            if len(frac_boxes) > 0:
                for _ in range(k_pos):
                    centers.append(self._bbox_to_center(random.choice(frac_boxes)))
            else:
                for _ in range(k_pos):
                    centers.append(self._random_center_avoiding(avoid_boxes))

            for _ in range(k_bg):
                centers.append(self._sample_bg_center(frac_boxes, avoid_boxes))
        else:
            # negative bag:
            if len(frac_boxes) > 0:
                for _ in range(k_hard):
                    centers.append(self._bbox_to_center(random.choice(frac_boxes)))
            elif len(other_boxes) > 0:
                for _ in range(k_hard):
                    centers.append(self._bbox_to_center(random.choice(other_boxes)))
            else:
                for _ in range(k_hard):
                    cx = random.choice([random.uniform(0.05, 0.25), random.uniform(0.75, 0.95)])
                    cy = random.uniform(0.15, 0.85)
                    if self._avoid_center(cx, cy, avoid_boxes):
                        cx, cy = self._random_center_avoiding(avoid_boxes)
                    centers.append((cx, cy))

            for _ in range(k_rand):
                centers.append(self._random_center_avoiding(avoid_boxes))

        patches = []
        for (cx, cy) in centers:
            p = self._crop_patch(img, cx, cy)
            p = self._augment(p)
            patches.append(self._to_tensor(p))

        x = torch.stack(patches, dim=0)  # [K,3,H,W]
        y = torch.tensor(gt, dtype=torch.float32)

        meta = {
            "img_path": str(img_path),
            "stem": stem,
            "label_path": str(lbl_path),
            "has_label": int(lbl_path.exists()),
            "n_frac_boxes": int(len(frac_boxes)),
        }
        return x, y, meta
# =========================
# Model
# =========================
class ConvNeXtMIL(nn.Module):
    def __init__(self, backbone_name: str, topk_k: int, use_checkpointing: bool):
        super().__init__()
        self.topk_k = int(topk_k)

        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="")
        # Determine feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE)
            feat = self.backbone.forward_features(dummy)
            if feat.ndim == 4:
                feat = feat.mean(dim=(2,3))
            self.feat_dim = feat.shape[-1]

        self.instance_head = nn.Linear(self.feat_dim, 1)
        self.use_checkpointing = bool(use_checkpointing)

        if self.use_checkpointing:
            # timm convnext has stages; we can checkpoint forward_features via torch.utils.checkpoint
            try:
                import torch.utils.checkpoint as ckpt
                self._ckpt = ckpt
            except Exception:
                self._ckpt = None
                self.use_checkpointing = False

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,3,H,W]
        if not self.use_checkpointing:
            feat = self.backbone.forward_features(x)
        else:
            # checkpoint the forward_features as a whole (simple but effective)
            feat = self._ckpt.checkpoint(self.backbone.forward_features, x)
        if feat.ndim == 4:
            feat = feat.mean(dim=(2,3))
        return feat

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        bag: [B,K,3,H,W]
        Returns:
          bag_logit: [B]
          inst_prob: [B,K]
        """
        B, K, C, H, W = bag.shape
        x = bag.view(B*K, C, H, W)
        feat = self._forward_features(x)           # [B*K, D]
        inst_logit = self.instance_head(feat).view(B, K)  # [B,K]
        inst_prob = torch.sigmoid(inst_logit)

        # top-k mean pooling on probabilities, convert back to logit for BCEWithLogits
        k = min(self.topk_k, K)
        topk, _ = torch.topk(inst_prob, k=k, dim=1, largest=True, sorted=False)  # [B,k]
        bag_prob = topk.mean(dim=1)  # [B]
        bag_logit = safe_logit(bag_prob)

        return bag_logit, inst_prob


# =========================
# Loss
# =========================
class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B], targets: [B]
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1 - p)
        loss = bce * ((1 - pt) ** self.gamma)
        return loss.mean()



def _auroc_rank(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUROC via rank statistic (Mann–Whitney U). Returns NaN if undefined."""
    try:
        y_true = y_true.astype(np.int32)
        pos = y_prob[y_true == 1]
        neg = y_prob[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        combined = np.concatenate([pos, neg])
        ranks = combined.argsort().argsort().astype(np.float64) + 1.0
        r_pos = ranks[:len(pos)].sum()
        auroc = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
        return float(auroc)
    except Exception:
        return float("nan")


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                    mode: str = "youden",
                    grid_size: int = 401,
                    thr_min: float = 0.0,
                    thr_max: float = 1.0,
                    # targets for constrained modes
                    rec_target: float = 0.0,
                    spec_target: float = 0.0,
                    # optional global hard constraints (applied in all modes)
                    min_spec: float = 0.0,
                    min_rec: float = 0.0) -> float:
    """Find threshold in [thr_min, thr_max] that optimizes a mode.

    Modes:
      - youden: maximize (recall + specificity - 1)
      - f1: maximize F1
      - spec_at_rec: maximize specificity subject to recall >= rec_target
      - rec_at_spec: maximize recall subject to specificity >= spec_target
    """
    y_true = y_true.astype(np.int32)
    if len(y_true) == 0:
        return 0.5

    mode = str(mode).lower().strip()
    grid_size = int(max(11, grid_size))

    # sanitize bounds
    thr_min = float(np.clip(thr_min, 0.0, 1.0))
    thr_max = float(np.clip(thr_max, 0.0, 1.0))
    if thr_max <= thr_min:
        thr_min, thr_max = 0.0, 1.0

    thrs = np.linspace(thr_min, thr_max, grid_size, dtype=np.float32)

    best_thr = 0.5
    best_score = -1e18

    for thr in thrs:
        y_pred = (y_prob >= thr).astype(np.int32)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)

        # global hard constraints
        if spec < float(min_spec) or rec < float(min_rec):
            continue

        if mode == "f1":
            score = 2.0 * prec * rec / max(1e-12, (prec + rec))
        elif mode == "spec_at_rec":
            if rec < float(rec_target):
                continue
            score = spec
        elif mode == "rec_at_spec":
            if spec < float(spec_target):
                continue
            score = rec
        else:
            # default: Youden's J
            score = rec + spec - 1.0

        if score > best_score:
            best_score = score
            best_thr = float(thr)

    # fallback: if no threshold satisfied constraints, relax to unclamped Youden on full [0,1]
    if best_score <= -1e17:
        thrs = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
        best_thr = 0.5
        best_score = -1e18
        for thr in thrs:
            y_pred = (y_prob >= thr).astype(np.int32)
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            rec = tp / max(1, tp + fn)
            spec = tn / max(1, tn + fp)
            score = rec + spec - 1.0
            if score > best_score:
                best_score = score
                best_thr = float(thr)

    return float(best_thr)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    y_true = y_true.astype(np.int32)
    y_pred = (y_prob >= float(thr)).astype(np.int32)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))
    auroc = _auroc_rank(y_true, y_prob)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "acc": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1,
        "auroc": auroc,
        "thr": float(thr),
    }



# =========================
# Train / Eval
# =========================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    """DDP-safe evaluation with optional threshold tuning on rank0.

    - Collects y_true/y_prob on each rank (CPU).
    - If CFG.TUNE_THRESHOLD:
        * rank0 gathers arrays from all ranks via all_gather_object
        * computes best threshold (Youden/F1 with constraints)
        * broadcasts threshold (and AUROC) back to all ranks
    - Confusion counts are computed locally from cached arrays, then all-reduced.
    """
    model.eval()

    y_true_list = []
    y_prob_list = []

    for bag, y, _meta in loader:
        bag = bag.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1)

        bag_logit, _inst = model(bag)
        prob = torch.sigmoid(bag_logit).detach().float().cpu().numpy()
        yt = (y >= 0.5).detach().to(torch.int32).cpu().numpy()

        y_prob_list.append(prob)
        y_true_list.append(yt)

    y_true_local = np.concatenate(y_true_list, axis=0) if len(y_true_list) else np.zeros((0,), dtype=np.int32)
    y_prob_local = np.concatenate(y_prob_list, axis=0) if len(y_prob_list) else np.zeros((0,), dtype=np.float32)

    # Determine threshold + AUROC on rank0 (gather), then broadcast
    thr = 0.5
    auroc = float("nan")

    if getattr(CFG, "TUNE_THRESHOLD", True):
        if is_distributed() and torch.distributed.is_initialized():
            rank, world, _ = get_dist_info()
            gathered_true = [None for _ in range(world)]
            gathered_prob = [None for _ in range(world)]
            torch.distributed.all_gather_object(gathered_true, y_true_local)
            torch.distributed.all_gather_object(gathered_prob, y_prob_local)

            if rank == 0:
                y_true_all = np.concatenate([g for g in gathered_true if g is not None and len(g) > 0], axis=0) if any(g is not None and len(g) > 0 for g in gathered_true) else np.zeros((0,), dtype=np.int32)
                y_prob_all = np.concatenate([g for g in gathered_prob if g is not None and len(g) > 0], axis=0) if any(g is not None and len(g) > 0 for g in gathered_prob) else np.zeros((0,), dtype=np.float32)

                thr = _best_threshold(
                    y_true_all, y_prob_all,
                    mode=str(getattr(CFG, "THRESH_MODE", getattr(CFG, "THRESH_STRATEGY", "youden"))),
                    grid_size=int(getattr(CFG, "THRESH_GRID_SIZE", 401)),
                    thr_min=float(getattr(CFG, "THRESH_MIN", 0.0)),
                    thr_max=float(getattr(CFG, "THRESH_MAX", 1.0)),
                    rec_target=float(getattr(CFG, "THRESH_REC_TARGET", 0.0)),
                    spec_target=float(getattr(CFG, "THRESH_SPEC_TARGET", 0.0)),
                    min_spec=float(getattr(CFG, "THRESH_MIN_SPEC", 0.0)),
                    min_rec=float(getattr(CFG, "THRESH_MIN_REC", 0.0)),
                )
                auroc = _auroc_rank(y_true_all, y_prob_all)

            thr_t = torch.tensor([float(thr)], device=device, dtype=torch.float32)
            auc_t = torch.tensor([float(auroc) if (auroc == auroc) else -1.0], device=device, dtype=torch.float32)
            torch.distributed.broadcast(thr_t, src=0)
            torch.distributed.broadcast(auc_t, src=0)
            thr = float(thr_t.item())
            auroc = float("nan") if float(auc_t.item()) < 0 else float(auc_t.item())
        else:
            # single process
            thr = _best_threshold(
                y_true_local, y_prob_local,
                mode=str(getattr(CFG, "THRESH_MODE", getattr(CFG, "THRESH_STRATEGY", "youden"))),
                grid_size=int(getattr(CFG, "THRESH_GRID_SIZE", 401)),
                thr_min=float(getattr(CFG, "THRESH_MIN", 0.0)),
                thr_max=float(getattr(CFG, "THRESH_MAX", 1.0)),
                rec_target=float(getattr(CFG, "THRESH_REC_TARGET", 0.0)),
                spec_target=float(getattr(CFG, "THRESH_SPEC_TARGET", 0.0)),
                min_spec=float(getattr(CFG, "THRESH_MIN_SPEC", 0.0)),
                min_rec=float(getattr(CFG, "THRESH_MIN_REC", 0.0)),
            )
            auroc = _auroc_rank(y_true_local, y_prob_local)

    # Compute confusion at chosen threshold using cached arrays (no second forward pass)
    y_pred_local = (y_prob_local >= float(thr)).astype(np.int32)
    tp = int(((y_true_local == 1) & (y_pred_local == 1)).sum())
    fp = int(((y_true_local == 0) & (y_pred_local == 1)).sum())
    fn = int(((y_true_local == 1) & (y_pred_local == 0)).sum())
    tn = int(((y_true_local == 0) & (y_pred_local == 0)).sum())

    counts = torch.tensor([tp, fp, fn, tn], device=device, dtype=torch.long)
    counts = all_reduce_sum(counts)
    tp, fp, fn, tn = [int(x) for x in counts.tolist()]

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))

    metrics = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "acc": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1,
        "auroc": float(auroc),
        "thr": float(thr),
    }
    return metrics, y_true_local.astype(np.float32), y_prob_local.astype(np.float32)



# =========================
# OOM Mitigation Helpers (keep batch / K_INSTANCES)
# =========================
def _set_requires_grad(mod: nn.Module, flag: bool) -> None:
    for p in mod.parameters():
        p.requires_grad = bool(flag)


def _get_convnext_stages(backbone: nn.Module):
    # timm ConvNeXt/ConvNeXtV2 usually exposes .stages (len=4). Keep robust fallbacks.
    if hasattr(backbone, "stages") and isinstance(getattr(backbone, "stages"), (list, nn.ModuleList, nn.Sequential)):
        return list(backbone.stages)
    if hasattr(backbone, "stages_") and isinstance(getattr(backbone, "stages_"), (list, nn.ModuleList, nn.Sequential)):
        return list(backbone.stages_)
    # fallback: try blocks (may be a flat list)
    if hasattr(backbone, "blocks") and isinstance(getattr(backbone, "blocks"), (list, nn.ModuleList, nn.Sequential)):
        return [backbone.blocks]
    return None


def apply_freeze_schedule(model: ConvNeXtMIL, epoch: int) -> bool:
    """Apply freeze/unfreeze schedule. Returns True if trainable set changed."""
    if not getattr(CFG, "FREEZE_BACKBONE", True):
        return False

    bb = model.backbone
    stages = _get_convnext_stages(bb)

    def _freeze_all():
        _set_requires_grad(bb, False)
        _set_requires_grad(model.instance_head, True)

    changed = False

    # epoch is 1-indexed in training loop
    if epoch <= int(getattr(CFG, "FREEZE_EPOCHS", 0)):
        # head-only
        # Only do once if not already frozen
        if any(p.requires_grad for p in bb.parameters()):
            _freeze_all()
            changed = True
        return changed

    # after freeze period: by default keep backbone frozen, then unfreeze requested stages
    if any(p.requires_grad for p in bb.parameters()) is False:
        # still fully frozen -> will unfreeze some stage(s)
        pass

    # Start from all-frozen baseline then open stages
    # (ensures deterministic trainable set even if someone ran partial unfreeze earlier)
    _freeze_all()

    # unfreeze stage4
    if epoch >= int(getattr(CFG, "UNFREEZE_STAGE4_EPOCH", 10**9)):
        if stages is not None and len(stages) >= 4:
            _set_requires_grad(stages[-1], True)
        else:
            # fallback: unfreeze last ~25% of blocks if flat
            if hasattr(bb, "blocks"):
                blocks = list(bb.blocks)
                n = max(1, len(blocks) // 4)
                for b in blocks[-n:]:
                    _set_requires_grad(b, True)

    # optional unfreeze stage3
    if epoch >= int(getattr(CFG, "UNFREEZE_STAGE3_EPOCH", 10**9)):
        if stages is not None and len(stages) >= 4:
            _set_requires_grad(stages[-2], True)

    # Determine if changed: compare number of trainable params cached on model
    cur = sum(p.requires_grad for p in model.parameters())
    prev = getattr(model, "_trainable_param_count", None)
    if prev is None or prev != cur:
        model._trainable_param_count = cur
        changed = True
    return changed


def build_optimizer(model: nn.Module):
    """Build optimizer with minimal VRAM (trainable-only)."""
    params = model.parameters()
    if getattr(CFG, "OPTIM_TRAINABLE_ONLY", True):
        params = [p for p in model.parameters() if p.requires_grad]

    if getattr(CFG, "USE_8BIT_ADAMW", False):
        try:
            import bitsandbytes as bnb  # type: ignore
            return bnb.optim.AdamW8bit(params, lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        except Exception as e:
            print(f"[WARN] USE_8BIT_ADAMW=True but bitsandbytes not available ({e}). Falling back to torch AdamW.")
            return torch.optim.AdamW(params, lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)

    return torch.optim.AdamW(params, lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)


def apply_threshold_policy(view_name: str):
    """Adjust threshold selection policy based on current view/age split.

    Rationale:
    - age0/1/2 splits tend to be pos-heavy (per scan reports) -> use spec_at_rec with a slightly relaxed rec target.
    - age3 split tends to be closer to balanced or neg-heavy -> use rec_at_spec to prevent excessive FP while preserving recall.
    - Also widen clamp range to avoid infeasible constraints caused by too-narrow [0.2,0.8].
    " triggering extreme thresholds.
    """
    if not bool(getattr(CFG, "AUTO_THR_POLICY", True)):
        return

    name = str(view_name)
    m = re.search(r"^(AP|LAT)_age([0-3])$", name)
    if not m:
        # best effort: try to infer trailing age digit
        m2 = re.search(r"(AP|LAT).*age([0-3])", name)
        if not m2:
            return
        view = m2.group(1)
        age = int(m2.group(2))
    else:
        view = m.group(1)
        age = int(m.group(2))

    if age == 3:
        thr_min, thr_max = getattr(CFG, "AUTO_THR_RANGE_AGE3", (0.05, 0.95))
        CFG.THRESH_MODE = "rec_at_spec"
        CFG.THRESH_SPEC_TARGET = float(getattr(CFG, "AUTO_THR_SPEC_TARGET", 0.95))
        # keep rec_target as-is (unused in this mode)
    else:
        thr_min, thr_max = getattr(CFG, "AUTO_THR_RANGE_POSHEAVY", (0.10, 0.90))
        CFG.THRESH_MODE = "spec_at_rec"
        CFG.THRESH_REC_TARGET = float(getattr(CFG, "AUTO_THR_REC_TARGET", 0.90))
        # keep spec_target as-is (unused in this mode)

    CFG.THRESH_MIN = float(thr_min)
    CFG.THRESH_MAX = float(thr_max)

    if is_main_process():
        print(f"[THR-POLICY] {view}_age{age} -> mode={CFG.THRESH_MODE} "
              f"range=[{CFG.THRESH_MIN:.2f},{CFG.THRESH_MAX:.2f}] "
              f"rec_target={getattr(CFG,'THRESH_REC_TARGET',0.0):.2f} "
              f"spec_target={getattr(CFG,'THRESH_SPEC_TARGET',0.0):.2f} "
              f"grid={int(getattr(CFG,'THRESH_GRID_SIZE',401))}")

def train_one_view(view_name: str, train_csv: str, val_csv: str, out_dir: Path, mlflow=None):
    print(f"\n========== [VIEW={view_name}] ==========")
    apply_threshold_policy(view_name)


    # DDP init (no-op if single GPU)
    ddp_setup()
    rank, world_size, local_rank = get_dist_info()
    if is_distributed():
        print(f"[DDP] rank={rank} world_size={world_size} local_rank={local_rank}")

    out_dir = ensure_dir(out_dir)

    labels_dir = Path(CFG.YOLO_LABELS_DIR)
    if not labels_dir.exists():
        print(f"[WARN] YOLO_LABELS_DIR not found: {labels_dir}. Labels will be treated as missing.")
    # Pre-scan report
    if is_main_process():
        scan_dataset(train_csv, labels_dir, out_dir, tag=f"{view_name}_train")
    if is_main_process():
        scan_dataset(val_csv, labels_dir, out_dir, tag=f"{view_name}_val")

    # Dataset / Loader
    train_ds = WristMILDataset(train_csv, CFG.YOLO_LABELS_DIR, CFG.IMG_SIZE, CFG.K_INSTANCES,
                               CFG.POS_RATIO, CFG.HARDNEG_RATIO, CFG.FRACTURE_CLASS_ID, split_tag="train")
    val_ds = WristMILDataset(val_csv, CFG.YOLO_LABELS_DIR, CFG.IMG_SIZE, CFG.K_INSTANCES,
                             CFG.POS_RATIO, CFG.HARDNEG_RATIO, CFG.FRACTURE_CLASS_ID, split_tag="val")

    
    # Samplers (DDP-safe)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed() else None

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=(train_sampler is None),
                          sampler=train_sampler,
                          num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True,
                          persistent_workers=(CFG.NUM_WORKERS > 0))
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                        sampler=val_sampler,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False,
                        persistent_workers=(CFG.NUM_WORKERS > 0))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtMIL(CFG.TIMM_BACKBONE, CFG.TOPK_POOL_K, CFG.USE_CHECKPOINTING)

    if CFG.CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    model.to(device)

    # Apply freeze schedule BEFORE wrapping with DDP so requires_grad is set correctly
    net = model
    apply_freeze_schedule(net, epoch=1)

    if is_distributed():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=True)

    if CFG.TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    net = model.module if hasattr(model, 'module') else model

    opt = build_optimizer(net)

    
    if CFG.USE_FOCAL:
        crit = FocalLossWithLogits(gamma=CFG.FOCAL_GAMMA)
    else:
        pos_weight_t = None
        if getattr(CFG, "USE_POS_WEIGHT", True):
            # pos_weight = (#neg / #pos). For pos-dominant splits, pos_weight < 1 -> reduces positive bias.
            try:
                y_col = train_ds.gt_col
                y_vals = train_ds.df[y_col].astype(np.float32).values
                pos = float((y_vals >= 0.5).sum())
                neg = float((y_vals < 0.5).sum())
                if pos > 0:
                    pw = neg / pos
                    lo, hi = getattr(CFG, "POS_WEIGHT_CLAMP", (0.25, 4.0))
                    pw = float(max(float(lo), min(float(hi), pw)))
                    pos_weight_t = torch.tensor([pw], device=device, dtype=torch.float32)
                    if is_main_process():
                        print(f"[LOSS] Using BCE pos_weight={pw:.4f} (neg={neg:.0f}, pos={pos:.0f})")
            except Exception as e:
                if is_main_process():
                    print(f"[WARN] pos_weight calc failed: {e}")
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t) if pos_weight_t is not None else nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=CFG.AMP, init_scale=float(getattr(CFG,'AMP_INIT_SCALE',1024.0)), growth_interval=int(getattr(CFG,'AMP_GROWTH_INTERVAL',2000)))

    best_score = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    global_step = 0
    for epoch in range(1, CFG.EPOCHS + 1):
        if is_distributed() and train_loader.sampler is not None and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        net = model.module if hasattr(model, 'module') else model

        # Freeze/Unfreeze schedule (keeps batch/K_INSTANCES unchanged)
        if apply_freeze_schedule(net, epoch=epoch) and epoch != 1:
            # Trainable set changed -> rebuild optimizer on trainable params only
            opt = build_optimizer(net)
            opt.zero_grad(set_to_none=True)
        t0 = time.time()

        running_loss = 0.0
        n_batches = 0

        opt.zero_grad(set_to_none=True)

        for it, (bag, y, _meta) in enumerate(train_loader, start=1):
            bag = bag.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1)

            with torch.cuda.amp.autocast(enabled=CFG.AMP):
                bag_logit, inst_prob = model(bag)
                loss_main = crit(bag_logit, y)

                loss = loss_main

                if CFG.NEG_SUPPRESS:
                    # penalize high instance probabilities in negative bags
                    neg_mask = (y < 0.5).view(-1, 1)  # [B,1]
                    if neg_mask.any():
                        neg_probs = inst_prob * neg_mask
                        # only consider negatives
                        # relu(p - margin)
                        penalty = F.relu(neg_probs - CFG.NEG_MARGIN)
                        # average over instances and negatives only
                        denom = neg_mask.sum() * inst_prob.shape[1] + 1e-6
                        loss_neg = penalty.sum() / denom
                        loss = loss + CFG.NEG_LAMBDA * loss_neg

            
# Stability: skip non-finite loss before backward to avoid NaN propagation
            # Stability: skip non-finite loss before backward to avoid NaN propagation
            if getattr(CFG, "SKIP_NONFINITE_LOSS", True) and (not torch.isfinite(loss)):
                if it % CFG.PRINT_EVERY == 0:
                    print(f"[WARN] non-finite loss detected. Skipping batch. ep={epoch} it={it} loss={loss}")
                opt.zero_grad(set_to_none=True)
                # scaler.update()  # skipped because no inf checks are recorded when we skip before backward
                continue

            loss = loss / CFG.ACCUM_STEPS
            scaler.scale(loss).backward()

            if it % CFG.ACCUM_STEPS == 0:
                # unscale for clipping
                if CFG.AMP:
                    scaler.unscale_(opt)
                if getattr(CFG, "GRAD_CLIP_NORM", 0.0) and float(CFG.GRAD_CLIP_NORM) > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), float(CFG.GRAD_CLIP_NORM))

                # extra safety: if any grad is non-finite, skip step
                grads_ok = True
                for p in net.parameters():
                    if p.grad is not None and (not torch.isfinite(p.grad).all()):
                        grads_ok = False
                        break
                if grads_ok:
                    scaler.step(opt)
                else:
                    if it % CFG.PRINT_EVERY == 0:
                        print(f"[WARN] non-finite gradients detected. Skipping optimizer step. ep={epoch} it={it}")
                    opt.zero_grad(set_to_none=True)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += float(loss.detach().cpu().item()) * CFG.ACCUM_STEPS
            n_batches += 1

            if it % CFG.PRINT_EVERY == 0:
                print(f"[Train] Ep{epoch:03d} it{it:05d}/{len(train_loader)} loss={running_loss/max(1,n_batches):.4f}")

        # flush any remainder
        if (len(train_loader) % CFG.ACCUM_STEPS) != 0:
            if CFG.AMP:
                scaler.unscale_(opt)
            if getattr(CFG, "GRAD_CLIP_NORM", 0.0) and float(CFG.GRAD_CLIP_NORM) > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(CFG.GRAD_CLIP_NORM))
            grads_ok = True
            for p in net.parameters():
                if p.grad is not None and (not torch.isfinite(p.grad).all()):
                    grads_ok = False
                    break
            if grads_ok:
                scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            global_step += 1

        train_loss = running_loss / max(1, n_batches)

        val_metrics, _, _ = evaluate(model, val_loader, device)

        dt = time.time() - t0
        msg = (f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | "
               f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} "
               f"val_prec={val_metrics['precision']:.4f} val_rec={val_metrics['recall']:.4f} "
               f"val_spec={val_metrics['specificity']:.4f} val_thr={val_metrics.get('thr',0.5):.3f} val_auc={val_metrics['auroc']:.4f} | "
               f"time={dt:.1f}s")
        print(msg)

        # MLflow
        if (mlflow is not None) and is_main_process():
            mlflow.log_metric(f"{view_name}/train_loss", train_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"{view_name}/val_{k}", float(v), step=epoch)

        # Save last
        if is_main_process():
            torch.save({"epoch": epoch, "model": net.state_dict(), "cfg": CFG.__dict__}, last_path)

        # Save best
        key = CFG.SAVE_BEST_BY
        score = float(val_metrics["f1"] if key == "val_f1" else val_metrics.get(key.replace("val_", ""), val_metrics["f1"]))
        if score > best_score:
            best_score = score
            if is_main_process():
                torch.save({"epoch": epoch, "model": net.state_dict(), "cfg": CFG.__dict__, "val": val_metrics}, best_path)
            print(f"✅ Saved best: {best_path} (score={best_score:.4f})")

    print(f"[DONE] {view_name} best_score={best_score:.4f} best_path={best_path}")


def main():
    ddp_setup()
    set_seed(CFG.SEED)
    out_root = ensure_dir(CFG.SAVE_DIR)

    mlflow = try_import_mlflow()
    run_ctx = None
    if (mlflow is not None) and is_main_process():
        run_ctx = mlflow.start_run(run_name=f"mil_{Path(CFG.SAVE_DIR).name}")
        mlflow.log_params({k: str(v) for k, v in CFG.__dict__.items()})

    try:
        views = {
            "AP":  (CFG.TRAIN_CSV_AP, CFG.VAL_CSV_AP),
            "LAT": (CFG.TRAIN_CSV_LAT, CFG.VAL_CSV_LAT),
        }

        for view, (tr_csv, va_csv) in views.items():
            print(f"\n========== [VIEW={view}] ==========")
            view_dir = ensure_dir(out_root / view)

            if not getattr(CFG, "USE_AGE_GROUPS", True):
                # single model per view (legacy)
                train_one_view(view, tr_csv, va_csv, view_dir, mlflow=mlflow)
                continue

            # Load once, create age_group
            tr_df = pd.read_csv(tr_csv)
            va_df = pd.read_csv(va_csv)

            age_col_tr = detect_age_col(tr_df)
            age_col_va = detect_age_col(va_df)
            if age_col_tr is None or age_col_va is None:
                raise RuntimeError(
                    f"[AGE] Could not detect age column. "
                    f"Train age_col={age_col_tr}, Val age_col={age_col_va}. "
                    f"Columns(train)={list(tr_df.columns)[:30]}"
                )
            tr_df = add_age_group_column(tr_df, age_col_tr)
            va_df = add_age_group_column(va_df, age_col_va)

            # Drop missing ages if configured
            if getattr(CFG, "DROP_AGE_NAN", True):
                tr_before, va_before = len(tr_df), len(va_df)
                tr_df = tr_df[tr_df["age_group"] >= 0].reset_index(drop=True)
                va_df = va_df[va_df["age_group"] >= 0].reset_index(drop=True)
                dropped_tr = tr_before - len(tr_df)
                dropped_va = va_before - len(va_df)
                if dropped_tr or dropped_va:
                    print(f"[AGE] Dropped rows with missing/invalid age: train={dropped_tr}, val={dropped_va}")

            # Persist splits for reproducibility
            split_dir = ensure_dir(view_dir / "age_splits")
            # Train 4 models for age groups 0..3
            for g in [0, 1, 2, 3]:
                g_tr = tr_df[tr_df["age_group"] == g].reset_index(drop=True)
                g_va = va_df[va_df["age_group"] == g].reset_index(drop=True)

                if len(g_tr) == 0 or len(g_va) == 0:
                    print(f"[AGE] Skip {view}_age{g}: empty split (train={len(g_tr)}, val={len(g_va)})")
                    continue

                g_tr_csv = split_dir / f"{view}_age{g}_train.csv"
                g_va_csv = split_dir / f"{view}_age{g}_val.csv"
                g_tr.to_csv(g_tr_csv, index=False)
                g_va.to_csv(g_va_csv, index=False)

                model_dir = ensure_dir(view_dir / f"age{g}")
                # train with unique name
                train_one_view(f"{view}_age{g}", str(g_tr_csv), str(g_va_csv), model_dir, mlflow=mlflow)
    finally:
        if mlflow is not None and run_ctx is not None:
            mlflow.end_run()


if __name__ == "__main__":
    try:
        main()
    finally:
        ddp_cleanup()
