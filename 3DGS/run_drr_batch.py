#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spine CT NIfTI 파일들을 일괄로 DRR 생성하는 스크립트.
- 회전 시 패딩으로 clipping 방지
- 각 projection 별 local contrast normalization으로 너무 어두운 문제 개선
"""

import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import rotate
from PIL import Image
from tqdm import tqdm

# ==========================================================
# 1) 사용자 파라미터 정의
# ==========================================================
CT_ROOT = "./SpineCT/spine_ct_kaggle/spine_segmentation_nnunet_v2/volumes"
OUT_ROOT = "./SpineCT/drr_out_spine"

ANGLES = list(range(0, 360, 10))   # 0~350도, 10° 간격
AXIS = "y"                         # 회전축: x / y / z
DRR_MODE = "sum"                   # sum / mean / max

# HU 기반 글로벌 클리핑 (기본 bone window 느낌)
GLOBAL_CLIP_HU = (-1000.0, 2000.0)

# 회전 시 clipping 방지를 위한 padding 양 (voxel)
PADDING = 128

# local contrast 옵션 (projection 별로 퍼센타일 기반 정규화)
USE_LOCAL_CONTRAST = True
LOCAL_P_LOW = 5.0
LOCAL_P_HIGH = 99.5

SAVE_16BIT = False                 # True → 16-bit PNG
# ==========================================================


def normalize_projection(proj: np.ndarray,
                         out_16bit: bool = False,
                         use_local: bool = True) -> np.ndarray:
    """
    DRR projection을 [0, 255] 또는 [0, 65535]로 정규화.
    - use_local=True: 각 projection의 퍼센타일 기반으로 local contrast 조정
    - use_local=False: GLOBAL_CLIP_HU 기준으로 고정 window 사용
    """
    v = proj.astype(np.float32)

    if use_local:
        # NaN 제거
        v = np.nan_to_num(v, nan=0.0)

        lo = np.percentile(v, LOCAL_P_LOW)
        hi = np.percentile(v, LOCAL_P_HIGH)

        if hi <= lo:
            # 예외적으로 값 범위가 너무 좁으면 global clip으로 fallback
            lo, hi = GLOBAL_CLIP_HU
    else:
        lo, hi = GLOBAL_CLIP_HU

    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo + 1e-8)
    v = np.clip(v, 0.0, 1.0)

    if out_16bit:
        return (v * 65535.0).astype(np.uint16)
    else:
        return (v * 255.0).astype(np.uint8)


def volume_to_drr(vol_hu: np.ndarray,
                  angle_deg: float,
                  axis: str = "y",
                  mode: str = "sum") -> np.ndarray:
    """
    3D HU 볼륨을 주어진 각도로 회전 후 한 축으로 적분하여 DRR 생성 (parallel beam).
    입력 vol_hu shape: [z, y, x]
    """
    if axis == "x":
        rot_axes = (1, 0)  # (y, z)
        proj_axis = 0      # z 방향 적분
    elif axis == "y":
        rot_axes = (0, 2)  # (z, x)
        proj_axis = 0
    else:  # 'z'
        rot_axes = (1, 2)  # (y, x)
        proj_axis = 0

    rotated = rotate(
        vol_hu,
        angle=angle_deg,
        axes=rot_axes,
        reshape=False,   # padding으로 clipping 방지
        order=1,
        mode="nearest"
    )

    if mode == "sum":
        proj = rotated.sum(axis=proj_axis)
    elif mode == "mean":
        proj = rotated.mean(axis=proj_axis)
    else:
        proj = rotated.max(axis=proj_axis)

    return proj


def save_drr_image(drr_2d: np.ndarray, out_path: str):
    img_np = normalize_projection(
        drr_2d,
        out_16bit=SAVE_16BIT,
        use_local=USE_LOCAL_CONTRAST
    )

    if SAVE_16BIT:
        mode = "I;16"
        im = Image.fromarray(img_np, mode)
    else:
        im = Image.fromarray(img_np)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path)


def process_one_ct(ct_path: str, outdir: str):
    print(f"[LOAD] {ct_path}")
    img = sitk.ReadImage(ct_path)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # [z, y, x]

    print(f"[INFO] original shape = {vol.shape}")

    # 1) global HU 클리핑 (extent 제한)
    vol = np.clip(vol, GLOBAL_CLIP_HU[0], GLOBAL_CLIP_HU[1])

    # 2) padding 추가 (회전 시 clipping 방지)
    pad_val = GLOBAL_CLIP_HU[0]  # 공기 수준으로 패딩
    vol = np.pad(
        vol,
        ((PADDING, PADDING), (PADDING, PADDING), (PADDING, PADDING)),
        mode="constant",
        constant_values=pad_val
    )
    print(f"[INFO] padded shape   = {vol.shape}")

    # 3) angle loop
    for angle in ANGLES:
        drr = volume_to_drr(vol, angle_deg=angle, axis=AXIS, mode=DRR_MODE)
        out_path = os.path.join(outdir, f"drr_angle_{angle:03d}.png")
        save_drr_image(drr, out_path)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    nii_list = sorted([
        f for f in os.listdir(CT_ROOT)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])
    print(f"[INFO] Found {len(nii_list)} CT volumes in {CT_ROOT}")

    for nii in tqdm(nii_list, desc="Processing CT cases"):
        ct_path = os.path.join(CT_ROOT, nii)
        case_name = os.path.splitext(nii.replace(".gz", ""))[0]
        outdir = os.path.join(OUT_ROOT, case_name)
        os.makedirs(outdir, exist_ok=True)
        process_one_ct(ct_path, outdir)

    print("[DONE] All DRRs generated.")


if __name__ == "__main__":
    main()
