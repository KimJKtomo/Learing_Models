#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_drr_batch_phys_sample10.py

- nii orientation 제각각이라 numpy rotate 쓰면 축 개념 깨짐
- 그래서:
  1) LPS로 통일 (DICOM이랑 기준 맞추기)
  2) 물리좌표(mm) 기준으로 회전 (Euler3DTransform + Resample)
  3) DRR은 일단 parallel-beam(프로토타입)로만 뽑고 축이 맞는지 확인

- nii 많아서 샘플 10개만
- 각도도 크게(0/90/180/270) 돌려서 "누워있는 축"인지 빠르게 확인
"""

import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

# =========================================
# 여기만 바꿔서 실험
# =========================================
CT_ROOT = "./SpineCT/spine_ct_kaggle/spine_segmentation_nnunet_v2/volumes"
OUT_ROOT = "./SpineCT/drr_out_spine_sample"

N_SAMPLES = 10  # 너무 많아서 일단 10개만

# 축 확인은 크게 돌려보는게 제일 빠름
ANGLES    = [0, 90, 180, 270]

# 물리좌표계 기준 회전축 (LPS 기준 x/y/z)
# 한 번에 한 축만 확인하려면 아래 1개만 넣어도 됨
ROT_AXES  = ["x", "y", "z"]

# DRR projection: sum은 잘 터짐. mean이 제일 무난.
DRR_MODE  = "mean"   # mean / max / sum(비권장)

# 대충 bone window 느낌
HU_CLIP   = (-1000.0, 2000.0)

# 배경 처리용 (공기)
AIR_HU    = -1000.0

# 저장 포맷
SAVE_16BIT = False   # True면 16-bit PNG

# 각 DRR마다 명암 자동으로 좀 살리기(확인용)
USE_LOCAL_CONTRAST = True
P_LOW  = 5.0
P_HIGH = 99.5

# 회전축 확인용: 중앙 단면 3장(xy/xz/yz) 같이 저장
SAVE_MID_SLICES = True
# =========================================


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def get_physical_center(img: sitk.Image) -> np.ndarray:
    # 회전 중심이 볼륨 중앙이어야 결과가 안정적임
    # (origin/direction/spacing 반영해서 mm 좌표로 계산)
    size = np.array(img.GetSize(), dtype=np.float64)      # (x,y,z)
    sp   = np.array(img.GetSpacing(), dtype=np.float64)
    org  = np.array(img.GetOrigin(), dtype=np.float64)
    dirm = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)

    center_idx = (size - 1.0) / 2.0
    center_mm  = org + dirm @ (center_idx * sp)
    return center_mm


def rotate_physical(img: sitk.Image, angle_deg: float, axis: str) -> sitk.Image:
    # numpy rotate 쓰면 direction 무시됨 -> 물리회전으로만 간다
    center = get_physical_center(img)

    t = sitk.Euler3DTransform()
    t.SetCenter(tuple(center))

    rad = np.deg2rad(angle_deg)
    rx = ry = rz = 0.0
    if axis == "x":
        rx = rad
    elif axis == "y":
        ry = rad
    else:
        rz = rad
    t.SetRotation(rx, ry, rz)

    # output grid는 입력 그대로 (축 확인용)
    # 잘리는게 보이면 다음 단계에서 reference 확장(padding/resample ref)로 해결
    out = sitk.Resample(
        img, img, t,
        sitk.sitkLinear,
        AIR_HU,
        img.GetPixelID()
    )
    return out


def normalize_to_uint(img2d: np.ndarray) -> np.ndarray:
    v = img2d.astype(np.float32)
    v = np.nan_to_num(v, nan=0.0)

    if USE_LOCAL_CONTRAST:
        lo = np.percentile(v, P_LOW)
        hi = np.percentile(v, P_HIGH)
        if hi <= lo:
            lo, hi = float(v.min()), float(v.max() + 1e-6)
    else:
        lo, hi = float(v.min()), float(v.max() + 1e-6)

    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo + 1e-8)
    v = np.clip(v, 0.0, 1.0)

    if SAVE_16BIT:
        return (v * 65535.0).astype(np.uint16)
    return (v * 255.0).astype(np.uint8)


def save_png(arr: np.ndarray, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    if SAVE_16BIT:
        im = Image.fromarray(arr, mode="I;16")
    else:
        im = Image.fromarray(arr)
    im.save(out_path)


def save_mid_slices(img: sitk.Image, outdir: str, tag: str):
    # 축 감 잡는 용도: 중앙 단면 3장만 뽑아두면 대부분 판단 가능
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # [z,y,x]
    vol = np.clip(vol, HU_CLIP[0], HU_CLIP[1])

    zc = vol.shape[0] // 2
    yc = vol.shape[1] // 2
    xc = vol.shape[2] // 2

    xy = vol[zc, :, :]
    xz = vol[:, yc, :]
    yz = vol[:, :, xc]

    for name, sl in [("xy", xy), ("xz", xz), ("yz", yz)]:
        u8 = normalize_to_uint(sl)
        save_png(u8, os.path.join(outdir, f"mid_{tag}_{name}.png"))


def drr_project(img: sitk.Image, mode: str) -> np.ndarray:
    # parallel-beam 프로젝션 (프로토타입)
    # - sum은 금방 saturation 나서 mean 권장
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # [z,y,x]
    vol = np.clip(vol, HU_CLIP[0], HU_CLIP[1])

    # 패딩/공기(-1000) 쪽이 projection에 덜 끼게 대충 mask
    mask = (vol > (AIR_HU + 50)).astype(np.float32)  # -950HU 이상만 유효로

    if mode == "max":
        proj = vol.max(axis=0)
    elif mode == "sum":
        proj = (vol * mask).sum(axis=0)
    else:
        num = (vol * mask).sum(axis=0)
        den = mask.sum(axis=0) + 1e-6
        proj = num / den

    return proj


def process_one_case(ct_path: str, outdir: str):
    img = sitk.ReadImage(ct_path)

    # nii마다 orientation 제각각이라 일단 LPS로 통일
    # (DICOM이랑 같이 볼거면 이게 편함)
    img = sitk.DICOMOrient(img, "LPS")

    # 메타 한 번 찍어두면 나중에 사고 안남
    dir_mat = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    print(f"\n[CASE] {os.path.basename(ct_path)}")
    print(f"  size   = {img.GetSize()}   spacing = {img.GetSpacing()}")
    print(f"  origin = {img.GetOrigin()}")
    print(f"  direction(LPS) =\n{dir_mat}")

    if SAVE_MID_SLICES:
        save_mid_slices(img, outdir, tag="orig_LPS")

    # 축별로 한 번에 스윕해서 "어느 축이 gantry 회전축인지" 감 잡기
    for ax in ROT_AXES:
        for ang in ANGLES:
            img_r = rotate_physical(img, angle_deg=ang, axis=ax)

            if SAVE_MID_SLICES:
                save_mid_slices(img_r, outdir, tag=f"rot{ax}_{ang:03d}")

            proj = drr_project(img_r, mode=DRR_MODE)
            u8 = normalize_to_uint(proj)
            save_png(u8, os.path.join(outdir, f"drr_{ax}_{ang:03d}.png"))


def main():
    _ensure_dir(OUT_ROOT)

    nii_list = sorted([
        f for f in os.listdir(CT_ROOT)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])

    print(f"[INFO] Found {len(nii_list)} volumes in {CT_ROOT}")
    nii_list = nii_list[:N_SAMPLES]
    print(f"[INFO] Running only N_SAMPLES={N_SAMPLES}")

    for nii in tqdm(nii_list, desc="Sample cases"):
        ct_path = os.path.join(CT_ROOT, nii)
        case_name = os.path.splitext(nii.replace(".gz", ""))[0]
        outdir = os.path.join(OUT_ROOT, case_name)
        _ensure_dir(outdir)

        process_one_case(ct_path, outdir)

    print("\n[DONE] sample DRR generated.")


if __name__ == "__main__":
    main()
