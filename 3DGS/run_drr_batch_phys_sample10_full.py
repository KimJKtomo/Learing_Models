#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_drr_batch_phys_sample10_full.py

- Spine / Skull 공용
- path 고정 (argparse 없음)
- python run으로 바로 실행
"""

import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================
# PATH (기존과 동일하게 고정)
# =====================================================
CT_ROOT  = "./SpineCT/spine_ct_kaggle/spine_segmentation_nnunet_v2/volumes"
OUT_ROOT = "./SpineCT/drr_out_sample10"

N_SAMPLES = 10


# =====================================================
# DRR / IMAGE PARAM
# =====================================================
ANGLES   = [0, 90, 180, 270]
ROT_AXES = ["x", "y", "z"]

DRR_MODE = "mean"
HU_CLIP  = (-1000.0, 2000.0)
AIR_HU   = -1000.0

SAVE_16BIT = False
USE_LOCAL_CONTRAST = True
P_LOW, P_HIGH = 5.0, 99.5
SAVE_MID_SLICES = True


# =====================================================
# POSE PARAM
# =====================================================
POSE_RADIUS_MM = 800.0
POSE_RAY_LEN   = 140.0
POSE_RAY_EVERY = 1
SAVE_POSE_PLOT = True


# =====================================================
# Skull clinical views (Euler XYZ, deg)
# =====================================================
SKULL_VIEWS = {
    "PA":        (0,    0,    0),
    "AP":        (0,  180,    0),
    "LAT_R":     (0,   90,    0),
    "LAT_L":     (0,  -90,    0),
    "TOWNE":     (-30, 180,   0),
    "CALDWELL":  (15,   0,    0),
    "SMV":       (90,   0,    0),
}


# =====================================================
# UTIL
# =====================================================
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def get_physical_center(img):
    size = np.array(img.GetSize(), dtype=np.float64)
    sp   = np.array(img.GetSpacing(), dtype=np.float64)
    org  = np.array(img.GetOrigin(), dtype=np.float64)
    dirm = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    center_idx = (size - 1.0) / 2.0
    return org + dirm @ (center_idx * sp)


def normalize_to_uint(img2d):
    v = img2d.astype(np.float32)
    if USE_LOCAL_CONTRAST:
        lo = np.percentile(v, P_LOW)
        hi = np.percentile(v, P_HIGH)
    else:
        lo, hi = v.min(), v.max() + 1e-6
    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo + 1e-6)
    return (v * 255).astype(np.uint8)


def save_png(arr, path):
    _ensure_dir(os.path.dirname(path))
    Image.fromarray(arr).save(path)


def drr_project(img):
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    vol = np.clip(vol, HU_CLIP[0], HU_CLIP[1])
    mask = (vol > (AIR_HU + 50)).astype(np.float32)
    num = (vol * mask).sum(axis=0)
    den = mask.sum(axis=0) + 1e-6
    return num / den


# =====================================================
# ROTATION
# =====================================================
def rotate_physical_axis(img, angle_deg, axis):
    center = get_physical_center(img)
    t = sitk.Euler3DTransform()
    t.SetCenter(tuple(center))
    r = np.deg2rad(angle_deg)
    rx = ry = rz = 0.0
    if axis == "x": rx = r
    if axis == "y": ry = r
    if axis == "z": rz = r
    t.SetRotation(rx, ry, rz)
    return sitk.Resample(img, img, t, sitk.sitkLinear, AIR_HU)


def rotate_physical_euler(img, rx, ry, rz):
    center = get_physical_center(img)
    t = sitk.Euler3DTransform()
    t.SetCenter(tuple(center))
    t.SetRotation(np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz))
    return sitk.Resample(img, img, t, sitk.sitkLinear, AIR_HU)


# =====================================================
# POSE (ALL AXES ONE COORD)
# =====================================================
def rotmat(axis, deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    if axis == "x":
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis == "y":
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def make_orbit_centers(axis, angles, center, r):
    if axis == "x": base = center + [0,r,0]
    elif axis == "y": base = center + [0,0,r]
    else: base = center + [r,0,0]
    return np.array([center + rotmat(axis,a)@(base-center) for a in angles])


def save_pose_all_axes(out_path, center):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"x":"r","y":"g","z":"b"}
    ax.scatter(*center, marker="x", s=60)
    for axn,c in colors.items():
        pts = make_orbit_centers(axn, ANGLES, center, POSE_RADIUS_MM)
        ax.plot(pts[:,0],pts[:,1],pts[:,2],c=c,label=axn)
        ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=c)
    ax.legend()
    plt.savefig(out_path,dpi=200)
    plt.close()


# =====================================================
# MAIN
# =====================================================
def main():
    _ensure_dir(OUT_ROOT)

    files = sorted([f for f in os.listdir(CT_ROOT) if f.endswith(".nii") or f.endswith(".nii.gz")])[:N_SAMPLES]

    for f in tqdm(files):
        img = sitk.ReadImage(os.path.join(CT_ROOT,f))
        img = sitk.DICOMOrient(img,"LPS")
        case = os.path.splitext(f.replace(".gz",""))[0]
        outdir = os.path.join(OUT_ROOT,case)
        _ensure_dir(outdir)

        center = get_physical_center(img)

        # pose plot
        save_pose_all_axes(os.path.join(outdir,"pose_all_axes.png"), center)

        # axis sweep DRR
        for axn in ROT_AXES:
            for a in ANGLES:
                img_r = rotate_physical_axis(img,a,axn)
                proj = drr_project(img_r)
                save_png(normalize_to_uint(proj),
                         os.path.join(outdir,f"drr_{axn}_{a:03d}.png"))

        # skull views
        skull_dir = os.path.join(outdir,"skull_views")
        _ensure_dir(skull_dir)
        for name,(rx,ry,rz) in SKULL_VIEWS.items():
            img_r = rotate_physical_euler(img,rx,ry,rz)
            proj = drr_project(img_r)
            save_png(normalize_to_uint(proj),
                     os.path.join(skull_dir,f"drr_{name}.png"))

    print("[DONE] all processing finished.")


if __name__ == "__main__":
    main()
