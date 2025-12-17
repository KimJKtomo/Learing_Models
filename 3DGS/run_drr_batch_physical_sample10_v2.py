#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_drr_batch_phys_sample10_with_poseplot.py

- sample N개(.nii/.nii.gz)만 골라서
  1) LPS로 방향 통일
  2) 물리좌표(mm) 기준으로 회전 (SimpleITK Euler3DTransform)
  3) parallel-beam DRR(prototype) 생성
  4) 케이스별 pose plot 저장
     - axis=x/y/z 각각
     - xyz(3분할)
     - all_axes(⭐ 하나의 좌표계에 x/y/z 궤도 전부 겹침)
"""

import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

# 서버/CLI에서도 저장되게 backend 고정
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================
# 여기만 바꿔서 실험 (경로는 그대로 둠)
# =========================================
CT_ROOT = "./SpineCT/spine_ct_kaggle/spine_segmentation_nnunet_v2/volumes"
OUT_ROOT = "./SpineCT/drr_out_spine_sample_1215"

N_SAMPLES = 10

# 축/각도: 축 확인 목적이면 크게, 실제 orbit 생성이면 촘촘하게
ANGLES   = [0, 90, 180, 270]          # 빠른 확인
ROT_AXES = ["x", "y", "z"]            # 축별 스윕

DRR_MODE = "mean"                     # mean / max / sum(비권장)
HU_CLIP  = (-1000.0, 2000.0)
AIR_HU   = -1000.0

SAVE_16BIT = False
USE_LOCAL_CONTRAST = True
P_LOW, P_HIGH = 5.0, 99.5

SAVE_MID_SLICES = True

# pose plot 쪽 설정
POSE_RADIUS_MM = 800.0                # 보기용 orbit radius (mm). 대충 500~1200
POSE_RAY_LEN   = 140.0                # 시선(ray) 길이
POSE_RAY_EVERY = 1                    # 1이면 전부, 6이면 6개마다 1개
SAVE_POSE_PLOT = True                 # 케이스별 pose plot png 저장
# =========================================


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def get_physical_center(img: sitk.Image) -> np.ndarray:
    # 회전 중심 / pose center로 쓸 물리 중심(mm)
    size = np.array(img.GetSize(), dtype=np.float64)      # (x,y,z)
    sp   = np.array(img.GetSpacing(), dtype=np.float64)
    org  = np.array(img.GetOrigin(), dtype=np.float64)
    dirm = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)

    center_idx = (size - 1.0) / 2.0
    center_mm  = org + dirm @ (center_idx * sp)
    return center_mm


def rotate_physical(img: sitk.Image, angle_deg: float, axis: str) -> sitk.Image:
    # 물리좌표(mm) 기준 회전. numpy rotate 금지.
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

    # output grid는 입력 그대로(축 확인용). 잘리면 다음 단계에서 ref 확장.
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
    # 축 감 잡는 용도: 중앙 단면 3장만
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
    # parallel-beam projection (프로토타입)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # [z,y,x]
    vol = np.clip(vol, HU_CLIP[0], HU_CLIP[1])

    # 공기 영향 줄이기 (대충 air보다 좀 큰 값만 통과)
    mask = (vol > (AIR_HU + 50)).astype(np.float32)

    if mode == "max":
        proj = vol.max(axis=0)
    elif mode == "sum":
        proj = (vol * mask).sum(axis=0)
    else:
        # mean(= air 제외 평균) 느낌
        num = (vol * mask).sum(axis=0)
        den = mask.sum(axis=0) + 1e-6
        proj = num / den

    return proj


# =========================
# pose plot (pointcloud 느낌)
# =========================
def rotmat(axis: str, deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    if axis == "x":
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s,  c]], dtype=np.float64)
    if axis == "y":
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)


def make_orbit_centers(axis: str, angles_deg, center_mm: np.ndarray, radius_mm: float) -> np.ndarray:
    # 핵심: 회전축이랑 baseline이 평행이면 원이 "안 보임"
    # - 예) baseline을 +X로 잡고 axis=x로 돌리면 점이 거의 안 움직임
    # 그래서 axis별로 baseline을 "축에 수직"으로 잡음.
    if axis == "x":
        base = center_mm + np.array([0.0, radius_mm, 0.0], dtype=np.float64)  # +Y
    elif axis == "y":
        base = center_mm + np.array([0.0, 0.0, radius_mm], dtype=np.float64)  # +Z
    else:  # "z"
        base = center_mm + np.array([radius_mm, 0.0, 0.0], dtype=np.float64)  # +X

    centers = []
    for a in angles_deg:
        R = rotmat(axis, a)
        C = center_mm + R @ (base - center_mm)
        centers.append(C)
    return np.stack(centers, axis=0)


def set_equal_aspect(ax, pts: np.ndarray):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    r = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    if r < 1e-6:
        r = 1.0

    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def save_pose_pointcloud_png(out_path: str,
                             center_mm: np.ndarray,
                             rot_axis: str,
                             angles_deg,
                             radius_mm: float,
                             ray_len: float,
                             ray_every: int):
    cams = make_orbit_centers(rot_axis, angles_deg, center_mm, radius_mm)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Camera viewpoints (pointcloud) | axis={rot_axis} | N={len(angles_deg)}")

    # center
    ax.scatter([center_mm[0]], [center_mm[1]], [center_mm[2]], marker="x")
    ax.text(center_mm[0], center_mm[1], center_mm[2], "  center")

    # camera centers + orbit
    ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], s=10)
    ax.plot(cams[:, 0], cams[:, 1], cams[:, 2])

    # view rays (camera -> center) 일부만
    step = max(1, int(ray_every))
    for i in range(0, len(angles_deg), step):
        C = cams[i]
        v = center_mm - C
        v = v / (np.linalg.norm(v) + 1e-8)
        P2 = C + v * ray_len
        ax.plot([C[0], P2[0]], [C[1], P2[1]], [C[2], P2[2]])
        ax.text(C[0], C[1], C[2], f" {angles_deg[i]}°")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    set_equal_aspect(ax, np.vstack([cams, center_mm[None, :]]))

    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pose_pointcloud_xyz_png(out_path: str,
                                 center_mm: np.ndarray,
                                 angles_deg,
                                 radius_mm: float,
                                 ray_len: float,
                                 ray_every: int):
    # x/y/z를 한 장에 3개 subplot으로 저장 (비교용)
    fig = plt.figure(figsize=(18, 6))
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]
    rot_axes = ["x", "y", "z"]

    for ax_i, rot_axis in enumerate(rot_axes):
        ax3d = axes[ax_i]
        cams = make_orbit_centers(rot_axis, angles_deg, center_mm, radius_mm)

        ax3d.set_title(f"axis={rot_axis} | N={len(angles_deg)}")

        # center
        ax3d.scatter([center_mm[0]], [center_mm[1]], [center_mm[2]], marker="x")
        ax3d.text(center_mm[0], center_mm[1], center_mm[2], "  center")

        # camera centers + orbit
        ax3d.scatter(cams[:, 0], cams[:, 1], cams[:, 2], s=10)
        ax3d.plot(cams[:, 0], cams[:, 1], cams[:, 2])

        # view rays (일부만)
        step = max(1, int(ray_every))
        for i in range(0, len(angles_deg), step):
            C = cams[i]
            v = center_mm - C
            v = v / (np.linalg.norm(v) + 1e-8)
            P2 = C + v * ray_len
            ax3d.plot([C[0], P2[0]], [C[1], P2[1]], [C[2], P2[2]])
            ax3d.text(C[0], C[1], C[2], f" {angles_deg[i]}°")

        ax3d.set_xlabel("X (mm)")
        ax3d.set_ylabel("Y (mm)")
        ax3d.set_zlabel("Z (mm)")

        set_equal_aspect(ax3d, np.vstack([cams, center_mm[None, :]]))

    fig.suptitle("Camera viewpoints (pointcloud) | XYZ", fontsize=16)
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pose_pointcloud_all_axes_png(out_path: str,
                                      center_mm: np.ndarray,
                                      angles_deg,
                                      radius_mm: float):
    # ⭐ 네가 원한거: 하나의 좌표계에서 x/y/z 궤도 전부 겹쳐보기
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Camera viewpoints (pointcloud) | all axes")

    # center
    ax.scatter([center_mm[0]], [center_mm[1]], [center_mm[2]], marker="x", s=60)
    ax.text(center_mm[0], center_mm[1], center_mm[2], "  center")

    axis_colors = {"x": "red", "y": "green", "z": "blue"}

    all_pts = [center_mm[None, :]]

    for rot_axis, color in axis_colors.items():
        cams = make_orbit_centers(rot_axis, angles_deg, center_mm, radius_mm)
        all_pts.append(cams)

        ax.plot(cams[:, 0], cams[:, 1], cams[:, 2], color=color, linewidth=2, label=f"axis={rot_axis}")
        ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], color=color, s=25)

        # 너무 지저분해지니까 0도만 라벨
        ax.text(cams[0, 0], cams[0, 1], cams[0, 2], f" {rot_axis}:0°", color=color)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()

    all_pts = np.vstack(all_pts)
    set_equal_aspect(ax, all_pts)

    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_one_case(ct_path: str, outdir: str):
    img = sitk.ReadImage(ct_path)

    # nii 방향 다 달라서 일단 LPS로 통일 (DICOM쪽이랑 맞추는 목적)
    img = sitk.DICOMOrient(img, "LPS")

    # 케이스 center(mm)
    center_mm = get_physical_center(img)

    # 메타 한 번 찍어두기
    dir_mat = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    print(f"\n[CASE] {os.path.basename(ct_path)}")
    print(f"  size   = {img.GetSize()}   spacing = {img.GetSpacing()}")
    print(f"  origin = {img.GetOrigin()}")
    print(f"  center(mm) = {center_mm.tolist()}")
    print(f"  direction(LPS) =\n{dir_mat}")

    # pose plot 저장
    if SAVE_POSE_PLOT:
        # 축별 따로
        for ax_ in ROT_AXES:
            pose_png = os.path.join(outdir, f"pose_pointcloud_axis{ax_}.png")
            save_pose_pointcloud_png(
                out_path=pose_png,
                center_mm=center_mm,
                rot_axis=ax_,
                angles_deg=ANGLES,
                radius_mm=POSE_RADIUS_MM,
                ray_len=POSE_RAY_LEN,
                ray_every=POSE_RAY_EVERY
            )

        # xyz 3분할(비교용)
        pose_xyz_png = os.path.join(outdir, "pose_pointcloud_xyz.png")
        save_pose_pointcloud_xyz_png(
            out_path=pose_xyz_png,
            center_mm=center_mm,
            angles_deg=ANGLES,
            radius_mm=POSE_RADIUS_MM,
            ray_len=POSE_RAY_LEN,
            ray_every=POSE_RAY_EVERY
        )

        # ⭐ all axes 한 좌표계(이해용 핵심)
        pose_all_png = os.path.join(outdir, "pose_pointcloud_all_axes.png")
        save_pose_pointcloud_all_axes_png(
            out_path=pose_all_png,
            center_mm=center_mm,
            angles_deg=ANGLES,
            radius_mm=POSE_RADIUS_MM
        )

    if SAVE_MID_SLICES:
        save_mid_slices(img, outdir, tag="orig_LPS")

    # DRR 생성
    for ax_ in ROT_AXES:
        for ang in ANGLES:
            img_r = rotate_physical(img, angle_deg=ang, axis=ax_)

            if SAVE_MID_SLICES:
                save_mid_slices(img_r, outdir, tag=f"rot{ax_}_{ang:03d}")

            proj = drr_project(img_r, mode=DRR_MODE)
            u8 = normalize_to_uint(proj)
            save_png(u8, os.path.join(outdir, f"drr_{ax_}_{ang:03d}.png"))


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

    print("\n[DONE] sample DRR + pose plots saved.")


if __name__ == "__main__":
    main()
