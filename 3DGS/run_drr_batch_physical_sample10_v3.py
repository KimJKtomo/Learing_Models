#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_drr_batch_physical_sample10_v3.py

- Spine CT (NIfTI) 여러 케이스에서 DRR(prototype parallel-beam) 생성
- NIfTI/DICOM 좌표계 꼬임 방지용으로 케이스마다 LPS로 재정렬(DICOMOrient)
- 물리좌표(mm) 기준으로 Euler 회전(sitk.Euler3DTransform + Resample)

추가(요청사항):
  1) 축별 orbit pose(pointcloud) PNG 저장 (x/y/z)
  2) x/y/z 전부를 "하나의 좌표계"에서 한 번에 보여주는 pose_pointcloud_xyz.png 저장
  3) (추후 Skull용) 지정된 view angle set으로만 DRR 생성 + pose plot 저장 옵션

NOTE
  - 지금 DRR은 "prototype" (parallel-beam + mean/max/sum) 이고,
    실제 X-ray(원근/투영/선감쇠) 모델과 다름. pose plot도 시각화용.
"""

import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

# matplotlib은 서버/CLI에서도 저장되게 backend를 Agg로 박아둠
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================
# 여기만 바꿔서 실험 (path는 기존과 동일 유지)
# =========================================
CT_ROOT = "./SpineCT/spine_ct_kaggle/spine_segmentation_nnunet_v2/volumes"
OUT_ROOT = "./SpineCT/drr_out_spine_sample_1215_3"

N_SAMPLES = 10

# 축/각도: 축 확인 목적이면 크게
ANGLES = [0, 90, 180, 270]
ROT_AXES = ["x", "y", "z"]

DRR_MODE = "mean"                     # mean / max / sum(비권장)
HU_CLIP = (-1000.0, 2000.0)
AIR_HU = -1000.0

SAVE_16BIT = False
USE_LOCAL_CONTRAST = True
P_LOW, P_HIGH = 5.0, 99.5

SAVE_MID_SLICES = True

# pose plot 쪽 설정
POSE_RADIUS_MM = 800.0                 # 보기용 orbit radius (mm). 대충 500~1200
POSE_RAY_LEN = 140.0                   # 시선(ray) 길이
POSE_RAY_EVERY = 1                     # 1이면 전부, 6이면 6개마다 1개
SAVE_POSE_PLOT = True

# --- Skull-like view set (옵션) ---
RUN_SKULL_VIEWS = True                # True면 아래 VIEWS만 DRR + pose plot 생성

VIEWS = {
    # name: (rx, ry, rz) in degrees  (ITK Euler: x,y,z order)
    "PA":        (0,   0,   0),
    "AP":        (0, 180,   0),
    "LAT_R":     (0,  90,   0),
    "LAT_L":     (0, -90,   0),
    "TOWNE":     (-30, 180, 0),
    "CALDWELL":  (15,  0,   0),
    "SMV":       (90,  0,   0),
}

SKULL_POSE_RADIUS_MM = 1200.0
# =========================================


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def get_physical_center(img: sitk.Image) -> np.ndarray:
    # 회전 중심 / pose center로 쓸 물리 중심(mm)
    size = np.array(img.GetSize(), dtype=np.float64)      # (x,y,z)
    sp = np.array(img.GetSpacing(), dtype=np.float64)
    org = np.array(img.GetOrigin(), dtype=np.float64)
    dirm = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)

    center_idx = (size - 1.0) / 2.0
    center_mm = org + dirm @ (center_idx * sp)
    return center_mm


def rotate_physical_axis(img: sitk.Image, angle_deg: float, axis: str) -> sitk.Image:
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

    out = sitk.Resample(
        img, img, t,
        sitk.sitkLinear,
        AIR_HU,
        img.GetPixelID()
    )
    return out


def rotate_physical_euler(img: sitk.Image, rx_deg: float, ry_deg: float, rz_deg: float) -> sitk.Image:
    # 지정 view(rx,ry,rz) 용
    center = get_physical_center(img)
    t = sitk.Euler3DTransform()
    t.SetCenter(tuple(center))
    t.SetRotation(np.deg2rad(rx_deg), np.deg2rad(ry_deg), np.deg2rad(rz_deg))

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

    # 공기 영향 줄이기
    mask = (vol > (AIR_HU + 50)).astype(np.float32)

    if mode == "max":
        proj = vol.max(axis=0)
    elif mode == "sum":
        proj = (vol * mask).sum(axis=0)
    else:
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
                         [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float64)


def euler_rotmat(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    # ITK Euler3DTransform과 비슷하게 (fixed axes x,y,z 순서)로 적용한다고 가정
    Rx = rotmat("x", rx_deg)
    Ry = rotmat("y", ry_deg)
    Rz = rotmat("z", rz_deg)
    return Rz @ Ry @ Rx


def make_orbit_centers(axis: str, angles_deg, center_mm: np.ndarray, radius_mm: float) -> np.ndarray:
    # axis랑 baseline이 평행이면 원이 안 보임 -> 축에 수직한 방향으로 baseline 잡기
    if axis == "x":
        base = center_mm + np.array([0.0, radius_mm, 0.0], dtype=np.float64)  # +Y
    elif axis == "y":
        base = center_mm + np.array([0.0, 0.0, radius_mm], dtype=np.float64)  # +Z
    else:  # "z"
        base = center_mm + np.array([radius_mm, 0.0, 0.0], dtype=np.float64)  # +X

    centers = []
    for a in angles_deg:
        R = rotmat(axis, a)
        v = base - center_mm
        p = center_mm + (R @ v)
        centers.append(p)
    return np.stack(centers, axis=0)


def make_view_centers_from_euler(views: dict, center_mm: np.ndarray, radius_mm: float) -> dict:
    # view별 카메라 위치를 "center에서 radius만큼 떨어진 점"으로 정의
    # 기준점은 +Z로 두고, (rx,ry,rz)로 offset 회전
    base_off = np.array([0.0, 0.0, radius_mm], dtype=np.float64)
    out = {}
    for name, (rx, ry, rz) in views.items():
        R = euler_rotmat(rx, ry, rz)
        out[name] = center_mm + (R @ base_off)
    return out


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


def _plot_center_and_rays(ax, center_mm: np.ndarray, cams: np.ndarray, labels, ray_len: float, ray_every: int):
    # center
    ax.scatter([center_mm[0]], [center_mm[1]], [center_mm[2]], marker="x")
    ax.text(center_mm[0], center_mm[1], center_mm[2], "  center")

    # camera centers
    ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], s=10)
    ax.plot(cams[:, 0], cams[:, 1], cams[:, 2])

    # view rays (camera -> center) 일부만
    step = max(1, int(ray_every))
    for i in range(0, len(labels), step):
        C = cams[i]
        v = center_mm - C
        v = v / (np.linalg.norm(v) + 1e-8)
        P2 = C + v * ray_len
        ax.plot([C[0], P2[0]], [C[1], P2[1]], [C[2], P2[2]])
        ax.text(C[0], C[1], C[2], f" {labels[i]}")


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

    _plot_center_and_rays(ax, center_mm, cams, [f"{a}°" for a in angles_deg], ray_len, ray_every)

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
    # x/y/z orbit을 하나의 좌표계에 전부 그리기
    cams_x = make_orbit_centers("x", angles_deg, center_mm, radius_mm)
    cams_y = make_orbit_centers("y", angles_deg, center_mm, radius_mm)
    cams_z = make_orbit_centers("z", angles_deg, center_mm, radius_mm)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Camera viewpoints (x/y/z orbits) | N={len(angles_deg)}")

    # center
    ax.scatter([center_mm[0]], [center_mm[1]], [center_mm[2]], marker="x")
    ax.text(center_mm[0], center_mm[1], center_mm[2], "  center")

    # 각 orbit은 default color cycle로 알아서 구분
    ax.scatter(cams_x[:, 0], cams_x[:, 1], cams_x[:, 2], s=10, label="axis=x")
    ax.plot(cams_x[:, 0], cams_x[:, 1], cams_x[:, 2])

    ax.scatter(cams_y[:, 0], cams_y[:, 1], cams_y[:, 2], s=10, label="axis=y")
    ax.plot(cams_y[:, 0], cams_y[:, 1], cams_y[:, 2])

    ax.scatter(cams_z[:, 0], cams_z[:, 1], cams_z[:, 2], s=10, label="axis=z")
    ax.plot(cams_z[:, 0], cams_z[:, 1], cams_z[:, 2])

    # ray는 너무 지저분해질 수 있어서, xyz 합친 그림은 label만 최소로
    if ray_len > 0:
        step = max(1, int(ray_every))
        for cams in [cams_x, cams_y, cams_z]:
            for i in range(0, len(angles_deg), step):
                C = cams[i]
                v = center_mm - C
                v = v / (np.linalg.norm(v) + 1e-8)
                P2 = C + v * ray_len
                ax.plot([C[0], P2[0]], [C[1], P2[1]], [C[2], P2[2]])

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend(loc="upper right")
    set_equal_aspect(ax, np.vstack([cams_x, cams_y, cams_z, center_mm[None, :]]))

    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_pose_pointcloud_named_views_png(out_path: str,
                                         center_mm: np.ndarray,
                                         views: dict,
                                         radius_mm: float,
                                         ray_len: float):
    # Skull view set 같은 "discrete views" pointcloud
    view_centers = make_view_centers_from_euler(views, center_mm, radius_mm)
    names = list(view_centers.keys())
    cams = np.stack([view_centers[k] for k in names], axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Camera viewpoints (named views) | N={len(names)}")

    # center
    ax.scatter([center_mm[0]], [center_mm[1]], [center_mm[2]], marker="x")
    ax.text(center_mm[0], center_mm[1], center_mm[2], "  center")

    ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], s=20)
    for i, name in enumerate(names):
        C = cams[i]
        ax.text(C[0], C[1], C[2], f" {name}")

        if ray_len > 0:
            v = center_mm - C
            v = v / (np.linalg.norm(v) + 1e-8)
            P2 = C + v * ray_len
            ax.plot([C[0], P2[0]], [C[1], P2[1]], [C[2], P2[2]])

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    set_equal_aspect(ax, np.vstack([cams, center_mm[None, :]]))

    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def process_one_case(ct_path: str, outdir: str):
    img = sitk.ReadImage(ct_path)

    # nii 방향이 제각각이라, 일단 LPS로 통일
    # (ITK/SimpleITK 기준에서 LPS는 identity direction으로 정의됨)
    img = sitk.DICOMOrient(img, "LPS")

    # 케이스 center(mm)
    center_mm = get_physical_center(img)

    # 메타 한 번 찍어두기 (좌표계/회전 디버깅 핵심)
    dir_mat = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    print(f"\n[CASE] {os.path.basename(ct_path)}")
    print(f"  size   = {img.GetSize()}   spacing = {img.GetSpacing()}")
    print(f"  origin = {img.GetOrigin()}")
    print(f"  center(mm) = {center_mm.tolist()}")
    print(f"  direction(LPS) =\n{dir_mat}")

    # --- pose plot 저장 ---
    if SAVE_POSE_PLOT:
        # 축별
        for ax_name in ROT_AXES:
            pose_png = os.path.join(outdir, f"pose_pointcloud_axis{ax_name}.png")
            save_pose_pointcloud_png(
                out_path=pose_png,
                center_mm=center_mm,
                rot_axis=ax_name,
                angles_deg=ANGLES,
                radius_mm=POSE_RADIUS_MM,
                ray_len=POSE_RAY_LEN,
                ray_every=POSE_RAY_EVERY,
            )

        # xyz 합친 1장
        pose_xyz_png = os.path.join(outdir, "pose_pointcloud_xyz.png")
        save_pose_pointcloud_xyz_png(
            out_path=pose_xyz_png,
            center_mm=center_mm,
            angles_deg=ANGLES,
            radius_mm=POSE_RADIUS_MM,
            ray_len=POSE_RAY_LEN,
            ray_every=POSE_RAY_EVERY,
        )

    # 중앙 슬라이스(원본)
    if SAVE_MID_SLICES:
        save_mid_slices(img, outdir, tag="orig_LPS")

    # --- DRR 생성 (axis sweep) ---
    for ax_name in ROT_AXES:
        for ang in ANGLES:
            img_r = rotate_physical_axis(img, angle_deg=ang, axis=ax_name)

            if SAVE_MID_SLICES:
                save_mid_slices(img_r, outdir, tag=f"rot{ax_name}_{ang:03d}")

            proj = drr_project(img_r, mode=DRR_MODE)
            u8 = normalize_to_uint(proj)
            save_png(u8, os.path.join(outdir, f"drr_{ax_name}_{ang:03d}.png"))

    # --- Skull-like view set (옵션) ---
    if RUN_SKULL_VIEWS:
        skull_dir = os.path.join(outdir, "skull_views")
        _ensure_dir(skull_dir)

        # pose plot (named)
        pose_named_png = os.path.join(skull_dir, "pose_pointcloud_named_views.png")
        save_pose_pointcloud_named_views_png(
            out_path=pose_named_png,
            center_mm=center_mm,
            views=VIEWS,
            radius_mm=SKULL_POSE_RADIUS_MM,
            ray_len=POSE_RAY_LEN,
        )

        # DRR per view
        for name, (rx, ry, rz) in VIEWS.items():
            img_v = rotate_physical_euler(img, rx, ry, rz)
            proj = drr_project(img_v, mode=DRR_MODE)
            u8 = normalize_to_uint(proj)
            save_png(u8, os.path.join(skull_dir, f"drr_view_{name}.png"))


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
