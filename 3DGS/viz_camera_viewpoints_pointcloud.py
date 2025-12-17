#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
viz_camera_viewpoints_pointcloud.py

- "카메라 viewpoint가 3D에서 점군처럼 어떻게 배치되는지"만 먼저 보자.
- 카메라 위치(centers)를 점으로 찍고, 각 점에서 center를 향하는 시선(ray)도 짧게 표시.
- pose가 맞는지 감 잡는 용도.

실행:
  python viz_camera_viewpoints_pointcloud.py

필요하면 ROT_AXIS / ANGLES / RADIUS만 바꾸면 됨.
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# 여기만 바꾸면 됨
# =========================
CENTER = np.array([0.0, 0.0, 0.0])         # isocenter(CT center)라고 생각
RADIUS = 800.0                              # mm (보기 좋게 500~1200)
ROT_AXIS = "z"                              # 'x'/'y'/'z' : gantry rotation 축이라고 생각
ANGLES = list(range(0, 360, 10))            # DRR angles랑 맞추면 됨

# 시선(ray) 길이
RAY_LEN = 140.0

# 너무 많으면 시선은 일부만
RAY_EVERY = 6                                # 1이면 전부, 6이면 6개마다 1개
# =========================


def rotmat(axis: str, deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    if axis == "x":
        return np.array([[1, 0, 0],
                         [0, c,-s],
                         [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]], dtype=np.float64)
    return np.array([[c,-s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float64)


def make_orbit_centers(axis: str, angles_deg, center: np.ndarray, radius: float) -> np.ndarray:
    # baseline: +X에서 center를 본다고 가정
    base = center + np.array([radius, 0.0, 0.0], dtype=np.float64)
    centers = []
    for a in angles_deg:
        R = rotmat(axis, a)
        C = center + R @ (base - center)
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


def main():
    cams = make_orbit_centers(ROT_AXIS, ANGLES, CENTER, RADIUS)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Camera viewpoints (pointcloud) | axis={ROT_AXIS} | N={len(ANGLES)}")

    # center
    ax.scatter([CENTER[0]], [CENTER[1]], [CENTER[2]], marker="x")
    ax.text(CENTER[0], CENTER[1], CENTER[2], "  center")

    # camera centers = pointcloud
    ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], s=10)

    # orbit line
    ax.plot(cams[:, 0], cams[:, 1], cams[:, 2])

    # view rays (camera -> center)
    for i in range(0, len(ANGLES), max(1, RAY_EVERY)):
        C = cams[i]
        v = CENTER - C
        v = v / (np.linalg.norm(v) + 1e-8)
        P2 = C + v * RAY_LEN  # 짧게만 표시
        ax.plot([C[0], P2[0]], [C[1], P2[1]], [C[2], P2[2]])
        ax.text(C[0], C[1], C[2], f" {ANGLES[i]}°")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    set_equal_aspect(ax, np.vstack([cams, CENTER[None, :]]))

    plt.show()


if __name__ == "__main__":
    main()
