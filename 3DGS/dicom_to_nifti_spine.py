#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DICOM 시리즈 → NIfTI 변환 (Spine CT용 샘플)
"""

import os
import argparse
import SimpleITK as sitk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_root", type=str, required=True,
                        help="여러 DICOM 시리즈가 들어있는 상위 폴더")
    parser.add_argument("--outdir", type=str, required=True,
                        help="NIfTI를 저장할 폴더")
    return parser.parse_args()


def is_dicom_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    files = os.listdir(path)
    files = [f for f in files if not f.startswith(".")]
    if not files:
        return False
    # 간단 체크: 확장자 없거나 .dcm 같은 파일 존재 시 DICOM으로 가정
    for f in files:
        if f.lower().endswith(".dcm") or "." not in f:
            return True
    return False


def convert_series(dicom_dir: str, out_nii: str):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        print(f"[WARN] No series in {dicom_dir}")
        return
    series_files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(series_files)
    img = reader.Read()
    os.makedirs(os.path.dirname(out_nii), exist_ok=True)
    sitk.WriteImage(img, out_nii)
    print(f"[SAVE] {out_nii} (size={img.GetSize()}, spacing={img.GetSpacing()})")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for root, dirs, files in os.walk(args.dicom_root):
        # DICOM 시리즈 폴더인지 판단
        if is_dicom_dir(root):
            case_id = os.path.basename(root.rstrip("/"))
            out_nii = os.path.join(args.outdir, f"{case_id}.nii.gz")
            convert_series(root, out_nii)


if __name__ == "__main__":
    main()
