import argparse
import os
from pathlib import Path

from tqdm import tqdm
from PIL import Image

import torch
from facenet_pytorch import MTCNN


def init_mtcnn(device: str = "cuda", image_size: int = 224, min_face_size: int = 40) -> MTCNN:
    """
    MTCNN 초기화
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        device = "cpu"

    mtcnn = MTCNN(
        image_size=image_size,
        margin=0,
        min_face_size=min_face_size,
        keep_all=True,  # 여러 얼굴 중 가장 큰 것만 선택
        device=device,
        post_process=False,  # 여기서는 우리가 직접 crop할 거라 bbox만 사용
    )
    return mtcnn


def detect_and_crop_faces(
    mtcnn: MTCNN,
    img_path: Path,
    out_path: Path,
    image_size: int = 224,
) -> bool:
    """
    단일 이미지에서 얼굴 탐지 후 가장 큰 얼굴을 crop & resize 해서 저장.
    성공 시 True, 실패 시 False 반환.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open image {img_path}: {e}")
        return False

    # facenet_pytorch MTCNN은 directly __call__ 시 face crop 이미지를 반환할 수 있지만,
    # 여기서는 bbox를 직접 쓰는 방식으로 간다 (control을 위해).
    boxes, _ = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        # 얼굴 없음
        return False

    # 가장 큰 얼굴 선택 (bbox: [x1, y1, x2, y2])
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    max_idx = int(max(range(len(areas)), key=lambda i: areas[i]))
    x1, y1, x2, y2 = boxes[max_idx]

    # float → int, boundary clipping
    w, h = img.size
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return False

    face = img.crop((x1, y1, x2, y2))
    face = face.resize((image_size, image_size), Image.BILINEAR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        face.save(out_path)
    except Exception as e:
        print(f"[WARN] Failed to save face to {out_path}: {e}")
        return False

    return True


def process_split(
    split_name: str,
    frames_root: Path,
    faces_root: Path,
    mtcnn: MTCNN,
    image_size: int = 224,
) -> None:
    """
    frames/<split>/ 아래의 모든 video_id 디렉토리에서 얼굴 crop 생성.

    frames_root: data/celebdf/frames/real 또는 fake
    faces_root : data/celebdf/faces/real 또는 fake
    """
    if not frames_root.exists():
        print(f"[WARN] frames_root not found: {frames_root}")
        return

    video_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])
    print(f"[INFO] Processing split='{split_name}', video_dirs={len(video_dirs)}")

    for vd in tqdm(video_dirs, desc=f"{split_name} videos"):
        video_id = vd.name
        out_video_dir = faces_root / video_id

        # 이미 어느 정도 처리되어 있으면 스킵해도 됨
        existing_faces = list(out_video_dir.glob("*.png"))
        if len(existing_faces) > 0:
            # 필요하면 여기서 continue를 제거하고 덮어쓰기 가능
            continue

        frame_paths = sorted(vd.glob("*.png"))
        if len(frame_paths) == 0:
            continue

        saved_count = 0
        for fp in frame_paths:
            # 출력 파일명: frame 이름 그대로 사용
            out_path = out_video_dir / fp.name
            ok = detect_and_crop_faces(
                mtcnn=mtcnn,
                img_path=fp,
                out_path=out_path,
                image_size=image_size,
            )
            if ok:
                saved_count += 1

        if saved_count == 0:
            # 이 video_id에서는 얼굴을 전혀 찾지 못함
            print(f"[INFO] No faces found for video: {vd}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Celeb-DF (v2) frames → faces (face crop) 변환 스크립트"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=None,
        help="Deepfake_Detection 프로젝트 루트 경로. "
             "기본값: 이 스크립트 기준 상위 상위 디렉토리",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="출력 얼굴 이미지 크기 (정사각형)",
    )
    parser.add_argument(
        "--min_face_size",
        type=int,
        default=40,
        help="MTCNN에서 고려할 최소 얼굴 크기 (pixel 기준)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="MTCNN 실행 디바이스: 'cuda' 또는 'cpu'",
    )
    parser.add_argument(
        "--process_real",
        action="store_true",
        help="frames/real → faces/real 처리",
    )
    parser.add_argument(
        "--process_fake",
        action="store_true",
        help="frames/fake → faces/fake 처리",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # project_root 기본값: src/ 의 상위 디렉토리 (Deepfake_Detection/)
    if args.project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    else:
        project_root = Path(args.project_root).resolve()

    data_root = project_root / "data" / "celebdf"
    frames_root = data_root / "frames"
    faces_root = data_root / "faces"

    frames_real = frames_root / "real"
    frames_fake = frames_root / "fake"

    faces_real = faces_root / "real"
    faces_fake = faces_root / "fake"

    mtcnn = init_mtcnn(
        device=args.device,
        image_size=args.image_size,
        min_face_size=args.min_face_size,
    )

    # 처리할 split 선택 (둘 다 선택 안 했으면 real+fake 모두 처리)
    process_real = args.process_real or (not args.process_fake)
    process_fake = args.process_fake or (not args.process_real)

    if process_real:
        process_split(
            split_name="real",
            frames_root=frames_real,
            faces_root=faces_real,
            mtcnn=mtcnn,
            image_size=args.image_size,
        )

    if process_fake:
        process_split(
            split_name="fake",
            frames_root=frames_fake,
            faces_root=faces_fake,
            mtcnn=mtcnn,
            image_size=args.image_size,
        )

    print("[DONE] Face cropping finished.")


if __name__ == "__main__":
    main()
