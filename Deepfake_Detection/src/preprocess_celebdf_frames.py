import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    max_frames: int = 32,
) -> int:
    """
    하나의 비디오에서 최대 max_frames 개의 프레임을 균등 간격으로 추출.
    이미 out_dir 에 파일이 있으면 스킵.
    반환값: 실제로 저장한 프레임 수
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 이미 프레임이 존재하면 스킵 (idempotent)
    existing = list(out_dir.glob("*.png"))
    if len(existing) >= max_frames:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print(f"[WARN] Invalid frame_count ({frame_count}) for {video_path}")
        cap.release()
        return 0

    # 추출할 프레임 인덱스 선택 (0 ~ frame_count-1)
    # 균등 간격으로 max_frames 개까지
    indices = []
    if frame_count <= max_frames:
        indices = list(range(frame_count))
    else:
        step = frame_count / max_frames
        indices = [int(i * step) for i in range(max_frames)]

    saved = 0
    current_idx = 0
    target_set = set(indices)

    video_id = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in target_set:
            # BGR -> RGB 로 굳이 바꿀 필요는 없지만,
            # 일관성 위해 BGR 그대로 저장 (cv2.imwrite)
            frame_name = f"{video_id}_frame{current_idx:06d}.png"
            out_path = out_dir / frame_name
            cv2.imwrite(str(out_path), frame)
            saved += 1

            # 모두 저장했으면 종료
            if saved >= len(indices):
                break

        current_idx += 1

    cap.release()
    return saved


def process_split(
    split_name: str,
    in_root: Path,
    out_root: Path,
    max_frames: int = 32,
) -> None:
    """
    Real 또는 Fake 그룹에 대해 모든 비디오에 대해 프레임 추출.
    split_name: "real" or "fake"
    in_root: raw_videos 하위의 입력 루트 (ex: Celeb-real, Celeb-synthesis)
    out_root: frames 하위의 출력 루트 (ex: frames/real, frames/fake)
    """
    video_paths = sorted(in_root.glob("*.mp4"))

    print(f"[INFO] Processing split='{split_name}', videos={len(video_paths)}")
    out_root.mkdir(parents=True, exist_ok=True)

    for vp in tqdm(video_paths, desc=f"{split_name} videos"):
        video_id = vp.stem
        video_out_dir = out_root / video_id
        saved = extract_frames_from_video(
            video_path=vp,
            out_dir=video_out_dir,
            max_frames=max_frames,
        )
        if saved == 0:
            # 이미 있는 경우 또는 실패
            continue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Celeb-DF (v2) raw_videos → frames 변환 스크립트"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=None,
        help="Deepfake_Detection 프로젝트 루트 경로. "
             "기본값: 이 스크립트 기준 상위 상위 디렉토리",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=32,
        help="비디오당 최대 추출 프레임 수 (균등 샘플링)",
    )
    parser.add_argument(
        "--process_real",
        action="store_true",
        help="Celeb-real (real) 비디오 처리",
    )
    parser.add_argument(
        "--process_fake",
        action="store_true",
        help="Celeb-synthesis (fake) 비디오 처리",
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
    raw_root = data_root / "raw_videos"
    frames_root = data_root / "frames"

    celeb_real_dir = raw_root / "Celeb-real"
    celeb_fake_dir = raw_root / "Celeb-synthesis"

    real_out_root = frames_root / "real"
    fake_out_root = frames_root / "fake"

    if not celeb_real_dir.exists():
        print(f"[WARN] Celeb-real dir not found: {celeb_real_dir}")
    if not celeb_fake_dir.exists():
        print(f"[WARN] Celeb-synthesis dir not found: {celeb_fake_dir}")

    # 처리할 split 선택 (둘 다 선택 안 했으면 real+fake 모두 처리)
    process_real = args.process_real or (not args.process_fake)
    process_fake = args.process_fake or (not args.process_real)

    if process_real and celeb_real_dir.exists():
        process_split(
            split_name="real",
            in_root=celeb_real_dir,
            out_root=real_out_root,
            max_frames=args.max_frames,
        )

    if process_fake and celeb_fake_dir.exists():
        process_split(
            split_name="fake",
            in_root=celeb_fake_dir,
            out_root=fake_out_root,
            max_frames=args.max_frames,
        )

    print("[DONE] Frame extraction finished.")


if __name__ == "__main__":
    main()
