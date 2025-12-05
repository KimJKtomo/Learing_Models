import argparse
import random
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple

import pandas as pd


def find_test_list_path(data_root: Path) -> Optional[Path]:
    """
    Celeb-DF List_of_testing_videos.txt 경로 자동 탐색.
    우선순위:
      1) data/celebdf/raw_videos/List_of_testing_videos.txt
      2) data/celebdf/raw_videos/metadata/List_of_testing_videos.txt
    """
    candidates = [
        data_root / "raw_videos" / "List_of_testing_videos.txt",
        data_root / "raw_videos" / "metadata" / "List_of_testing_videos.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_test_video_ids(test_list_path: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    List_of_testing_videos.txt 를 읽어서
    real / fake / youtube-real video_id 집합을 반환.
    """
    test_real_ids: Set[str] = set()
    test_fake_ids: Set[str] = set()
    test_youtube_ids: Set[str] = set()

    with test_list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) == 1:
                rel_path = parts[0]
            else:
                # 보통 "1 Celeb-real/..." 또는 "0 Celeb-synthesis/..." 형식이므로
                rel_path = parts[-1]

            rel_path = rel_path.strip()
            if not rel_path:
                continue

            rel = Path(rel_path)
            video_id = rel.stem
            top = rel.parts[0].lower()

            if "celeb-real" in top:
                test_real_ids.add(video_id)
            elif "celeb-synthesis" in top:
                test_fake_ids.add(video_id)
            elif "youtube-real" in top:
                test_real_ids.add(video_id)
                test_youtube_ids.add(video_id)
            else:
                # 예외 케이스는 일단 무시
                pass

    return test_real_ids, test_fake_ids, test_youtube_ids


def collect_faces_per_video(faces_root: Path) -> Dict[str, List[Path]]:
    """
    faces_root (예: data/celebdf/faces/real) 하위에서
    video_id -> [image_path, ...] 매핑을 수집.
    """
    mapping: Dict[str, List[Path]] = {}
    if not faces_root.exists():
        return mapping

    for vd in faces_root.iterdir():
        if not vd.is_dir():
            continue
        video_id = vd.name
        imgs = sorted(vd.glob("*.png"))
        if len(imgs) == 0:
            continue
        mapping[video_id] = imgs
    return mapping


def split_train_val_video_ids(
    trainval_ids: List[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """
    비테스트 video_id 리스트를 train / val 로 random split.
    """
    rnd = random.Random(seed)
    ids = list(trainval_ids)
    rnd.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_ratio))
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    return set(train_ids), set(val_ids)


def build_rows_for_split(
    split_name: str,
    real_ids: Set[str],
    fake_ids: Set[str],
    faces_real_map: Dict[str, List[Path]],
    faces_fake_map: Dict[str, List[Path]],
    project_root: Path,
) -> List[dict]:
    """
    주어진 split (train/val/test)에 대해 CSV row 리스트 생성.
    """
    rows: List[dict] = []

    # real
    for vid in sorted(real_ids):
        img_paths = faces_real_map.get(vid, [])
        for ip in img_paths:
            rel_path = ip.relative_to(project_root)
            rows.append(
                {
                    "image_path": str(rel_path),
                    "label": 0,
                    "video_id": vid,
                    "split": split_name,
                    "source": "real",
                }
            )

    # fake
    for vid in sorted(fake_ids):
        img_paths = faces_fake_map.get(vid, [])
        for ip in img_paths:
            rel_path = ip.relative_to(project_root)
            rows.append(
                {
                    "image_path": str(rel_path),
                    "label": 1,
                    "video_id": vid,
                    "split": split_name,
                    "source": "fake",
                }
            )

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Celeb-DF (v2) faces → train/val/test CSV 생성 스크립트"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=None,
        help="Deepfake_Detection 프로젝트 루트 경로. "
             "기본값: 이 스크립트 기준 상위 상위 디렉토리",
    )
    parser.add_argument(
        "--test_list_path",
        type=str,
        default=None,
        help="List_of_testing_videos.txt 직접 지정 경로. "
             "미지정 시 자동 탐색",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="train/val split ratio (video-level, train 비율)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="train/val split용 random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # project_root 설정
    if args.project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    else:
        project_root = Path(args.project_root).resolve()

    data_root = project_root / "data" / "celebdf"
    faces_root = data_root / "faces"
    csv_root = data_root / "csv"
    csv_root.mkdir(parents=True, exist_ok=True)

    faces_real_root = faces_root / "real"
    faces_fake_root = faces_root / "fake"

    # faces → video_id -> image list 매핑 생성
    faces_real_map = collect_faces_per_video(faces_real_root)
    faces_fake_map = collect_faces_per_video(faces_fake_root)

    all_real_ids: Set[str] = set(faces_real_map.keys())
    all_fake_ids: Set[str] = set(faces_fake_map.keys())

    print(f"[INFO] Total real video_ids with faces: {len(all_real_ids)}")
    print(f"[INFO] Total fake video_ids with faces: {len(all_fake_ids)}")

    # test list 경로 결정
    if args.test_list_path is not None:
        test_list_path = Path(args.test_list_path).resolve()
    else:
        test_list_path = find_test_list_path(data_root)

    if test_list_path is None or not test_list_path.exists():
        raise FileNotFoundError(
            "List_of_testing_videos.txt 를 찾지 못했습니다. "
            "--test_list_path 로 직접 지정하거나, "
            "data/celebdf/raw_videos/ 또는 raw_videos/metadata/ 아래에 두세요."
        )

    print(f"[INFO] Using test list: {test_list_path}")

    test_real_ids_raw, test_fake_ids_raw, test_youtube_ids = load_test_video_ids(
        test_list_path
    )

    # 실제 faces 에 존재하는 video_id 와 교집합
    test_real_ids = all_real_ids & test_real_ids_raw
    test_fake_ids = all_fake_ids & test_fake_ids_raw

    print(f"[INFO] Test real video_ids (with faces): {len(test_real_ids)}")
    print(f"[INFO] Test fake video_ids (with faces): {len(test_fake_ids)}")

    # train/val 후보: test 제외
    trainval_real_ids = sorted(all_real_ids - test_real_ids)
    trainval_fake_ids = sorted(all_fake_ids - test_fake_ids)

    print(f"[INFO] TrainVal real video_ids: {len(trainval_real_ids)}")
    print(f"[INFO] TrainVal fake video_ids: {len(trainval_fake_ids)}")

    # train/val split (class-wise)
    train_real_ids, val_real_ids = split_train_val_video_ids(
        trainval_real_ids, train_ratio=args.train_ratio, seed=args.seed
    )
    train_fake_ids, val_fake_ids = split_train_val_video_ids(
        trainval_fake_ids, train_ratio=args.train_ratio, seed=args.seed + 1
    )

    print(f"[INFO] Train real video_ids: {len(train_real_ids)}")
    print(f"[INFO] Train fake video_ids: {len(train_fake_ids)}")
    print(f"[INFO] Val real video_ids: {len(val_real_ids)}")
    print(f"[INFO] Val fake video_ids: {len(val_fake_ids)}")

    # 각 split별 row 생성
    train_rows = build_rows_for_split(
        split_name="train",
        real_ids=train_real_ids,
        fake_ids=train_fake_ids,
        faces_real_map=faces_real_map,
        faces_fake_map=faces_fake_map,
        project_root=project_root,
    )
    val_rows = build_rows_for_split(
        split_name="val",
        real_ids=val_real_ids,
        fake_ids=val_fake_ids,
        faces_real_map=faces_real_map,
        faces_fake_map=faces_fake_map,
        project_root=project_root,
    )
    test_rows = build_rows_for_split(
        split_name="test",
        real_ids=test_real_ids,
        fake_ids=test_fake_ids,
        faces_real_map=faces_real_map,
        faces_fake_map=faces_fake_map,
        project_root=project_root,
    )

    print(f"[INFO] Num train rows: {len(train_rows)}")
    print(f"[INFO] Num val rows  : {len(val_rows)}")
    print(f"[INFO] Num test rows : {len(test_rows)}")

    # CSV 저장
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)

    train_csv = csv_root / "train.csv"
    val_csv = csv_root / "val.csv"
    test_csv = csv_root / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"[SAVE] {train_csv}")
    print(f"[SAVE] {val_csv}")
    print(f"[SAVE] {test_csv}")
    print("[DONE] Celeb-DF CSV build finished.")


if __name__ == "__main__":
    main()
