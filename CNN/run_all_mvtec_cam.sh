#!/bin/bash

# =========================================
# Run CAM (ConvNeXt V2) for all MVTec categories
# =========================================
# 사용 전:
#   chmod +x run_all_mvtec_cam.sh
#   ./run_all_mvtec_cam.sh
# =========================================

GPU=0

# huggingface/pytorch-image-models(timm) 모델명
# convnextv2_tiny.fcmae_ft_in22k_in1k  / base / large 등으로 변경 가능
MODEL="convnextv2_tiny.fcmae_ft_in22k_in1k"

IMG_SIZE=224
CAM_TYPE="gradcampp"          # gradcam / gradcampp / eigencam
WEIGHT_PATH=""                # 따로 학습한 ckpt 쓰면 여기에 경로; 공백이면 timm pretrained 사용
BATCH_SIZE=16
NUM_WORKERS=4

BASE_DIR="./mvtec"            # MVTec AD 데이터 루트
OUT_DIR="./results_cam_convnextv2"   # CAM 결과 저장 루트

CLASSES=(
    "bottle"
    "cable"
    "capsule"
    "carpet"
    "grid"
    "hazelnut"
    "leather"
    "metal_nut"
    "pill"
    "screw"
    "tile"
    "toothbrush"
    "transistor"
    "wood"
    "zipper"
)

for cls in "${CLASSES[@]}"; do
    echo "======================================"
    echo "  [ConvNeXt V2] Running CAM for: ${cls}"
    echo "======================================"

    python test_cam_convnextv2.py \
        --gpu ${GPU} \
        --model "${MODEL}" \
        --data_dir "${BASE_DIR}/${cls}" \
        --save_dir "${OUT_DIR}/${cls}" \
        --img_size ${IMG_SIZE} \
        --cam_type ${CAM_TYPE} \
        --weight_path "${WEIGHT_PATH}" \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS}

    echo "[Done] ${cls}"
    echo
done

echo "======================================"
echo " All classes processed."
echo " Results saved under: ${OUT_DIR}"
echo "======================================"
