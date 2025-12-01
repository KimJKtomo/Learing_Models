#!/bin/bash

# =========================================
# Run CAM (ConvNeXt V2) for a single MVTec category
# =========================================
# 사용 예:
#   chmod +x run_one_mvtec_cam.sh
#   ./run_one_mvtec_cam.sh bottle
# =========================================

if [ $# -lt 1 ]; then
    echo "Usage: $0 <mvtec_class>"
    echo "Example: $0 bottle"
    exit 1
fi

CLS=$1

GPU=0
MODEL="convnextv2_tiny.fcmae_ft_in22k_in1k"
IMG_SIZE=224
CAM_TYPE="gradcampp"
WEIGHT_PATH=""            # 비우면 ImageNet/IN22k+IN1k pretrained 사용
BATCH_SIZE=16
NUM_WORKERS=4

BASE_DIR="./mvtec"
OUT_DIR="./results_cam_convnextv2_single"

echo "======================================"
echo " [ConvNeXt V2] Running CAM for: ${CLS}"
echo "======================================"

python test_cam_convnextv2.py \
    --gpu ${GPU} \
    --model "${MODEL}" \
    --data_dir "${BASE_DIR}/${CLS}" \
    --save_dir "${OUT_DIR}/${CLS}" \
    --img_size ${IMG_SIZE} \
    --cam_type ${CAM_TYPE} \
    --weight_path "${WEIGHT_PATH}" \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS}

echo "======================================"
echo " Done. Saved to: ${OUT_DIR}/${CLS}"
echo "======================================"
