# 📘 Learing Models

해당 디렉토리는 **Learing Models** 딥러닝 모델 실험을 정리한
공간이다.\
모델 구조, 학습 스크립트, 파라미터 설정, 전처리, 평가 지표,
시각화(GradCAM·ROC·Confusion Matrix 등)를 포함한다.

------------------------------------------------------------------------

## 📁 Directory Structure

    {MODEL_NAME}/
     ├── README.md
     ├── configs/
     ├── scripts/
     ├── models/
     ├── datasets/
     └── results/

------------------------------------------------------------------------

## 📌 1. Overview

-   모델 목적
-   Task: Classification / Detection / Segmentation / Generation
-   선택 이유

예시: Swin Transformer 기반 X-ray 분류 모델.

------------------------------------------------------------------------

## 📌 2. Requirements

### Python

    python >= 3.9

### Dependencies

    pip install -r requirements.txt

주요 패키지: - torch - torchvision - timm - opencv-python -
albumentations - matplotlib - scikit-learn - mlflow (선택)

------------------------------------------------------------------------

## 📌 3. Dataset Structure

### Classification

    dataset/
     ├── train/
     │    ├── class0/
     │    └── class1/
     └── val/
          ├── class0/
          └── class1/

### Detection (YOLO)

    dataset/
     ├── images/train, val
     └── labels/train, val

------------------------------------------------------------------------

## 📌 4. Configuration

    model: swin_tiny
    img_size: 384
    batch_size: 32
    epochs: 50
    optimizer:
      name: adamw
      lr: 1e-4
      weight_decay: 0.05
    scheduler:
      name: cosine
      warmup_epochs: 3

------------------------------------------------------------------------

## 📌 5. Training

    python scripts/train.py --cfg configs/train_config.yaml

옵션: \| 옵션 \| 설명 \| \|------\|------\| \| --cfg \| 설정 파일 \| \|
--resume \| 이어하기 \| \| --device \| GPU 선택 \|

------------------------------------------------------------------------

## 📌 6. Evaluation

    python scripts/eval.py --weights results/weights/best.pt

출력: - Accuracy / F1 / AUROC - Confusion Matrix - ROC Curve (Youden
Index) - Classification Report - CSV 저장

------------------------------------------------------------------------

## 📌 7. Inference

    python scripts/infer.py --img path.jpg --weights best.pt
    python scripts/infer.py --folder input_dir/ --weights best.pt

Output: - 이미지 + heatmap - CSV 로그 - Summary JSON

------------------------------------------------------------------------

## 📌 8. Visualization

-   Grad-CAM / Grad-CAM++
-   Attention Rollout
-   ROC Curve
-   Confusion Matrix
-   Loss Curve

Plots 저장 위치:

    results/plots/

------------------------------------------------------------------------

## 📌 9. Logging

MLflow:

    mlflow ui --port 5000

TensorBoard:

    tensorboard --logdir results/logs

------------------------------------------------------------------------

## 📌 10. Checkpoints

    results/
     ├── weights/
     │     ├── best.pt
     │     └── last.pt
     ├── logs/
     └── plots/

------------------------------------------------------------------------

## 📌 11. Future Work

-   Backbone 확장
-   Multimodal 학습
-   Ensemble
-   Dataset 확장

------------------------------------------------------------------------

## 📌 12. References

-   논문 링크
-   GitHub repo
