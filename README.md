# Learning_Models

컴퓨터 비전 SOTA 모델들을 실습하고,  
산업군(제조, 의료, 일반 도메인)에 적용해보는 개인 연구 레포지토리.

## 디렉토리 구조

- `CNN/`  
  - ConvNeXtV2 기반 분류 모델 실험  
  - Tiny-ImageNet / MVTec AD2 등 일반 이미지 및 산업군 데이터셋 실험 포함
- `DETR/`  
  - DETR 계열 Object Detection 실험 예정
- `DIFF/`  
  - Diffusion 기반 생성/변환 모델 실습 예정
- `GAN/`  
  - GAN 계열 생성 모델 실습 예정
- `Transformer/`  
  - Vision Transformer, Swin Transformer 등 일반 Transformer 기반 CV 실험
- `YOLO/`  
  - YOLO 계열 Detection 실험 (v5, v8, v9 등) 정리 예정

## CNN/MVTec AD2 서브프로젝트

- `CNN/mvtec_exp/`  
  - MVTec AD2 데이터를 이용한 산업 이상 탐지(Anomaly Detection) 실험  
  - ConvNeXtV2 Tiny + timm + Grad-CAM XAI 파이프라인  
  - 세부 내용은 `CNN/mvtec_exp/README.md` 참고

