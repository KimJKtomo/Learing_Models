# CT 기반 멀티뷰 X-ray 증거 융합을 통한 두개골 골절 위치화

본 저장소는 **Brain CT와 임상 Skull X-ray 4-view를 결합하여**, 멀티뷰 기하 합의(geometric consensus)를 통해 두개골 골절 병변을 **확정 및 위치화(BBOX + Heatmap)** 하는 전체 워크플로우를 정리한 프로젝트 문서이다.

---

## 1. 문제 정의 (Problem Statement)

### 입력 데이터
- 환자별 **Brain CT**
- 동일 환자의 **임상 Skull X-ray (4 View)**
  - Right AP
  - Left AP
  - 정면(AP/PA)
  - Towne

### 출력 결과
- **View별 결과**: BBOX + Heatmap overlay
- **최종 결과**:
  - 멀티뷰 합의 기반 병변 확정
  - 3D Heatmap (Skull surface 상)
  - 3D Bounding Box (OBB)

### 해결해야 할 문제점
- Skull X-ray는 **투과 영상 특성으로 인해 구조적 겹침(overlap)이 매우 심함**
- 단일 view 기반 판단은 **위양성(False Positive)** 발생률이 높음
- 임상 환경에서는 **2D X-ray 기반 해석**이 필수적임

---

## 2. 핵심 설계 철학

1. **병변은 2D에서 관측한다**  
   - 골절선 및 미세 균열은 생성(hallucination) 대상이 아님

2. **3D는 복원이 아니라 ‘통합 공간’이다**  
   - 2D 병변 증거를 합의(consensus)시키기 위한 좌표계

3. **CT는 정답 기하(prior)이다**  
   - DRR을 통해 X-ray와 3D 기하를 정확히 연결

4. **멀티뷰 융합은 score fusion이 아니라 기하 기반 fusion이다**

---

## 3. End-to-End Workflow

### Step 1. CT → Skull Surface 생성 (3D 지지체)
- Brain CT에서 bone segmentation 수행
- 내부 volume 제거
- **연결된 skull surface(mesh 또는 point cloud)**만 유지

**역할**
- 병변이 존재할 수 있는 3D 공간 정의
- 2D 병변 증거를 올릴 기준면 제공

---

### Step 2. 임상 프로토콜 기반 DRR 생성 (정답 기하 정의)

실제 임상에서 사용되는 두개골 촬영 프로토콜을 기반으로 DRR 생성:
- 정면(AP/PA)
- Right / Left AP (또는 oblique)
- Towne (예: OML 기준 30° caudad)

각 DRR 생성 시 다음 정보를 함께 저장:
- Intrinsic 행렬 **K**
- Extrinsic 파라미터 **R | t**
- Detector geometry, SID

**의미**
- DRR은 단순한 synthetic 이미지가 아니라  
  **X-ray 촬영 기하의 정답 정의**
- 이후 모든 2D → 3D 매핑은 기하학적 연산으로 귀결됨

---

### Step 3. DRR 기반 선학습 → 임상 X-ray 적응

- **DRR 데이터 학습 목적**
  - 해부학적 구조 prior 학습
  - 멀티뷰 간 기하 일관성 확보

- **임상 X-ray 적응**
  - intensity, noise, scatter, 장비별 차이 보정
  - 목표는 “DRR처럼 보이게”가 아니라  
    **DRR 기하 위에서 해석 가능하도록 만드는 것**

---

### Step 4. View-wise 병변 증거 추출 (2D)

각 view마다 독립적으로 수행:

- **Detector**
  - 병변 후보 BBOX 및 confidence 산출

- **Heatmap 생성**
  - Grad-CAM++ 또는 weakly-supervised localization
  - 픽셀 단위 병변 evidence 생성

**중요한 제약 조건**
- 병변 검출은 오직 2D 단계에서만 수행
- 3D 공간에서 병변을 생성하지 않음

---

### Step 5. 2D Evidence → 3D 공간 통합 (핵심 단계)

- 각 view의 heatmap을
- 해당 view의 **K, R|t**를 사용하여
- Skull surface로 투영

개념적 수식:

X ∈ Surface  
 x_v = π(K_v [R_v | t_v] X)  
 H_3D(X) = Σ_v w_v · H_v(x_v)

가중치 **w_v** 구성 요소:
- detector confidence
- view별 신뢰도
- surface normal–view direction cosine

**결과**
- 겹침으로 인한 위양성(FP)은 상쇄
- 실제 병변은 여러 view에서 **동일한 3D 위치로 수렴**

---

### Step 6. 3D 병변 확정 및 위치화

- 3D heat thresholding
- Surface 기반 클러스터링
- 대표 cluster 선택
- **3D Bounding Box 생성 (OBB 권장)**

**최종 판정 기준**
- 단일 view 양성 ❌
- **멀티뷰 기하 합의 ⭕**

---

### Step 7. 3D 결과 → View-wise 재투영 (임상 전달 단계)

- 3D heatmap 및 3D BBOX를
- 각 view로 재투영

**최종 산출물**
- Original X-ray + BBOX + Heat overlay

**해석 흐름**
- AI는 3D 공간에서 판단
- **의사는 2D 영상에서 최종 확인**

---

## 4. 3D Gaussian Splatting(3DGS)의 역할

- 3DGS는 **병변 생성 모델이 아님**
- 역할:
  - 멀티뷰 병변 증거를 담는 고속 3D 표현
  - 빠른 재투영 및 시각화 엔진

X-ray 특성을 고려하여 **Radiative / DRR-style projection**과 결합

→ 본 연구는 3DGS를
> **Evidence Fusion Space**로 활용하는 새로운 관점을 제시

---

## 5. 기존 방법 대비 차별점

| 항목 | 기존 방식 | 본 Workflow |
|------|-----------|-------------|
| Single-view FP | 높음 | 낮음 |
| 멀티뷰 처리 | score fusion | 기하 기반 fusion |
| 3D 활용 | 없음 / 복원 | 증거 통합 공간 |
| 설명 가능성 | 낮음 | 높음 |
| 임상 적합성 | 제한적 | 높음 |

---

## 6. 핵심 기여 요약

- CT 정의 기하 기반 멀티뷰 X-ray 병변 합의
- 2D 병변 증거 보존 + 3D 공간 통합
- Skull 겹침 구조에 특화된 위양성 감소 전략
- 임상 2D X-ray workflow와 직접 연결되는 3D 판단 구조

---

## 7. 관련 연구 포지셔닝

- **XraySyn (AAAI 2021)**  
  CT prior와 differentiable projection을 통한 X-ray–3D 연결

- **Radiative / X-ray Gaussian Splatting (ECCV 2024)**  
  X-ray 특화 3D 표현 및 고속 재투영

- **Biplane X-ray 기반 3D 위치 추정 (정형외과 분야)**  
  멀티뷰 X-ray에 공간 정보가 내재되어 있음을 입증

- **Grad-CAM++ / Weakly-supervised localization**  
  Heatmap 기반 병변 evidence의 타당성

---

## 8. 한 문장 요약 (Abstract용)

> *본 연구는 CT로 정의된 기하 정보를 기반으로 멀티뷰 X-ray에서 추출된 2D 병변 증거를 3D 표면 공간으로 통합하여, 영상 겹침 문제를 기하학적 합의를 통해 해결하는 두개골 골절 위치화 프레임워크를 제안한다.*
