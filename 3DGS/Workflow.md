# CT-Guided Multi-View X-ray Evidence Fusion for Skull Fracture Localization

본 저장소는 **Brain CT와 임상 Skull X-ray 4-view를 결합하여**, 멀티뷰 기하 합의(geometric consensus)를 통해 두개골 골절 병변을 **확정·위치화(BBOX + Heatmap)** 하는 전체 워크플로우를 정리한 프로젝트 문서이다.

---

## 1. Problem Statement

### Input
- Patient-specific **Brain CT**
- Corresponding **Clinical Skull X-ray (4 Views)**
  - Right AP
  - Left AP
  - Frontal (AP/PA)
  - Towne

### Output
- **View-wise results**: BBOX + Heatmap overlay
- **Final result**: Multi-view consensus–based lesion confirmation with
  - 3D Heatmap on skull surface
  - 3D Bounding Box (OBB)

### Challenges
- Skull X-ray is a **projective and overlapping modality**
- Single-view interpretation suffers from **intrinsically high false positives**
- Clinical decision-making is still performed on **2D X-ray images**

---

## 2. Core Design Philosophy

1. **Lesions are observed in 2D, not generated in 3D**  
   - Fracture lines and fissures must come from actual image evidence

2. **3D is not for reconstruction, but for fusion**  
   - 3D space serves as a consensus domain for multi-view evidence

3. **CT provides ground-truth geometry**  
   - DRR bridges CT and X-ray via exact projection geometry

4. **Multi-view fusion is geometric, not score-level**

---

## 3. End-to-End Workflow

### Step 1. CT → Skull Surface Generation
- Perform bone segmentation on Brain CT
- Remove internal volume
- Preserve only a **connected skull surface (mesh / point cloud)**

**Purpose**
- Defines the 3D spatial domain where lesions may exist
- Acts as a support for lifting 2D evidence into 3D

---

### Step 2. DRR Generation with Clinical View Protocols

Generate DRRs using **clinically standard skull projections**:
- Frontal (AP/PA)
- Right / Left AP (or oblique)
- Towne (e.g., 30° caudad to OML)

For each DRR, record:
- Intrinsic matrix **K**
- Extrinsic parameters **R | t**
- Detector geometry and SID

**Key Insight**
- DRR is not a synthetic image but a **definition of X-ray acquisition geometry**
- All subsequent 2D→3D mappings become deterministic geometric operations

---

### Step 3. DRR-based Pretraining → Clinical X-ray Adaptation

- **DRR domain**
  - Learn anatomical structure priors
  - Enforce multi-view geometric consistency

- **Clinical X-ray domain**
  - Adapt to intensity, noise, scatter, and device-specific variations

Goal is not to mimic DRR appearance, but to ensure **clinical X-rays are interpretable within DRR-defined geometry**

---

### Step 4. View-wise Lesion Evidence Extraction (2D)

Performed independently for each view:

- **Detector**
  - Outputs lesion candidate BBOX with confidence

- **Heatmap generator**
  - Grad-CAM++ or weakly-supervised localization
  - Produces pixel-level lesion evidence

**Important Constraint**
- Lesions are detected only in 2D
- No lesion synthesis is performed in 3D

---

### Step 5. 2D Evidence → 3D Geometric Fusion (Core Step)

For each view:
- Project 2D heatmaps onto the skull surface using corresponding **K, R|t**

Conceptually:

X ∈ Surface  
 x_v = π(K_v [R_v | t_v] X)  
 H_3D(X) = Σ_v w_v · H_v(x_v)

Where weights **w_v** may include:
- Detector confidence
- View-specific reliability
- Surface normal–view direction cosine

**Outcome**
- Overlap-induced false positives are suppressed
- True lesions converge to consistent 3D surface locations

---

### Step 6. 3D Lesion Confirmation and Localization

- Threshold 3D heatmap
- Perform surface-based clustering
- Select representative cluster
- Generate **3D Bounding Box (Oriented BBox recommended)**

**Final decision criterion**
- Single-view positive ❌
- **Multi-view geometric consensus ⭕**

---

### Step 7. 3D → 2D Reprojection for Clinical Use

- Reproject 3D heatmap and 3D BBOX back to each X-ray view

**Final deliverables**
- Original X-ray + BBOX + Heat overlay

**Interpretation flow**
- AI reasons in 3D
- Clinicians validate in 2D

---

## 4. Role of 3D Gaussian Splatting (3DGS)

- 3DGS is **not used to generate lesions**
- It serves as:
  - A fast 3D representation for multi-view evidence fusion
  - A high-speed rendering and reprojection engine

When combined with **radiative / DRR-style projection**, 3DGS enables:
- Efficient multi-view fusion
- Consistent visualization across views

This positions the method as a:

> **3DGS-based Evidence Fusion Space**, not a reconstruction model

---

## 5. Comparison with Conventional Approaches

| Aspect | Conventional | Proposed Workflow |
|------|-------------|------------------|
| Single-view FP | High | Low |
| Multi-view usage | Score fusion | Geometric fusion |
| Role of 3D | None / Reconstruction | Evidence fusion space |
| Explainability | Low | High |
| Clinical compatibility | Limited | High |
---

## 6. Key Contributions

- CT-defined geometry–guided multi-view X-ray lesion consensus
- Preservation of 2D evidence with 3D spatial fusion
- FP reduction tailored to skull overlap characteristics
- Seamless integration with clinical 2D X-ray workflow

---

## 7. Related Work Positioning

- **XraySyn (AAAI 2021)**  
  CT prior + differentiable projection connecting CT and X-ray

- **Radiative / X-ray Gaussian Splatting (ECCV 2024)**  
  X-ray–specific 3D representation and fast rendering

- **Biplane X-ray 3D localization (orthopedic literature)**  
  Demonstrates that multi-view X-ray encodes spatial information

- **Grad-CAM++ / Weakly-supervised localization**  
  Validates heatmap-based lesion evidence

---

## 8. One-sentence Summary (for Abstract)

> *We propose a CT-guided multi-view X-ray evidence fusion framework that localizes skull fractures by projecting view-wise 2D lesion evidence onto a common 3D surface space, resolving projection overlap through geometric consensus rather than image synthes