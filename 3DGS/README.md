# X-ray 기반 3D Reconstruction (X2CT) 사전 연구 프로젝트  
Sparse-view Multi-angle X-ray → AI-based 3D Bone Reconstruction

---

## 📌 1. 프로젝트 개요 (Overview)

본 프로젝트는 **CT 없이 X-ray 다중각도 촬영만으로 CT-유사 3D Bone 모델을 생성**하는 기술  
(X2CT: X-ray to CT Reconstruction)을 연구·검증하는 사전 단계(Pre-study) 프로젝트이다.

기존 CT는 정확한 3D 해부학 정보를 제공하지만:

- 비용이 높고  
- 방사선량이 많으며  
- 소아/청소년에서 위험성이 크고  
- 장비 접근성이 제한됨  

따라서 본 연구에서는 **저선량 X-ray(또는 C-arm multi-view)**를 활용해  
**NeRF / 3D Gaussian Splatting 기반 3D Reconstruction 기술**로  
정형외과·치과·척추 영역에서 CT 촬영을 대체하거나 보완할 방법을 탐색한다.

---

## 🎯 2. 최종 목표 (Goal)

1) Sparse-view X-ray 기반 CT-like Bone 3D Reconstruction 기술의 **실현 가능성 검증**  
2) 필요한 **회전 각도(orbit)와 view 수(view count)** 최적값 도출  
3) Bone geometry 재현 정확도에 대한 **정량 평가 체계 확립**  
4) 장비 도입 전, **CT → DRR 시뮬레이션 기반 사전 검증 수행**  
5) 임상 타겟(정형외과·척추·치과)에서 실제 적용 가능한 시나리오 정립  

---

## 🔍 3. 연구 필요성 (Rationale)

### 기존 CT의 한계
- 불필요한 고선량  
- 비용 부담  
- 소아 환자에서 선량 위험 증가  
- 장비 설치 비용 및 접근성 문제

### X2CT가 해결할 문제
- 저선량 X-ray 수 장만으로 CT 수준 Bone Shape 획득  
- 외래에서도 즉시 3D 제공 가능  
- 소아 척추(측만증) 평가 시 CT를 대체할 가능성  
- 치과 CBCT 대체 가능성 (저선량·저비용)  
- 수술 planning 및 navigation에 활용 가능  

---

## 🏥 4. 임상적 타겟 (Clinical Targets)

### ● 정형외과 골절
- Distal radius, proximal humerus  
- Tibial plateau, ankle, clavicle  
- Pediatric fracture planning  

### ● 척추(Spine)
- Scoliosis 3D curve reconstruction  
- Vertebral rotation / alignment  
- Compression fracture  

### ● 치과 / 교정 / 임플란트
- 악골 구조 3D 복원  
- 매복치 위치  
- 교정 planning  
- CBCT 대체 가능성  

### ● 수술 Planning
- Osteotomy planning  
- Varus/Valgus deformity 평가  
- Screw trajectory planning  

(※ Soft-tissue 재현은 X-ray 물리 특성상 불가능 → Bone-driven 질환에 최적화)

---

## 🛠 5. 기술 구성요소 (Technical Components)

### 5.1 Multi-view X-ray Acquisition
- Isocentric rotation 필수  
- Orbit: **150°~200°**  
- View 수: **16~64 views** 중심  
- Flat-panel detector 기반 geometry 필요  

### 5.2 DRR Simulation (사전 연구 핵심)
CT 오픈데이터 활용 → DRR 생성:

- Siddon ray-tracing  
- Distance-driven projection  
- Plastimatch / TIGRE / RTK 기반 DRR tool 사용 가능  

장비 없이도 reconstruction pipeline 검증 가능.

### 5.3 Reconstruction Pipeline
1. Multi-view X-ray 입력  
2. Camera pose alignment  
3. NeRF / 3D Gaussian Splatting 기반 volumetric reconstruction  
4. Bone surface extraction  
5. 정량적 평가 수행  

---

## 🎛 6. 사전 연구 실험 설계 (Pre-study Experiment Plan)

### 📌 단계 1: CT 데이터 준비
- Spine, extremity, dental CT 사용  
- CT → DRR 생성  
  - Orbit 180°  
  - 16 / 32 / 64 / 128 views  
  - Noise-free / Poisson noise 버전 생성  

### 📌 단계 2: Sparse-view Reconstruction 평가
| 단계 | view 수 | 간격 | 목적 |
|------|---------|-------|--------|
| A | 16 | ~11.25° | feasibility test |
| B | 32 | ~5.6° | 임상 적용 가능성 평가 |
| C | 64 | ~2.8° | high-quality sparse |
| D | 128 | ~1.4° | upper bound 평가 |

### 📌 단계 3: Bone Geometry 정량 평가
- Chamfer Distance  
- Surface Dice Score  
- 3D landmark error  
- Fracture line visibility 평가  

### 📌 단계 4: 최적 parameter 도출
- View 증가 → 품질 증가  
- View 증가 → 방사선량 증가  
→ 의료진이 허용 가능한 trade-off 탐색  

---

## 📡 7. 장비 요구조건 (Hardware Requirements)

아직 장비 도입 전 단계이므로, 사전 연구 후 선택.

**필수 조건:**
- 150°~200° isocentric rotation  
- Geometry calibration 제공  
- FPD 기반  
- 최소 16~64 shot 지원  

**후보 장비 예시:**
- Siemens Cios Spin  
- GE OEC 3D  
- Ziehm Vision RFD 3D  
- Mini C-arm (사지 연구용)  

---

## 🔬 8. 기대 효과 (Expected Impact)

- 고선량 CT 대체 또는 보조 가능  
- 저선량·저비용 3D imaging 제공  
- 소아/청소년 대상 부담 감소  
- 정형외과·척추 진료의 영상 혁신  
- 외래 기반 3D navigation 가능성  
- 3DGS/NeRF 기반 신의료기술 개발  

---

## 📊 9. 연구 산출물 (Deliverables)

- DRR 생성 스크립트  
- Sparse-view 3D reconstruction pipeline  
- Bone geometry quantitative analysis report  
- Sparse-view vs Reconstruction quality curve  
- Clinical 적용 가능성 평가  
- 실제 C-arm acquisition protocol guideline  
- Phase 1(시뮬레이션) → Phase 2(실촬영) 로드맵 문서  

---

## 🧪 10. DRR 기반 사전 연구의 핵심 장점

- 장비 없이 즉시 연구 가능  
- CT ground truth 대비 reconstruction 정확도 분석 가능  
- View 수·각도 변화 시뮬레이션 무한 반복 가능  
- 노이즈·geometry error 등 다양한 조건 실험 가능  
- 임상 적용 전에 기술 feasibility 완전 검증  

> **결론적으로, DRR 시뮬레이션만으로도  
>  실제 C-arm을 사용하기 전 기술 가능성과 한계를 충분히 평가할 수 있다.**

---

## 📍 11. 프로젝트 로드맵 (Roadmap)

### Phase 0 – 세팅 및 기획
- CT dataset selection  
- DRR simulation pipeline 구축  

### Phase 1 – DRR 기반 Reconstruction 실험
- 16/32/64/128 views  
- Accuracy 분석  
- Sparse-view 최적화  

### Phase 2 – 실제 장비 촬영 프로토콜 개발
- Orbit/shot 최적화  
- Radiation vs Quality curve 확보  

### Phase 3 – Pilot Clinical Study
- 소규모 데이터 기반 임상 검증  
- Surgeon acceptability 평가  
- 제품화/연구과제 proposal 생성  

---

## 📘 Summary

본 프로젝트는 **저선량 X-ray 다중각도 촬영만으로 CT 없이 3D Bone 구조를 재현**하려는 신기술(X2CT)의 사전 검증을 목표로 한다.

실제 장비 없이도:

- **CT 오픈데이터 → DRR 시뮬레이션**만으로  
- 기술적 upper bound,  
- Sparse-view reconstruction 성능,  
- 임상 적용 가능성을 충분히 검증할 수 있다.

이후 실제 C-arm 기반 임상 연구로 확장하여  
정형외과·척추·치과 분야에서 CT 대체 또는 보조 영상기술로 발전시킨다.

---

