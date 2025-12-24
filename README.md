# ⛑️ Implementation and Performance Analysis of Lightweight Object Detection System for Real-time Construction Safety Monitoring on Edge Device

This project uses compatible Pi and Hailo AI accelerators to verify the use of safety equipment in imports.

## 📝 Abstract (소개)
To address the limitations of manual supervision and cloud-based architectures in high-risk environments, this paper proposes a real-time Personal Protective Equipment (PPE) detection system implemented on a low-power edge device using a Raspberry Pi 5 and Hailo-8 NPU. By deploying a YOLOv8s model optimized via Post-Training Quantization (PTQ), we achieved a 54.7% reduction in model size while maintaining a high mAP@0.5 of 0.8817. Experimental results demonstrate that the proposed system reaches an inference speed of 32.99 FPS—an approximate 30-fold increase over CPU-only execution—thereby proving that a decentralized edge solution can effectively ensure bandwidth efficiency and privacy while delivering server-level performance for real-world safety monitoring.

## 📂 Dataset 
The following datasets were used for this project.
* Construction Site Safety Image Dataset Roboflow : https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow - For detecting people, helmets and vests

본 프로젝트는 Kaggle의 'Construction Site Safety' 데이터셋을 재가공하여 사용했습니다.

| 클래스 (Class) | 학습 (Train) | 검증 (Validation) | 테스트 (Test) | 합계 (Total Instances) | 비율 (Ratio) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Person (작업자)** | 10,026 | 163 | 181 | 10,370 | 63.0% |
| **Hardhat (안전모)** | 2,889 | 70 | 103 | 3,062 | 18.6% |
| **Safety Vest (조끼)** | 2,933 | 37 | 61 | 3,031 | 18.4% |
| **합계 (Images)** | **2,605장** | **114장** | **82장** | **2,801장** | **100%** |

> **Note:** 클래스 불균형(Imbalance)이 존재하지만, Person, Hardhat, Safety Vest 3가지 핵심 클래스를 중점적으로 학습했습니다.

## 🛠️ Environment (개발 환경)
* **Hardware**: Raspberry Pi 5 (Broadcom BCM2712), Hailo-8 NPU
* **Language**: Python 3.12.12
* **Libraries**: PyTorch 2.9.0+cu126, Ultralytics YOLOv8.3.233, Hailo Dataflow Compiler v3.33.0

## ⚙️ Training Configuration
모델 학습에 사용된 주요 하이퍼파라미터 및 설정값입니다.

| 분류 | 항목 (Item) | 설정값 (Value) | 목적 |
| :--- | :--- | :--- | :--- |
| **Model** | Architecture | **YOLOv8s** | 엣지 환경과 성능의 균형 (Small Model) |
| **Params** | Input Size | 640 x 640 | 표준 입력 크기 |
| | Max Epochs | 200 | 충분한 수렴을 위한 학습 횟수 |
| | Batch Size | 16 | 메모리 자원 최적화 |
| | Optimizer | Auto (AdamW/SGD) | 최적화 알고리즘 자동 선택 |
| | Initial LR | 0.01 | 초기 학습률 |
| **Augmentation** | Rotation | ±10° | 카메라/작업자 기울기 대응 |
| | Mosaic | 1.0 | 배경 복잡성 및 밀집 객체 학습 |
| | Erasing | 0.4 | 가려짐(Occlusion) 상황 대응 |

## 🆚 Comparison with Previous Works
기존 연구(Study A, Study B)와 비교하여, 본 프로젝트(Ours)는 **속도, 효율성, 프라이버시** 측면에서 가장 균형 잡힌 성능을 보입니다.

| 구분 | Study B [2] (Server) | Study A [1] (Edge Old) | **Ours (Edge New)** |
| :--- | :---: | :---: | :---: |
| **하드웨어** | NVIDIA RTX A6000 | RPi 4 + Intel NCS2 | **RPi 5 + Hailo-8 NPU** |
| **모델** | YOLOv7 (Heavy) | YOLOv4-tiny | **YOLOv8s (INT8)** |
| **정확도 (mAP)** | 92.36% | 86.30% | **88.17%** |
| **속도 (FPS)** | 28.65 | 6.80 (느림) | **32.99 (Real-time)** |
| **네트워크 의존** | 높음 (Cloud 필수) | 없음 | **없음 (On-Device)** |
| **비용/효율** | 고비용/고전력 | 저전력/저성능 | **저비용/고효율** |

## 📊 Performance & Benchmark (실험 결과)
[cite_start]`cpu_benchmark.py`와 `hailo_benchmark.py`를 통해 측정한 성능 비교 결과입니다. [cite: 199, 269]

### 1. 추론 속도 및 정확도 비교 (Inference Speed & Accuracy)
[cite_start]기존 CPU 단독 실행 대비 **약 30배**의 속도 향상을 달성하면서도, 양자화(Quantization)로 인한 정확도 손실을 최소화했습니다. [cite: 12, 205]

| 디바이스 (Device) | 모델 포맷 | 해상도 | 정확도 (mAP@0.5) | 추론 속도 (FPS) |
| :---: | :---: | :---: | :---: | :---: |
| Raspberry Pi 5 (CPU) | FP32 (.pt) | 640x640 | 0.9201 | 1.10 |
| **RPi 5 + Hailo-8 NPU** | **INT8 (.hef)** | **640x640** | **0.8817** | **32.99** |

> [cite_start]**Result:** Hailo-8 NPU 가속기를 적용했을 때, CPU 대비 **약 30배 (2,899%)** 빠른 추론 속도를 기록했습니다. 

---

### 2. 시스템 자원 효율성 (System Efficiency)
[cite_start]NPU 오프로딩을 통해 CPU 자원을 절약하고 발열을 억제하여 엣지 디바이스의 안정성을 확보했습니다. 

| 구성 (Configuration) | CPU 점유율 (Usage) | 기기 온도 (Temp) | 비고 |
| :--- | :---: | :---: | :--- |
| **PyTorch (CPU Only)** | ~40% (병목 발생) | 50°C | [cite_start]속도 매우 느림 (1.1 FPS)  |
| **ONNX Runtime (CPU)** | 100% (자원 포화) | **75°C (과열)** | [cite_start]발열로 인한 스로틀링 위험 [cite: 243] |
| **Hailo-8 NPU (Proposed)** | **~25% (여유)** | **50°C (안정)** | [cite_start]**고성능 & 저발열 구현**  |

## 🚀 How to Run (실행 방법)
# 1. 저장소 복제 및 이동
git clone [YOUR_REPOSITORY_LINK]
cd [REPOSITORY_NAME]

# 2. 의존성 라이브러리 설치 (PyTorch, Ultralytics 등)
pip install -r requirements.txt

# PTQ(Post-Training Quantization) 적용 및 HEF 변환
python convert_model.py --model yolov8s.pt --calib-imgs ./calibration_data/

# 웹캠을 통한 실시간 PPE 탐지 실행
python inference_hailo.py --model yolov8s_int8.hef --input 0
