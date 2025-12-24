# ⛑️ Hongik PPE Project (개인 보호 장비 탐지 시스템)

라즈베리파이와 Hailo AI 가속기를 활용하여 작업자의 안전 장비 착용 여부를 실시간으로 탐지하는 프로젝트입니다.

## 📝 Abstract (소개)
이 프로젝트는 건설 현장 등 위험 지역에서 작업자가 헬멧이나 조끼 등 보호 장비를 착용했는지 확인하기 위해 개발되었습니다. 
Edge Device인 라즈베리파이에서 구동되며, 추론 속도 향상을 위해 Hailo 가속기를 사용했습니다.

## 📂 Dataset (사용 데이터셋)
본 프로젝트 학습에는 아래의 데이터셋들이 활용되었습니다.
* **SH17 Dataset**: [링크 설명](링크주소) - 사람 및 헬멧 탐지용
* **Custom Dataset**: 직접 수집한 현장 데이터 500장

## 🛠️ Environment (개발 환경)
* **Hardware**: Raspberry Pi 4, Hailo-8L
* **Language**: Python 3.8
* **Libraries**: PyTorch, Hailo RT

## 📊 Performance & Benchmark (실험 결과)
`cpu_benchmark.py`와 `hailo_benchmark.py`를 통해 측정한 결과입니다.

| 디바이스 | 해상도 | 추론 속도 (FPS) | 전력 소모 (W) |
| :---: | :---: | :---: | :---: |
| Raspberry Pi (CPU) | 640x640 | 2.5 | 4.2 |
| **RPi + Hailo** | **640x640** | **30.1** | **5.5** |

> Hailo 가속기를 사용했을 때 CPU 대비 약 12배 빠른 속도를 보였습니다.

## 🚀 How to Run (실행 방법)
1. 의존성 라이브러리 설치
   ```bash
   pip install -r requirements.txt
