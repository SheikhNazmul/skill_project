# 🌶️ AI-Based Chilli Plant Health & Smart Irrigation System

University final-semester project combining **computer vision for chilli leaf disease classification** with an **IoT-based automatic irrigation controller**.

## Project Overview

The system has two complementary modules:

1. **AI Disease Detection** — a transfer-learning image classifier learns to classify chilli plant leaf images. The training pipeline uses a pretrained MobileNetV3 backbone and can be trained on the provided Kaggle dataset.
2. **Smart Irrigation** — an Arduino reads soil moisture and automatically controls a water pump using configurable moisture thresholds.

> **Academic note:** The image dataset is sourced from Kaggle and is not redistributed in this repository. Dataset attribution and the original source link are provided below.

## Dataset

**Chilli Plant Diseases Dataset — Kaggle**  
https://www.kaggle.com/datasets/ravindubandara3002/chilli-plant-diseases-dataset

Download the dataset separately and arrange it in an `ml/data/` directory using an ImageFolder-compatible structure:

```text
ml/data/
├── class_1/
│   ├── image1.jpg
│   └── image2.jpg
├── class_2/
│   └── ...
└── ...
```

Do not commit the full dataset or large trained model files to this repository.

## Architecture

```text
                ┌──────────────────────┐
                │ Chilli Leaf Image    │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ MobileNetV3 Transfer │
                │ Learning Classifier  │
                └──────────┬───────────┘
                           ↓
                Disease Classification

Soil Moisture Sensor → Arduino → Threshold Logic → Water Pump
                              │
                    < 30% → Pump ON
                    > 70% → Pump OFF
```

## Repository Structure

```text
AI-Chilli-Plant-Health-System/
├── README.md
├── iot/
│   └── smart_irrigation.ino
└── ml/
    ├── train.py
    ├── predict.py
    ├── app.py
    └── requirements.txt
```

## AI Module

The training script uses:

- Python
- PyTorch
- Torchvision
- MobileNetV3 Small pretrained weights
- Cross-entropy loss
- Adam optimizer
- Validation split

Install dependencies:

```bash
pip install -r ml/requirements.txt
```

Train:

```bash
python ml/train.py --data_dir ml/data --epochs 10
```

The best checkpoint is saved to `ml/artifacts/best_model.pt`.

Run the Streamlit prediction app:

```bash
streamlit run ml/app.py
```

## IoT Module

Hardware concept:

- Arduino-compatible board
- Analog soil moisture sensor
- Relay module
- DC water pump
- External pump power supply

The supplied Arduino program maps the calibrated sensor range to a 0–100% moisture value and uses hysteresis-style thresholds:

- Moisture below **30%** → pump ON
- Moisture above **70%** → pump OFF

> **Safety:** The pump should be powered through an appropriately rated relay/driver and separate suitable power supply. Do not connect a pump directly to an Arduino GPIO pin.

## Future Improvements

- Add more disease classes and balanced validation data
- Fine-tune the pretrained backbone for higher accuracy
- Add confidence scores and confusion-matrix reporting
- Connect IoT readings to a web/mobile dashboard
- Add MQTT/REST communication between the device and backend
- Store sensor history in PostgreSQL/Supabase
- Add scheduled irrigation and manual override

## Academic Contribution

The project demonstrates an applied **AI + IoT** workflow: computer vision helps identify plant-health conditions while sensor-driven automation supports irrigation decisions. The implementation is designed as a reproducible university project rather than a claim of a clinically/agronomically certified diagnostic system.

## Author

**Sheikh Nazmul Islam NiR**  
Computer Science & Engineering — Daffodil International University

- GitHub: https://github.com/SheikhNazmul
- LinkedIn: https://www.linkedin.com/in/sheikh-nazmul-islamm-1a649b296
