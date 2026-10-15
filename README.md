# 🐄 Bovine Analysis — AI-Powered Breed Identification and Health Analysis

An AI-powered bovine analysis application developed as part of a Smart India Hackathon (SIH) project. The system combines deep learning, computer vision, and image segmentation to analyse cattle and buffalo images, identify breeds, classify animal types, estimate visual body measurements, and provide preliminary image-based health classification.

Built using **Python, PyTorch, MobileNetV2, YOLO, OpenCV, and Streamlit**, the application provides an interactive interface for analysing bovine images.

## 🚀 Key Features

- **Breed Identification:** Predicts the bovine breed from an uploaded image using a MobileNetV2-based deep learning model.
- **Animal-Type Classification:** Uses a multi-output neural network to predict the breed and animal type.
- **Image-Based Health Classification:** Applies a separate MobileNetV2 model to classify images according to its trained health labels.
- **Bovine Segmentation:** Uses YOLO segmentation models to identify animal regions in images.
- **Body Measurement Estimation:** Extracts image dimensions from segmentation masks and estimates measurements using reference information.
- **Breed Information Database:** Provides breed-related information, including typical weight ranges, milk-yield descriptions, breed origins, and reference body lengths.
- **Interactive Web Application:** Provides an image-upload and results interface using Streamlit.

## 🧠 System Architecture

The application combines classification and segmentation into a single image-analysis workflow.

| Component | Technology | Purpose |
|---|---|---|
| Breed and animal-type classification | MobileNetV2 + custom output heads | Predicts breed and animal type |
| Health classification | MobileNetV2 | Predicts the trained health category |
| Bovine segmentation | Ultralytics YOLO | Identifies bovine regions and masks |
| Image processing | OpenCV, Pillow, NumPy | Processes images and segmentation masks |
| Model inference | PyTorch | Runs trained deep learning models |
| Application interface | Streamlit | Provides image uploads and displays results |

### How It Works

1. **Image Upload:** The user uploads a cattle or buffalo image through the application.
2. **Preprocessing:** The image is resized to 224 × 224 pixels and normalised using ImageNet mean and standard deviation values.
3. **Breed and Type Prediction:** A MobileNetV2-based model generates predictions for breed and animal type.
4. **Health Classification:** A second trained MobileNetV2 model predicts the image's health category.
5. **Segmentation:** YOLO identifies the animal region to support visual analysis.
6. **Measurement Estimation:** Segmentation masks and pixel dimensions are used to estimate body measurements, with breed reference lengths used where applicable.
7. **Result Presentation:** The Streamlit interface displays the available classification results and breed-related information.

## 🛠️ Technology Stack

**Programming Language**
- Python

**Deep Learning**
- PyTorch
- Torchvision
- MobileNetV2
- Ultralytics YOLO

**Computer Vision**
- OpenCV
- Pillow
- NumPy

**Web Application**
- Streamlit

## 📂 Repository Structure

```text
bovine-analysis-sih/
├── app.py
├── bovine_classifier_model.pth
├── health_classifier_model.pth
├── best.pt
├── best_rump.pt
├── requirements.txt
├── packages.txt
└── runtime.txt
```

| File | Description |
|---|---|
| `app.py` | Main Streamlit application, model definitions, inference pipeline, and breed information database |
| `bovine_classifier_model.pth` | Trained breed and animal-type classification checkpoint |
| `health_classifier_model.pth` | Trained health-classification checkpoint |
| `best.pt` | YOLO segmentation model checkpoint |
| `best_rump.pt` | Additional YOLO model checkpoint |
| `requirements.txt` | Python package dependencies |
| `packages.txt` | Additional system package configuration |
| `runtime.txt` | Runtime configuration |

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sreyas15/bovine-analysis-sih.git
cd bovine-analysis-sih
```

### 2. Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you encounter PyTorch or Torchvision compatibility problems, install compatible versions for your Python version and hardware using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

**Note:** The trained model checkpoint files must remain in the expected project directory because the application loads them using local file paths.

## 📊 Applications

Potential application areas include:

- Preliminary bovine breed identification.
- Digital livestock record management.
- Image-based livestock analysis.
- Livestock information systems.
- Research and experimentation in agricultural AI and computer vision.

## ⚠️ Limitations

- Predictions depend on image quality, camera angle, lighting, and training-data coverage.
- Classification confidence does not guarantee the correctness of a prediction.
- Health classification is experimental and must not be treated as a veterinary diagnosis.
- Body measurements are approximate. Accurate physical measurements require appropriate camera calibration, controlled image capture, and validation against real measurements.
- Breed characteristics and reference values can vary by age, sex, nutrition, and individual animal.
- The system should be validated using independent test data before use in real-world livestock management.

## 🔮 Future Improvements

- Add per-class precision, recall, F1-score, and confusion-matrix reporting.
- Improve robustness across different lighting conditions and camera angles.
- Incorporate calibrated measurement techniques for more reliable body-size estimation.
- Validate image-based health classification using veterinary-reviewed datasets.
- Introduce automated testing, model monitoring, and more comprehensive error handling.

## 👨‍💻 Author

**Sreyas Sanil Kumar**

- GitHub: [@sreyas15](https://github.com/sreyas15)
- Project Repository: [Bovine Analysis – SIH](https://github.com/sreyas15/bovine-analysis-sih)

---

*An applied AI and computer vision project exploring deep learning for bovine identification and image-based livestock analysis.*
