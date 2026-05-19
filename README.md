# 🌿 Plant Disease Detection Web App (Dynamic CNN + Grad-CAM++)

A Flask-based web application for plant disease classification using lightweight deep learning models with integrated Grad-CAM++ visualizations for interpretable predictions.

> 🏷️ This project implements the **DynLeafNet** architecture proposed in the paper:
> *"DynLeafNet: A dynamic lightweight architecture for plant disease classification using dynamic residual network with explainable artificial intelligence"* (ScienceDirect, 2025).
> The live demo and deployment were developed independently as part of this project.
---

## 🌐 Live Demo

Try the app live on Hugging Face Spaces:

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/HansineeS/plant-disease-detection)

> Upload a plant leaf image and get instant disease prediction with Grad-CAM++ explainability — no setup required.

---
## 🚀 Features

- 🌱 Multi-dataset support:
  - PlantVillage (38 classes)
  - Rice Leaf Disease (6 classes)
- ⚡ Lightweight PyTorch models optimized for efficient inference
- 🧠 Explainable AI using Grad-CAM++ heatmaps
- 📊 Outputs:
  - Predicted disease label
  - Confidence score
  - Visual explanation overlay
- 🌐 Simple web interface for real-time inference

---

## 📁 Project Structure

```text
project-root/
├── app.py                         # Flask app + model inference logic
├── templates/
│   └── index.html                # Frontend UI
├── models/
│   ├── plantvillage/             # PlantVillage model + class mapping
│   └── rice-leaf/                # Rice model + class mapping
├── static/                       # Uploaded images / outputs (optional)
├── requirements.txt              # Dependencies
└── README.md
```

---

## 🛠 Tech Stack

- **Backend:** Flask (Python)
- **Deep Learning:** PyTorch
- **Explainability:** Grad-CAM++
- **Frontend:** HTML, CSS (Jinja templates)

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- Flask
- torchvision
- OpenCV / PIL

> GPU is optional. The application automatically falls back to CPU if CUDA is unavailable.

---

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd plant-disease-app
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

**Windows**

```powershell
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000/
```

---

## 🧪 Usage

1. Upload a plant leaf image
2. Select dataset:

   * `PlantVillage`
   * `Rice`
3. Click **Submit**

### Output Includes:

* Predicted disease class
* Confidence score
* Original image preview
* Grad-CAM++ heatmap highlighting disease regions

---

## 🧠 Model Details

* Lightweight CNN architecture (DynLeafNet)
* Optimized for low-parameter inference
* Supports multi-class classification
* Integrated explainability using Grad-CAM++

---

## ⚠️ Notes

* Ensure model files exist in:

  * `models/plantvillage/`
  * `models/rice-leaf/`
* Missing files will raise runtime errors (`FileNotFoundError`)
* Debug mode is enabled by default (disable for production)

---

## 🚀 Future Improvements

* Deploy as a cloud-based API (Docker / AWS / GCP)
* Add mobile-friendly UI
* Support real-time camera inference
* Extend to multi-disease detection per image
