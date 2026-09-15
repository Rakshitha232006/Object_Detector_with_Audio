# Object Detector with Audio

A deep learning application that detects objects in an uploaded image, draws bounding boxes around the detected objects, generates a natural-language description, and converts the description into speech.

## 🚀 Live Demo

https://objectdetectorwithaudio-gmux7vx42xa2nkowmxuqi3.streamlit.app/

## 📌 Features

- Upload an image through a Streamlit web interface
- Detect objects using DETR
- Draw bounding boxes around detected objects
- Display confidence scores
- Count detected objects
- Generate a natural-language description
- Convert the description into audio using SpeechT5
- Play the generated audio directly in the application

## 🛠️ Technologies Used

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- DETR
- SpeechT5
- Pillow
- NumPy
- SciPy
- timm

## 🤖 Models Used

### Object Detection

**facebook/detr-resnet-50**

DETR (DEtection TRansformer) is used to detect objects in uploaded images.

### Text-to-Speech

**microsoft/speecht5_tts**

SpeechT5 is used to convert the generated image description into speech.

**microsoft/speecht5_hifigan**

SpeechT5 HiFi-GAN is used as the vocoder to generate the final audio waveform.

## 🔄 How It Works

```text
Upload Image
     ↓
DETR Object Detection
     ↓
Detected Objects + Bounding Boxes
     ↓
Object Counting
     ↓
Natural Language Description
     ↓
SpeechT5 Text-to-Speech
     ↓
Generated Audio
```

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Rakshitha232006/Object_Detector_with_Audio.git
```

### 2. Open the project

```bash
cd Object_Detector_with_Audio
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

On Windows:

```powershell
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run main.py
```

## 📂 Project Structure

```text
Object_Detector_with_Audio/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🖼️ Application Workflow

```text
Input Image
     ↓
Object Detection
     ↓
Processed Image with Bounding Boxes
     ↓
"This picture contains 2 dogs."
     ↓
Generated Audio
```

## 📊 Output

The application provides:

- Original uploaded image
- Processed image with bounding boxes
- Text description of detected objects
- Generated audio description

## ⚙️ Detection Threshold

The DETR detector uses a confidence threshold of `0.2` so that lower-confidence detections can also be considered.

## 🎯 Project Goal

The goal of this project is to combine computer vision and speech synthesis into a single application that can identify objects in an image and provide an audio description.

## 🌐 Deployment

The application is deployed using Streamlit Cloud.

### Live Application

https://objectdetectorwithaudio-gmux7vx42xa2nkowmxuqi3.streamlit.app/

## 👩‍💻 Author

**Rakshitha**

GitHub: https://github.com/Rakshitha232006
