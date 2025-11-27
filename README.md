# Style-Conditioned Diffusion

A lightweight implementation of **style-conditioned image generation** based on Stable Diffusion.
This project injects **CLIP-based style embeddings** into UNet cross-attention using a custom **Style Attention Processor**, enabling diffusion models to follow a specific artistic style during training and inference.

---

## 📁 Project Structure

```
.
├── config.py               # Global config (paths, model IDs, hyperparams)
├── dataset_loader.py       # Dataset for (style image, target image) pairs
├── downloader.py           # Scripts for downloading WikiArt/custom datasets
├── inference.py            # Style-conditioned image generation
├── model.py                # ImageProjModel + StyleAttnProcessor
├── process_data.py         # Preprocessing, VGG/CLIP feature extraction, KMeans
├── train.py                # Main training loop
```

---

# 🧩 Environment Setup (Mac & Windows)

This project uses Python **3.10+** and requires an isolated virtual environment.

---

## macOS

### 1. Create a virtual environment

```bash
python3 -m venv venv
```

### 2. Activate the environment

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Windows (PowerShell)

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the environment

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---
# 📥 Dataset Preparation

Before training the style-conditioned diffusion model, you must download the dataset and extract style/content features.

This project default dataset:
**WikiArt dataset (via HuggingFace)**

---
## 1️⃣ Download images

Use:

```bash
python downloader.py
```

This script downloads WikiArt images (or a subset) into the path defined in `config.py`:
```bash
RAW_DATA_PATH = "data/raw_images/"
METADATA_CSV = "data/metadata.csv"
```

To customize download size, filtering, or groups (e.g., exclude Unknown Artist/Genre), adjust downloader.py or `config.py`.

---
## 2️⃣ Preprocess: extract style/content features


After downloading raw images, run:
```bash
python process_data.py
```

This script performs:

* **DINOv2** style feature & **CLIP-Vision** content features

* Filtering bad images

* Images Group by `artist`, `genre`, `style`, then do PCA visualization for each group

* Split data into test pairs and testing images

Outputs are saved under:
```bash
test_data --> "./data/"
pca_visual and style_pairs --> "./data/processed"
```

These features are used by both the dataset loader and training pipeline.

---
# 🚀 Training

Train your style-conditioned diffusion model:

```bash
python train.py
```

You can modify hyperparameters and model configs in `config.py`.

---

# 🎨 Inference

Inference will take a style reference image to generate an image with style guidance:

```bash
python inference.py {image_name}
```


---

# ✨ Features

* **Style Attention Processor (SAP)**
  Injects learnable style tokens into UNet cross-attention.

* **CLIP-Based Style Embeddings**
  Uses CLIP-Vision or VGG Gram features as style representations.

* **Image Projector (ImageProjModel)**
  Converts style embeddings into cross-attention tokens.

* **Evaluations**
  VGG Gram style loss, simplified DDPM evaluator, and AB testing.

* **Dataset Tools**
  Supports WikiArt metadata, clustering, feature extraction, and preprocessing.

---

# 📦 Dependencies

```
torch
diffusers
transformers
torchvision
datasets
scikit-learn
matplotlib
Pillow
```

---

# 📄 License

MIT License.

---