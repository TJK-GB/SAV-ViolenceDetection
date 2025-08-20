# SAV-ViolenceDetection

This repository hosts the code, annotations, and experimental results for the **SAV Dataset**, a segment-level dataset designed for video violence detection.
It supports research on violence recognition using CNNs, RNNs, Transformers, and interpretability methods.

---
## Repository Structure

- **Annotations_Final/**
  - Final and historical annotation CSVs
  - `Old/` – older versions of annotations

- **Data_split/**
  - `Video_Source_Considered_split/` – train/test CSVs with source labels
  - `Violence_Label_Only_split/` – train/test CSVs with violence labels only

- **Results/**
  - **Interpretability/**
    - Attention RollOut (Swin, Swin+GRU)
    - GradCAM (CUE-Net, ResNet50GRU)
    - Original images
  - **Model Training (with video)/**
  - **Model Training (without video)/**
  - Summary spreadsheets (`Results Overall.xlsx`)

- **src/**
  - **Dataset related/** – scripts for dataset creation and preprocessing
  - **Model training/** – training scripts (source and label-only)
  - **Results related/Interpretability/**
  - Colab and alternative versions

- `.gitignore` – ignore rules  
- `requirements.txt` – Python dependencies  
- `Dataset instruction.txt` – dataset usage notes


## Dataset Access
The dataset is too large to host on GitHub. You can download it here(or also can be found in the 'Dataset instruction.txt'):  
[**Google Drive Link**](https://drive.google.com/drive/folders/1v6knivpMxbG1_S1musGSOx8wo0fQMRv3?usp=sharing)

## Contents include:
- **Frame-level data in `.npy` format**  
  - Unimodal version (frames only)  
  - Video-source–annotated version  
- **200 YouTube videos** used for dataset construction

---

## Installation
Clone the repository:
```bash
git clone https://github.com/TJK-GB/SAV-ViolenceDetection.git
cd SAV-ViolenceDetection
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Models Implemented
 - ResNet50 + Bi-GRU
 - Swin Transformer
 - Swin + GRU
 - CUE-Net
 - LLM-based experiments (prompt-based classification)


## Interpretability
- GradCAM (ResNet50, CUE-Net)  
- Attention Rollout (Swin, Swin+GRU)  

## Citation
If you use this dataset or code, please cite:
```
@misc{sav2025,
title = {SAV Dataset: Fine-Grained Segment-Level Annotations for Video Violence Detection},
author = {Kim, Taejin},
year = {2025},
url = {https://github.com/TJK-GB/SAV-ViolenceDetection}
}
```

## Contact
For questions or collaborations, please contact:
Taejin Kim: aa9188883@gmail.com
