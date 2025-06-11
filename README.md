# 📘 Sem V Mini Project

## 🌟 EcoSort: The Waste Classification Model

---

## ♻️ EcoSort

**EcoSort** is an AI-powered waste classification web application that helps users **identify** and **properly dispose** of waste by classifying it into six key categories using **deep learning**.

> Our mission is to promote environmental awareness and encourage responsible recycling habits through technology.

---

## 🌱 Project Objective

EcoSort is designed to:

- ✅ Automatically classify waste from images into six categories.  
- ✅ Provide real-time suggestions for proper disposal.  
- ✅ Raise public awareness on sustainable waste management.  

---

## 🧠 Tech Stack

- **Frontend**: `Streamlit`  
- **Backend**: `Python`  
- **Machine Learning**: `TensorFlow`, `Keras` *(VGG16 Model)*  
- **Libraries Used**: `OpenCV`, `NumPy`, `Matplotlib`  
- **Environment**: `Google Colab`, `Jupyter Notebook`

---

## 🗂️ Waste Categories

EcoSort classifies waste into the following six categories:

- 🟫 **Cardboard**  
- 🌿 **Compost**  
- 🧪 **Glass**  
- 🛢️ **Metal**  
- 📄 **Paper**  
- 🧴 **Plastic**  

---

## ⚙️ Model Overview

- **Model Used**: VGG16 *(Pre-trained on ImageNet and fine-tuned for 6 waste classes)*  
- **Optimizer**: `Adam`  
- **Loss Function**: `Categorical Crossentropy`  
- **Test Accuracy**: ~71%  
- **Data Augmentation**: Rotation, Zoom, Horizontal Flip

---

## 🚀 Features

- 📸 Upload waste images for instant classification  
- 📊 Real-time predictions with confidence scores  
- 🧾 Smart suggestions for eco-friendly disposal  
- 🌍 Informative insights about each waste category

---

## 🛠️ How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ecosort.git
   cd ecosort
