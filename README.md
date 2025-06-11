# SemV
Mini Project

EcoSort : The Waste Classification Model

♻️ EcoSort
EcoSort is an AI-powered waste classification web application that helps users identify and properly dispose of waste by classifying it into six key categories using deep learning. The project aims to promote environmental awareness and responsible recycling habits.

🌱 Project Objective
To build an intelligent system that:

Automatically classifies waste from images into six categories.

Provides real-time suggestions for proper disposal.

Raises public awareness on sustainable waste management.

🧠 Tech Stack
Frontend: Streamlit

Backend: Python

Machine Learning: TensorFlow, Keras (VGG16)

Libraries Used: OpenCV, NumPy, Matplotlib

Environment: Google Colab, Jupyter Notebook

🗂️ Waste Categories
EcoSort identifies the following six types of waste:

Cardboard

Compost

Glass

Metal

Paper

Plastic

⚙️ Model Overview
Model Used: VGG16 (pre-trained on ImageNet, fine-tuned for waste classification)

Optimizer: Adam

Loss Function: Categorical Crossentropy

Accuracy: ~71% on test data

Data Augmentation: Rotation, Zoom, Flipping to improve generalization

🚀 Features
📸 Upload real-time waste images for classification

📊 Instant prediction with category and confidence score

🧾 Smart suggestions for disposal and recycling

🌍 Informative section to educate users on each category
