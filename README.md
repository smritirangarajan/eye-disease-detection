
# Eye Disease Prediction System

A deep learning–based web application that predicts **human eye diseases** from retina images using **TensorFlow/Keras** and **Streamlit**.  
This project implements an **image classification system** trained on a Kaggle dataset, capable of detecting **multiple eye disease classes**.

---

## 🚀 Project Overview
- **Dataset**: Retina images from Kaggle ([link](https://www.kaggle.com/datasets/anirudhcv/labeled-optical-coherence-tomography-oct))  
- **Model Architecture**: **MobileNetV3**  
- **Frameworks**: TensorFlow, Keras, scikit-learn, Streamlit  
- **Web App**: Built with Streamlit (deployed live ✅)  
- **Features**:
  - Preprocessing of training and validation data  
  - Model training with transfer learning  
  - Evaluation (accuracy, precision, recall, F1-score, confusion matrix)  
  - Prediction web interface (upload retina image → get disease prediction)  
  - Recommendation feature (basic guidance based on prediction)  

---

## 📂 Project Structure
```
├── data/                     # (ignored in git, download from Kaggle)
├── notebooks/                # Jupyter notebooks for experimentation
├── Trained_Eye_disease_model.h5  # Saved trained model
├── app.py                     # Streamlit app entry point
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore
```

---

## 📊 Model Performance
- **Accuracy**: Achieved high accuracy on the test set  
- **Evaluation Metrics**: Classification report & confusion matrix used  
- **Visualization**: Training/validation loss and accuracy plots  

---

## 🌐 Streamlit Deployment
This project is deployed with **Streamlit**.  
You can run it locally or try it ([online](https://smritirangarajan-eye-disease-detection-app-hj7ofx.streamlit.app/)).  

**Run locally:**
```bash
# Clone repository
git clone https://github.com/your-username/eye-disease-detection.git
cd eye-disease-detection

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Mac/Linux
.venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

## 📦 Requirements
Main dependencies:
- TensorFlow ≥ 2.16  
- Keras  
- scikit-learn  
- NumPy, Pandas  
- Matplotlib  
- Streamlit  

(see `requirements.txt` for full list)

---

## 📥 Dataset
The dataset is **not included** in this repo due to size.  
Download it from Kaggle: [Eye Disease Dataset](https://www.kaggle.com/datasets/aniru...)  

After downloading, place it inside the `dataset/` folder.

---

## 📊 Results & Evaluation
- Precision, Recall, and F1-score calculated for each class  
- Confusion matrix visualization to interpret misclassifications  
- Model history plotted (loss & accuracy curves)

---
