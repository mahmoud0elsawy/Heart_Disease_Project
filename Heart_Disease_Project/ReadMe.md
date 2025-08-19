# 🫀 Heart Disease Prediction & Analysis

This project focuses on analyzing and predicting **heart disease risks** using **Machine Learning**.  
The workflow covers data preprocessing, feature selection, dimensionality reduction, supervised learning, clustering, and model deployment.

---

## 📂 Project Structure



project/
│
├── data/
│ └── heart_disease.csv # Dataset (processed and saved in each step)
│
├── models/
│ ├── LogisticRegression.pkl # Saved best supervised model
│ └── kmeans_model.pkl # Saved clustering model
│
├── notebooks/
│ ├── 01_pca.md # Dimensionality reduction with PCA
│ ├── 02_feature_selection.md# Feature selection with ANOVA F-test
│ ├── 03_classification.md # Classification models (LR, DT, RF, SVM)
│ ├── 04_clustering.md # KMeans & Hierarchical clustering
│ ├── 05_visualization.md # Data visualization & EDA
│ └── 06_deployment.md # Model saving & deployment steps
│
└── README.md # Project documentation


---

## 🔑 Steps Overview

### 1️⃣ PCA (Principal Component Analysis)
- Reduced dimensionality of the dataset.  
- Visualized explained variance and first 2 components.  
- Ensured **95% variance** coverage with ~8 components.  
- Saved transformed dataset.

### 2️⃣ Feature Selection
- Applied **ANOVA F-test** with `SelectKBest`.  
- Selected **top 7 features** based on importance scores.  
- Updated dataset with only selected features + target.

### 3️⃣ Classification Models
- Compared multiple classifiers:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine  
- Evaluated using **Accuracy, Confusion Matrix, and Classification Report**.  
- **Logistic Regression** performed best and was saved.

### 4️⃣ Clustering
- Applied **KMeans** with optimal `k` determined using Elbow Method.  
- Compared **KMeans vs. Hierarchical clustering**.  
- Visualized clusters after PCA (2D reduction).  
- Saved clustering pipeline with scaling.

### 5️⃣ Visualization
- Plotted key features, PCA explained variance, feature importance, clustering distributions, etc.  
- Used **Matplotlib** and **Seaborn** for insights.

### 6️⃣ Deployment
- Best supervised model saved as:
  - `../models/LogisticRegression.pkl`  
- Clustering model saved as:
  - `../models/kmeans_model.pkl`  
- Ready for loading and prediction in production apps.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-ml.git
   cd heart-disease-ml


Install dependencies:

pip install -r requirements.txt


Run Jupyter notebooks (or open .md converted versions).

To load saved models:

import joblib

model = joblib.load("models/LogisticRegression.pkl")
prediction = model.predict(new_data)  # new_data must match features

📊 Results

Logistic Regression achieved the best accuracy (~X%).

PCA reduced dimensions while keeping 95% variance.

Feature selection identified the most significant attributes for prediction.

Clustering showed meaningful group separation in 2D PCA space.

🛠️ Tools & Libraries

Python, Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Joblib