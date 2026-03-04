# ☕ Specialty Coffee Quality Predictor

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline that classifies the quality of specialty coffee based on sensory evaluations and physical attributes. Using data from the Coffee Quality Institute (CQI), the project demonstrates a complete data science workflow: from deep exploratory data analysis (EDA) and data cleaning to model building, hyperparameter tuning, and web deployment.

## 🚀 Key Highlights & Features
- **Exploratory Data Analysis (EDA):** Conducted in-depth analysis of coffee varieties and country of origin distributions.
- **Advanced Visualizations:** Created professional, presentation-ready plots using `Seaborn` and `Matplotlib` (e.g., custom palettes, whitegrid themes) to uncover data insights.
- **Data Preprocessing:** Cleaned a dataset of 1339 records and 43 features, handled missing values, dropped redundant columns, and encoded categorical variables.
- **Machine Learning Model:** Trained a robust `RandomForestClassifier`.
- **Hyperparameter Tuning:** Utilized `GridSearchCV` to exhaustively search for the optimal model parameters (`n_estimators`, `max_depth`, `min_samples_split`, `criterion`).
- **Interactive Web Dashboard:** Built and deployed a visually appealing, dark-themed Streamlit application for real-time quality predictions.

## 🛠️ Tech Stack & Libraries
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`
- **Model Serialization:** `joblib`
- **Web App Deployment:** `streamlit`

## 📊 Dataset Insights
The dataset contains Arabica and Robusta evaluations. Key findings from the EDA include:
- Identified 36 unique countries of origin, with Mexico, Colombia, and Guatemala being the top contributors.
- Analyzed 29 unique coffee varieties, highlighting *Caturra*, *Bourbon*, and *Typica* as the most frequent.

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/coffee-quality-classifier.git](https://github.com/yourusername/coffee-quality-classifier.git)
cd coffee-quality-classifier