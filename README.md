# Myntra Pants Price Prediction — Complete ML Pipeline

A comprehensive End-to-End Machine Learning regression pipeline built to predict the sales price of pants listed on Myntra using web-scraped behavioral, product, and brand data. 

## Table of Contents
- [Business Context](#business-context)
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Workflow](#project-workflow)
- [Models Trained & Evaluated](#models-trained--evaluated)
- [Final Results](#final-results)
- [Saved Pipeline Artifacts](#saved-pipeline-artifacts)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)

---

## Business Context
In competitive e-commerce platforms like Myntra, pricing strategies significantly drive conversion rates and profit margins. Dynamically forecasting the ideal selling price based on brand equity, raw Maximum Retail Price (MRP), customer rating footprint, and engineered descriptive attributes allows platforms or vendors to maximize inventory clearance while keeping discounts optimized.

---

## Project Overview
The goal is to predict the Sale Price (Regression) of products based on their brand, MRP, ratings, and physical characteristics extracted from their descriptions.

The engineered architecture includes:
* **Exploratory Data Analysis (EDA):** Detailed Univariate, Bivariate, and Multivariate analysis.
* **Feature Engineering:** Extracting structured attributes from unformatted structural text descriptions (`pants_description`).
* **Data Preprocessing:** Label encoding categorical elements and handling scaling with `StandardScaler`.
* **Multi-Regressor Benchmarking:** Parallel training and diagnostic tracking of **6 Regression Algorithms**.
* **Hyperparameter Optimization:** Automatic hyperparameter tuning via `GridSearchCV`.
* **Serialization:** Pickle-based persistence of best models, encoders, and transformers for seamless backend integration.

---

## Dataset Description
The model utilizes the **Myntra Pants Scraping Dataset** consisting of **52,120 rows** across **7 original feature columns**:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `brand_name` | Object (Categorical) | Name of the fashion brand (e.g., WROGN, Flying Machine, Levis) |
| `pants_description` | Object (Text) | Descriptive title string of the listed product item |
| `MRP` | Float64 | Maximum Retail Price of the product |
| `discount_percent` | Float64 | Discount percentage applied to the item |
| `ratings` | Float64 | Average customer rating score (1.0 to 5.0) |
| `number_of_ratings` | Float64 | Total volume count of customer reviews/ratings submitted |
| **`price` (Target)** | Float64 | Final selling price after discounts applied (Regression Target) |

---

## Project Workflow
Raw Dataset (52,120 rows, 7 columns)
       │
       ▼
Exploratory Data Analysis (EDA)
 ├── Univariate Distribution Analysis
 ├── Bivariate Attribute Correlations
 └── Multivariate Relationship Trends
       │
       ▼
Feature Engineering
 ├── Extract 'fit_type' from pants_description
 └── Extract 'fabric_type' from pants_description
       │
       ▼
Data Preprocessing & Encoding
 ├── Drop unneeded raw structural text columns
 └── Label Encode Categorical: ['brand_name', 'fit_type', 'fabric_type']
       │
       ▼
Train-Test Split (80% Train / 20% Test, random_state=42)
       │
       ▼
Feature Scaling
 └── Apply StandardScaler to independent variables (X)
       │
       ▼
Multi-Model Evaluation & Benchmarking
 ├── Train 6 Regressors (Linear, Ridge, Lasso, DT, RF, GBR)
 └── Track Performance Across Metrics (R², MAE, RMSE)
       │
       ▼
Hyperparameter Optimization
 └── Automated GridSearchCV on top-performing architectures
       │
       ▼
Model Evaluation & Diagnostic Diagnostics
 ├── Identify Best Model Configuration via Test R² Score
 └── Plot Actual vs. Predicted Price Distribution Curves
       │
       ▼
Pipeline Serialization (Export Artifacts)
 ├── Save Best Regressor ➔ myntra_price_best_model.pkl
 ├── Save Transformer ➔ myntra_price_scaler.pkl
 └── Save LabelEncoders ➔ myntra_price_encoders.pkl

---

## Models Trained & Evaluated
Six regression algorithms were evaluated and compared side-by-side to determine the optimal price estimation system:

1. **Linear Regression** (Baseline)
2. **Ridge Regression** (L2 Regularization)
3. **Lasso Regression** (L1 Regularization)
4. **Decision Tree Regressor**
5. **Random Forest Regressor** (Ensemble)
6. **Gradient Boosting Regressor** (Sequential Boosting)

*The optimal pipeline pipeline automatically identifies the best model configuration based on the highest Test **$R^2$ Score**.*

---

## Final Results
* **Auto-Identified Best Model:** *[Insert your notebook's top model name here, e.g., Gradient Boosting Regressor / Random Forest]*
* **Evaluation Metrics Tracked:**
  * **$R^2$ (Coefficient of Determination):** Evaluates variation match percentage.
  * **Mean Absolute Error (MAE):** Quantifies absolute pricing deviances.
  * **Root Mean Squared Error (RMSE):** Penalizes extreme pricing errors.
* Plots generated for validation include **Actual vs. Predicted Price Distribution Curves** and residual plots.

---

## Saved Pipeline Artifacts
The completion of the training cycle automatically dumps three cross-compatible serialization `.pkl` matrices into your workspace folder for live inference staging:

| Pickle Filename | Component Functionality |
| :--- | :--- |
| `myntra_price_best_model.pkl` | Tuned machine learning weights of the best performing regressor model. |
| `myntra_price_scaler.pkl` | Fitted `StandardScaler` transforming normalization configuration. |
| `myntra_price_encoders.pkl` | Packed dictionary containing fitted `LabelEncoder` objects for columns `brand_name`, `fit_type`, and `fabric_type`. |

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/myntra_price_prediction.git](https://github.com/yourusername/myntra_price_prediction.git)
cd myntra_price_prediction
```
### 2. Initialize a Virtual Environment (Recommended) 
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Required Dependencies
Ensure you have a requirements.txt containing the required dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- jupyter
Install them via pip:
```bash
pip install -r requirements.txt
```
## How to Run
- Execute the Core Training Pipeline via Jupyter
### 1.Launch the notebook instance:
```bash
jupyter notebook
```
### 2.Open myntra_price_ml.ipynb.
### 3.Update the data path to point toward your localized CSV location:
```python
df = pd.read_csv("path/to/myntra_dataset_ByScraping (1).csv")
```
### 4.Run all notebook cells to complete processing and export the .pkl files.
- Load the Pipeline Objects for Staged Inference
```python
import pickle
import pandas as pd

# 1. Load the exported pipeline artifacts
with open('myntra_price_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('myntra_price_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('myntra_price_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
```
### Key Learnings
- Text as a Feature Goldmine: Extracting hidden categorical structural attributes (fit_type, fabric_type) out of text blocks (pants_description) adds deep structural information to pricing predictions.

- Regularization vs Ensemble Performance: While linear and regularized baselines compute quickly, ensemble tree structures capture non-linear market pricing mechanics much better.

- The Importance of Target Scaling Checks: Evaluating variance across standard residual distributions helps locate pricing outbounds and extreme item classes.

### Future Improvements
[ ] Build and launch an interactive GUI frontend using Streamlit.

[ ] Add explainability matrices with SHAP / LIME values to trace feature contributions behind single item evaluations.

[ ] Implement automated deep hyperparameter extraction optimization routines via Optuna.

[ ] Experiment with Tabular Deep Learning frameworks like TabNet to track improvements against Gradient Boosted trees.

HuggingFace deployment link: https://huggingface.co/spaces/Mohanasree-2/myntra_price_prediction
