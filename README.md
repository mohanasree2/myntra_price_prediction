# Myntra Pants Price Prediction — ML Pipeline
This repository contains a complete Machine Learning pipeline for predicting the prices of pants listed on Myntra. The project involves data scraping, extensive Exploratory Data Analysis (EDA), feature engineering from text descriptions, and comparing multiple regression models to find the most accurate predictor.
# Project Overview
The goal is to predict the Sale Price (Regression) of products based on their brand, MRP, ratings, and physical characteristics extracted from their descriptions.
## Dataset Features
The dataset consists of 52,120 records with the following initial features:
brand_name: The manufacturer/brand of the product.
pants_description: Textual description of the item.
price: The actual selling price (Target Variable).
MRP: The Maximum Retail Price.
discount_percent: The percentage of discount offered.
ratings: Average user rating.
number_of_ratings: Total volume of reviews.
# Pipeline Steps
## 1. Exploratory Data Analysis (EDA)
   Univariate Analysis: Distribution of prices, MRPs, and ratings.
   Bivariate Analysis: Relationship between MRP vs. Price and Brand vs. Average Price.
   Multivariate Analysis: Correlation heatmaps and discount bucket segmentation.
## 2. Feature Engineering
Since the description column contained valuable information, two new features were extracted:
fit_type: Extracted styles such as Slim, Relaxed, Skinny, Regular, Straight, etc.
fabric_type: Extracted materials like Cotton, Denim, and Linen.
## 3. Data Preprocessing
Encoding: Categorical variables (brand_name, fit_type, fabric_type) were transformed using LabelEncoder.
Scaling: Features were normalized using StandardScaler to ensure model stability.
Splitting: Data was split into 80% Training and 20% Testing sets.
# Model Performance & Comparison
Ten different regression algorithms were trained and evaluated using R-squared ($R^2$), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE)
