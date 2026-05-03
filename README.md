# House Price Prediction

Predicting house prices in Bengaluru using Linear Regression.

## Dataset
Bengaluru House Price dataset from Kaggle — 13,000+ property listings with features like location, size, bathrooms, and area.

## What I did
- Exploratory Data Analysis (EDA)
- Data cleaning — handled missing values, removed duplicates
- Feature engineering — extracted BHK and property type from size column, handled range values in total_sqft
- Outlier removal using price per sqft and bhk filters
- One-hot encoding for categorical features
- Log transformation on target variable (price)
- Trained Linear Regression model

## Results
- R² Score: 0.74
- MAE and RMSE calculated on actual price scale (lakhs)

## Project Structure
