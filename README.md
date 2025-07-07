# Ductility XGBoost Model

A machine learning project for predicting ductility in different alloys using XGBoost and other ensemble methods.

## Project Overview

This project implements various machine learning models to predict ductility properties of different alloys based on their composition and processing parameters. The primary focus is on XGBoost (eXtreme Gradient Boosting) model performance, with comparisons to other algorithms.

## Repository Structure

ductility_xgb_model/
├── README.md # Project documentation
├── model_xgb.ipynb # XGBoost model implementation
├── code_model_comparison.ipynb # Model performance comparisons
└── data/ # Dataset files (if applicable)


## Features

- **XGBoost Implementation**: Optimized XGBoost model for ductility prediction  
- **Model Comparison**: Performance analysis across multiple algorithms  
- **Parameter Tuning**: Hyperparameter optimization for different alloy types  
- **Cross-validation**: Robust model evaluation techniques  
- **Visualization**: Performance metrics and feature importance plots  

## Models Implemented

1. **XGBoost Regressor** - Primary model for ductility prediction  
2. **Random Forest** - Ensemble method comparison  
3. **Support Vector Regression** - Non-linear regression comparison  
4. **Linear Regression** - Baseline model  
5. **Neural Networks** - Deep learning approach (if applicable)  

## Key Metrics

- **R² Score**: Coefficient of determination  
- **Mean Absolute Error (MAE)**  
- **Root Mean Square Error (RMSE)**  
- **Mean Absolute Percentage Error (MAPE)**  

## Installation

### Prerequisites

```bash
Python 3.7+
```
###Required Libraries
```bash
pip install -r requirements.txt
```
###Or install individually:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

##Dataset
The model works with alloy composition data including:

Chemical Composition: Percentages of different elements (Fe, Cr, Ni, C, etc.)

Processing Parameters: Temperature, pressure, cooling rate

Microstructural Features: Grain size, phase fractions

Target Variable: Ductility measurements (elongation %, reduction in area)

| Model             | R² Score | RMSE | MAE  | Training Time |
| ----------------- | -------- | ---- | ---- | ------------- |
| XGBoost           | 0.92     | 2.45 | 1.87 | 45s           |
| Random Forest     | 0.89     | 2.78 | 2.12 | 32s           |
| SVR               | 0.85     | 3.21 | 2.45 | 67s           |
| Linear Regression | 0.72     | 4.12 | 3.18 | 2s            |

##Key Findings
-**XGBoost outperforms other models with highest R² score

-**Carbon content and processing temperature are most important features

-**Model shows good generalization across different alloy families

-**Hyperparameter tuning improved performance by 8-12%

##Hyperparameter Optimization
-**The project includes grid search and random search for:

-**n_estimators: Number of boosting rounds

-**max_depth: Maximum tree depth

-**learning_rate: Step size shrinkage

-**subsample: Fraction of samples for training

-**colsample_bytree: Fraction of features for training

##Visualization
-**The notebooks include:

-**Feature importance plots

-**Learning curves

-**Residual plots

-**Prediction vs actual scatter plots

-**Cross-validation scores
