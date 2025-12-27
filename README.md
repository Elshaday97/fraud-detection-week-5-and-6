## Week 5&6 Challenge: Improved detection of fraud cases for E-Commerce and bank transactions

### Project Objectives

- Detect and reduce fraud in e-commerce and bank transactions using supervised machine learning.
- Provide reproducible preprocessing, feature engineering, model training, evaluation, and inference pipelines.

### Project Structure

- data/
  - raw/
  - processed/
- notebooks/
  - credit_card_modeling.ipynb
  - eda_credit_card.ipynb
  - eda_fraud.ipynb.ipynb
  - feature_engineering.ipynb
  - fraud_modeling.ipynb
- src/
  - data/
    - loader.py
    - preprocessor.py
  - features/
    - features.py
  - modeling/
    - pipeline.py
  - training/
    - train_model.py
- scripts/
  - constants.py
  - decorator.py
- models/
  - fraud_detection/
    - logistic_regression.pkl
    - random_forest.pkl
    - xg_boost.pkl
  - credit_card_fraud_detection/
    - logistic_regression.pkl
    - random_forest.pkl
    - xg_boost.pkl
- reports/
  - fraud_detection/
    - metrics/
    - figures/
  - credit_card_fraud_detection/
    - metrics/
    - figures/
- tests/
- requirements.txt
- Dockerfile

### Setup Guide

Quick start

1. Clone the repo:
   - git clone git@github.com:Elshaday97/fraud-detection-week-5-and-6.git
2. Create environment:
   - pip install -r requirements.txt
3. Place data:
   - Put raw datasets into data/raw/ following naming conventions
4. Run notebooks
   - Run notebooks in order to process data and train models.

### How to Use

1. Load data using src/data/loader.py
2. Preprocess data using src/data/preprocessor.py
3. Engineer features using src/features/features.py
4. Train models using src/training/train_model.py
5. Evaluate models and generate reports in reports/ directory.
