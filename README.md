# Credit Card Fraud Detection Using Classification Techniques

A beginner-friendly Streamlit project that uses Logistic Regression to classify a transaction as genuine or potentially fraudulent.

## Dataset

`dataset/creditcard.csv` contains 400 real transactions sampled from the public [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), collected by the Machine Learning Group at ULB in collaboration with Worldline.

The sample contains 250 genuine transactions and 150 fraudulent transactions, selected reproducibly with a fixed random seed. The original source has 284,807 rows, including 492 fraud cases. This small, deliberately enriched sample is intended for a college demonstration; its class balance and evaluation metrics do not represent real-world fraud prevalence.

Features `V1` through `V28` are anonymized principal components. `Time` and `Amount` are the other transaction features. `Class` is the target: `0` means genuine and `1` means fraudulent.

## Installation

Open a terminal in this project folder and install the requirements:

```powershell
python -m pip install -r requirements.txt
```

## Train the model

```powershell
python train.py
```

This trains Logistic Regression, prints the test results, and saves the model.

## Run the application

```powershell
streamlit run app.py
```

Streamlit opens the application in your browser. Use the sidebar to navigate between Home / Dashboard, Model Training, Batch Prediction, and Single Prediction.

## Model

`train.py` uses an 80/20 stratified train-test split and a Scikit-learn Pipeline containing `StandardScaler` and Logistic Regression with `class_weight="balanced"`. The scaler and model are fitted using training rows only. `app.py` loads and uses that same saved pipeline for individual and batch predictions.

For batch predictions, upload a CSV with the same feature columns as the training data. The `Class` column is optional and is never used as a prediction feature.
## Application Screenshot
## Application Screenshot

![Credit Card Fraud Detection App](app-screenshot.png)
