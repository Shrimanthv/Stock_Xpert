# 📈  StockXpert — Stock Movement Prediction using Random Forest Classifier

## Table of Contents
- [Demo](#demo)
- [Overview](#overview)
- [Motivation](#motivation)
- [Features](#features)
- [Installation](#installation)
- [Tech Stack](#tech-stack)
- [Deployment on Streamlit](#deployment-on-streamlit)
- [Project Structure](#project-structure)
- [Bug / Feature Request](#bug--feature-request)
- [Future Scope](#future-scope)
- [Technology Used](#technology-used)
- [Author](#author)
---

## Demo

[🚀 **Live App** – Click here to test the model! Curious what it thinks? Enter your review to see its sentiment prediction.]()


![📷 **Screenshots**: Screenshots of the UI here_]()

---

## Overview

StockXpert is a machine learning-powered stock movement classification system. It predicts whether a stock's price will go Up or Down the next day based on historical features such as closing price, volume, technical indicators, and moving averages. The backend uses a Random Forest Classifier, offering high performance and interpretability.
---

## Motivation

Forecasting the direction of stock price movement helps traders make informed decisions. While deep learning models are popular, classical machine learning models like Random Forests are fast, interpretable, and often perform well when given engineered features. This project demonstrates how Random Forest classification can be effectively applied to financial time series data.

---

## Features

📈 Binary Classification of stock price movement (Up / Down)

🌲 Random Forest Classifier ensures strong generalization and robustness

📊 Feature Engineering including lag features, returns, and moving averages

💬 Streamlit UI for easy interaction and prediction

📁 Upload Your CSV to classify your own stock data

📉 Confusion Matrix and Accuracy Metrics displayed in real time

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock-price-classifier.git
cd stock-price-classifier
```

### 2. Create and Activate a Virtual Environment
Using conda:

```bash
conda create -n stockclf-env python=3.10
conda activate stockclf-env

```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Application
```bash
streamlit run app.py
```
### Tech Stack
| Layer         | Tools Used                                 |
| ------------- | ------------------------------------------ |
| Frontend      | Streamlit                                  |
| Backend       | Python, scikit-learn, Pandas, NumPy        |
| Model         | Random Forest Classifier                   |
| Features      | Lag values, returns, moving averages, etc. |
| Visualization | Matplotlib, Plotly                         |


### Deployment on Streamlit
1. To deploy the app on Streamlit Cloud:

2. Log in or sign up at Streamlit Cloud.

3. Connect your GitHub repository.

4. Select the repo and app.py as the entry file.

5. Make sure requirements.txt is included.

6. Click Deploy.

### Project Structure
``` bash
Movie_Sentiment_Analysis_using_RNN/
│
├── app.py                  # Main Streamlit application
├── model.h5                # Trained RNN model
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├
└── notebooks/              # Jupyter notebooks for EDA & model training
    └── model_training.ipynb
```

## Bug / Feature Request
If you encounter any bugs or have suggestions for new features, please feel free to open an issue on the GitHub repository.

To report a bug:
Provide a clear description of the problem, steps to reproduce it, and any relevant screenshots or error messages.

To request a feature:
Describe the new functionality you'd like to see and explain how it would improve the project.

Your feedback helps improve AeroFare — thank you for contributing!

###  Future Scope
🔀 Support multi-class classification (e.g., Strong Up, Neutral, Down)
🧠 Add support for XGBoost, SVM, or Ensemble Voting
📊 Add feature importance visualization
🌐 Fetch live stock data via API (e.g., Yahoo, Alpha Vantage)
📱 Turn into mobile-ready Streamlit Cloud app


### Technology Used
<p align="center"> <img src="https://www.python.org/static/community_logos/python-logo.png" width="100" title="Python" /><img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width="90" title="scikit-learn" /> <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg" width="140" title="Streamlit" /> <img src="https://pandas.pydata.org/static/img/pandas_mark.svg" width="90" title="Pandas" /> <img src="https://numpy.org/images/logo.svg" width="100" title="NumPy" /> <img src="https://matplotlib.org/_static/logo2_compressed.svg" width="90" title="Matplotlib" /> </p>

## Author
Shrimanth V
Email: shrimanthv99@gmail.com
Feel free to reach out for any questions or collaboration!