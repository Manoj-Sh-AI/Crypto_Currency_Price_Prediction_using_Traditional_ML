# Crypto_Currency_Price_Prediction_using_Traditional_ML

Time series crypto price prediction using traditional ML regression. Shows how strong feature engineering (lags, rolling stats, preprocessing) enables high performance without neural networks, saving compute and time. Trained on 6 years of data, tested on 2 years, achieving **R² ≈ 0.99**.

<img src="https://github.com/Manoj-Sh-AI/Crypto_Currency_Price_Prediction_using_Traditional_ML/blob/7c282962416f5f4b206b16625849211820c62548/images/1728742499_.jpg?raw=true" height="50%" width="100%" />

---

## 📌 Project Overview

This project focuses on **multi-asset cryptocurrency price prediction** using **traditional machine learning regression models** applied to time series data.

Instead of relying on deep learning models, this work demonstrates how **strong feature engineering combined with classical ML algorithms** can deliver **near state-of-the-art performance** while drastically reducing **compute cost, training time, and deployment complexity**.

The pipeline is implemented for **five major cryptocurrencies**:

- **BTCUSDT** — Bitcoin  
- **ETHUSDT** — Ethereum  
- **SOLUSDT** — Solana  
- **XRPUSDT** — Ripple  
- **DOGEUSDT** — Dogecoin  

Each asset is modeled independently using the same **robust time series forecasting pipeline**, ensuring consistency and scalability.

---

## 🎯 Objectives

- Build a **scalable forecasting pipeline** for multiple cryptocurrencies  
- Apply **traditional regression algorithms** to time series prediction  
- Demonstrate the power of **feature engineering over model complexity**  
- Achieve **high accuracy without neural networks**  
- Maintain **low compute, fast training, and high interpretability**

---

## 📊 Dataset

- **Assets:** BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT  
- **Training Period:** 6 years of historical data  
- **Testing Period:** 2 years (holdout validation)  
- **Frequency:** 15-minute interval data  
- **Raw Features:**
  - Open  
  - High  
  - Low  
  - Close  
  - Volume  
  - Turnover  

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Missing value handling  
- Timestamp indexing  
- Log scaling for volume & turnover  
- EMA smoothing to reduce market noise  

### 2. Feature Engineering
- Lag features: `t-1 → t-n`
- Rolling window statistics (mean, std, min, max)
- Exponential Moving Averages (EMA)
- Log transformations
- Trend & volatility features

### 3. Modeling Approach

Traditional regression models used:

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor *(optional)*  

Each cryptocurrency is trained and evaluated **independently** using identical modeling pipelines.

---

## 🧪 Validation Strategy

- **TimeSeriesSplit Cross Validation**
- Chronological train-test split:
  - 6 years → Training
  - 2 years → Testing
- Strict prevention of **data leakage**

---

## 🏆 Results

| Asset | Test R² |
|---------|-----------|
| BTCUSDT | ≈ 0.99 |
| ETHUSDT | ≈ 0.99 |
| SOLUSDT | ≈ 0.99 |
| XRPUSDT | ≈ 0.99 |
| DOGEUSDT | ≈ 0.99 |

> **Key Insight:**  
> With strong feature engineering, **traditional ML regression models can rival deep learning performance** — while using **a fraction of the compute and training time**.

---

## 🚀 Why No Neural Networks?

This project intentionally avoids deep learning models to show that:

- Feature engineering > Model complexity  
- Traditional ML:
  - Trains faster
  - Requires far less compute
  - Is easier to tune
  - Is easier to interpret
  - Deploys easily on low-resource systems

This approach is especially beneficial for:
- Real-time trading systems
- Edge deployment
- Low-latency inference pipelines
- Cost-sensitive production environments

---

## 🛠 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Matplotlib, Seaborn**

---

## 📁 Project Structure

