# 📈 FinSight: Stock Market Forecasting with LSTM  

### 📘 Overview  
FinSight is a machine learning project exploring the use of **Long Short-Term Memory (LSTM)** networks to model and forecast stock price movements.  
Using historical **Apple (AAPL)** stock data, this project implements and compares **PyTorch** and **TensorFlow** LSTM architectures to predict short-term trends and evaluate the performance of sequence-based deep learning models on financial time series data.  

---

### 🧠 Objectives  
- Build an **end-to-end ML pipeline** for stock price prediction — from data processing to visualization.  
- Compare **PyTorch** and **TensorFlow** LSTM models on the same dataset.  
- Experiment with **sequence windowing**, **feature scaling**, and **loss minimization** for time series forecasting.  
- Analyze performance limitations and identify directions for improvement.  

---

### 🧩 Methods  
- **Data:** Apple stock historical prices obtained via public APIs or CSV.  
- **Preprocessing:** Feature scaling using MinMax normalization and time window segmentation.  
- **Modeling:**  
  - *PyTorch Model:* Sequential LSTM network with hidden layers and dropout regularization.  
  - *TensorFlow Model:* Comparable LSTM implementation for cross-framework evaluation.  
- **Evaluation:**  
  - Visualized **predicted vs. actual** stock price trends.  
  - Computed **MSE** and **RMSE** metrics for model comparison.  

---

### 📊 Key Insights  
- LSTM networks captured general stock movement trends but struggled with short-term volatility.  
- **Scaling mismatches** and **overfitting** influenced prediction accuracy — valuable insights for model tuning.  
- The **TensorFlow model** produced smoother trend approximations, while the **PyTorch model** was more sensitive to scaling and noise.  
- Demonstrated importance of proper **window size selection**, **feature normalization**, and **loss optimization** in time series forecasting.  

---

### 💻 Tech Stack  
`Python` | `PyTorch` | `TensorFlow` | `Pandas` | `NumPy` | `Matplotlib` | `Scikit-learn`

---

### 🚀 Next Steps  
- Incorporate **sentiment data** (news headlines, social media) as additional features.  
- Test **Transformer-based** architectures for improved long-range dependencies.  
- Expand the model to multiple assets for portfolio-level prediction.  
