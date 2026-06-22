# RNN vs LSTM for Climate Time Series Forecasting

A comparison of Simple Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks for temperature forecasting on the Jena Climate dataset.

## 📌 Project Overview

This project implements and compares **Simple RNN** and **LSTM** architectures to forecast temperature using historical climate observations. The goal is to evaluate how effectively each architecture learns temporal patterns and to understand whether LSTM's long-term memory mechanisms provide a meaningful advantage over a simpler RNN for this task.

## 🎯 Objectives

- Implement and train a Simple RNN model for temperature prediction
- Implement and train an LSTM model for temperature prediction
- Compare both models under identical preprocessing and training conditions
- Evaluate performance using Mean Absolute Error (MAE) and Mean Squared Error (MSE)
- Analyze the impact of longer input sequences on forecasting performance

## 📦 Dataset

**Jena Climate Dataset**
- Source: [Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/)
- Time range: January 2009 – December 2016
- Sampling rate: Every 10 minutes → **~420,000 rows**
- Features: 14 weather variables including temperature, pressure, humidity, wind speed, etc.

**Features used:**
- Temperature (T °C)
- Relative Humidity (%)
- Pressure (mbar)
- Wind Velocity (m/s)
- Wind Direction (deg)


Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mnassrib/jena-climate) and place the CSV in the `data/` directory.

## 🔧 Data Preprocessing

1. Loaded climate data using Pandas
2. Selected relevant weather features
3. Applied feature scaling (StandardScaler / MinMaxScaler)
4. Generated time-series sequences using a sliding window
5. Split data into training and testing sets

**Two experiments were run with different input sequence lengths:**
- Input: Previous 24 time steps → Output: Temperature at the next time step
- Input: Previous 96 time steps → Output: Temperature at the next time step

## 🏗️ Model Architectures

### Simple RNN
```python
model.add(SimpleRNN(64, return_sequences=True, input_shape=(TIME_STEPS, 5)))
model.add(SimpleRNN(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
```
- First RNN layer learns temporal patterns from the input sequence
- Second RNN layer extracts higher-level sequential features
- Dense layer learns complex relationships from extracted features
- Output layer predicts the future temperature value

### LSTM
```python
model.add(LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, 5)))
model.add(LSTM(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
```
- LSTM layers use memory cells and gating mechanisms
- Capable of retaining important information over longer periods
- Helps mitigate the vanishing gradient problem present in traditional RNNs

`TIME_STEPS` was set to **24** and **96** in two separate experiments to study the effect of sequence length on forecasting performance.

## ⚙️ Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Evaluation Metric | Mean Absolute Error (MAE) |
| Batch Size | 32 |
| Epochs | 10 |
| Time Steps | 24 and 96 (compared separately) |

## 📈 Results

### 24 Time Steps

| Model | Test Loss | Test MAE |
|---|---|---|
| RNN | 1.2189e-05 | 0.00228 |
| LSTM | 1.2070e-05 | 0.00233 |

### 96 Time Steps

| Model | Test Loss | Test MAE |
|---|---|---|
| RNN | 0.05409 | 0.18851 |
| LSTM | 0.05403 | 0.18862 |

> **Note:** Test Loss/MAE are not directly comparable across the two settings since the scaling and effective evaluation sample alignment differ between the 24-step and 96-step experiments.

### Observations
- At both 24 and 96 time steps, RNN and LSTM achieved nearly identical performance
- Increasing the sequence length from 24 to 96 time steps did not give LSTM a significant edge over the RNN
- The climate forecasting task appears to depend mainly on short-term temporal patterns
- The Simple RNN was able to capture these patterns effectively at both sequence lengths
- Although LSTM is theoretically better suited for long-term dependencies, its advantage was not substantial for this particular forecasting setup, regardless of input window size

## ✅ Conclusion

This project demonstrates the implementation and comparison of Simple RNN and LSTM architectures for climate time-series forecasting. Experimental results show that both models produce similar prediction accuracy on the Jena Climate dataset, even with longer input sequences. The findings suggest that for short- to medium-range temperature forecasting tasks, a Simple RNN can perform comparably to an LSTM while maintaining a simpler architecture.

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## 🚀 Future Improvements

- Predict temperatures multiple hours ahead
- Increase forecast horizon to capture long-term dependencies
- Incorporate additional climate features
- Experiment with GRU architectures
- Apply hyperparameter tuning and regularization techniques
- Compare results with Transformer-based forecasting models

## 📂 Repository Structure

```
.
├── data/                  # Dataset / data loading scripts
├── requirements/
├── notebooks/             # Jupyter notebooks for experimentation
└── README.md
```

## 📦 Installation

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

## ▶️ Usage

```bash
python src/train_rnn.py
python src/train_lstm.py
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
