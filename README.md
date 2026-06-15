# Jena-Climate-Dataset
A simple Deep Learning project comparing Simple RNN and LSTM models on the Jena Climate dataset for temperature forecasting. Includes data preprocessing, sequence generation, model training, and performance evaluation using TensorFlow/Keras.

# RNN vs LSTM: Efficiency & Accuracy Comparison on Jena Climate Dataset

Predicting the **24th successive time step** of climate data using vanilla RNN and LSTM networks — comparing their convergence speed, training efficiency, and forecasting accuracy.


## 📌 Overview

This project benchmarks **Simple RNN** against **LSTM** on the [Jena Climate Dataset](https://www.kaggle.com/datasets/mnassrib/jena-climate), a multivariate time series of weather measurements recorded every 10 minutes at the Weather Station in Jena, Germany.

The core task: given a window of past observations, **predict the temperature (T (degC)) at the 24th future time step** — i.e., 4 hours ahead (24 × 10 min).

## 📦 Dataset

**Jena Climate Dataset**
- Source: [Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/)
- Time range: January 2009 – December 2016
- Sampling rate: Every 10 minutes → **~420,000 rows**
- Features: 14 weather variables including temperature, pressure, humidity, wind speed, etc.

> **Target variable:** `T (degC)` — Air temperature in degrees Celsius  
> **Prediction horizon:** Step **+24** (4 hours ahead)

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mnassrib/jena-climate) and place the CSV in the `data/` directory.
