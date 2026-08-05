# Stock Price Prediction: LSTM vs GRU (FROTO.IS)

This PyTorch project compares multivariate LSTM and GRU models to predict the daily closing price of Ford Otosan (BIST: FROTO.IS). 

## Overview & Methodology
* **Objective:** Predict the next day's closing price using a regression approach.
* **Data:** Sourced dynamically via `yfinance` (Jan 2020 – Jul 2026). It utilizes 12 features, including 20-day OHLCV and technical indicators like SMA, EMA, MACD, and RSI.
* **Method:** Features are scaled using `MinMaxScaler`, formatted into 20-day sliding windows, and fed into 2-layer LSTM or GRU models (`hidden_dim=64`, `dropout=0.2`) using early stopping.

## Quick Start
To run the project, set up a virtual environment, install the `requirements.txt`, and execute `notebooks/Stock_Prediction_LSTM_GRU.ipynb`. The notebook automatically downloads the data, trains the models, and saves output plots (actual vs. predicted and loss curves) to the `results/` folder.

## Results & Limitations
| Model          | Test RMSE | Training Time (sec) |
|----------------|-----------|---------------------|
| LSTM           | 5.083     | 6.38                |
| **GRU**        | **2.690** | 22.29               |
| Naive Baseline | 2.050     | –                   |

* **Performance:** GRU clearly outperformed LSTM in accuracy (47% lower Test RMSE), though it trained slower in this instance because early stopping halted the LSTM much sooner.
* **Limitations:** Neither model beat the naive baseline ("tomorrow's price = today's price"). This reflects the "random walk" nature of short-term stock prices, highlighting the inherent difficulty of the problem rather than a setup flaw.

## Next Steps
Future iterations may explore predicting returns or price direction instead of exact prices, integrating news/sentiment data, and testing alternative architectures like Transformers or TCNs.
