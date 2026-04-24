# ₿ Bitcoin Price Evaluation Analyzer (Pro Series Layout)

A comprehensive, visually-striking dashboard and interactive Streamlit web application. Tailored for crypto analysts doing Bitcoin time-series forecasting, it offers robust model backtesting and predictive trend tracking on a TradingView-styled premium neon interface.

## 📋 Overview

The project acts as an advanced statistical interface enabling users to:
- Instantly ingest Bitcoin historical data via CSV (Kaggle formats) without manual formatting.
- Dynamically split data between Train and Test environments to establish a robust prediction environment.
- Compare predictive capabilities between Statistical Auto-regression (ARIMA) and Machine Learning (Decision Tree) tools.
- Output high-definition, color-delineated visual forecasts separating True Historical sequences, Backtest accuracy, and Unknown Future runs.

## ✨ Features

### Data Ingestion
- **CSV Upload**: Support for Kaggle-style Bitcoin historical data.
- **Auto-Detection**: Automatically identifies Date/Timestamp and Price columns.
- **Validation**: Chronological sorting, missing value handling, data integrity checks.

### Predictive Models
1. **ARIMA** - Statistical autoregressive model for temporal dependencies. Includes user-configurable Order (p, d, q).
2. **Decision Tree Regressor** - A non-linear machine learning approach to evaluating trends based on historical index points.

### Interactive Controls
- **Model Selection**: Choose which model algorithm to evaluate.
- **Forecast Horizon**: A slider to set how many days into the future to predict (between 7 and 90 days).
- **Confidence Intervals**: 50%-99% uncertainty bands available via slider.
- **Data Splitting**: Dynamically adjust the train-test split percentage (e.g. 80/20 train/test ratio).

### Visualization & Aesthetics
- **TradingView-Style Layout**: High-contrast, dark-mode styling with a wide monitor footprint (800px chart height), floating legend, and a professional right-side price Y-axis.
- **Neon Color Palette**: Traces use an Electric Cyan, Neon Orange, and Bright Yellow scheme to clearly distinguish actual data versus predictions without clutter.
- **Plotly Interactive Charts**: High-quality UI with hover details, unified tooltips (x unified), and zoom functionalities.
- **Test-Set Overlay**: Predictions isolated exclusively to the test data segment to clearly visualize backtest accuracy.
- **Future Forecasts**: Out-of-sample future predictions mapped flawlessly at the end of the test boundary.
- **Uncertainty Zones**: Color-matched, shaded confidence interval bands alongside test and future predictions.

### Performance Metrics
- **Mean Absolute Error (MAE)**: Average prediction deviation in USD calculated strictly on the test set.
- **Root Mean Squared Error (RMSE)**: Penalizes larger prediction errors on the test set.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone or download the repository**
   ```bash
   cd BTC
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Using venv
   python -m venv venv
   
   # Activate it
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   The application requires the following Python dependencies (defined in `requirements.txt`):
   - `streamlit==1.28.1`
   - `plotly==5.17.0`
   - `pandas==2.0.3`
   - `numpy==1.24.3`
   - `scikit-learn==1.3.2`
   - `statsmodels==0.14.0`
   - `scipy`
   
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 Dataset

### Recommended Kaggle Dataset
**Bitcoin Historical Data**
- **Link**: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
- **Format**: CSV with columns: Date, Open, High, Low, Close, Volume

### CSV Format Requirements
The uploaded CSV should contain:
- A date/timestamp column (auto-detected from: Date, Timestamp, Time, DateTime).
- A price column (select from: Close, Open, High, Low, Price).
- Optional: Additional columns will be ignored.

## 🔧 Configuration Guide

### Handling Crypto-Market Volatility
Cryptocurrency markets are notoriously volatile, prone to sudden spikes and sharp corrections. This application assesses and handles volatility by offering two contrasting modeling paradigms:

**1. ARIMA (Auto-Regressive Integrated Moving Average):**
- Handles volatility through its differencing parameter `d`, smoothing extreme fluctuations and stabilizing the mean of the time series.
- Employs autoregressive `p` and moving average `q` components to trace momentum and rapidly correct predictions following transient shocks.
- Outputs confidence intervals that naturally widen over time, reflecting increased uncertainty in long-term predictions of volatile assets.

**2. Decision Tree Regressor:**
- Non-linear machine learning algorithm capable of capturing abrupt, asymmetrical price jumps that linear statistical models might miss.
- Isolates distinct price breakpoints without strictly adhering to linear trends, handling structural breaks effectively.
- Residual-based confidence intervals are computed using z-scores to visualize the expected margin of unpredictable market swings during out-of-sample forecasts.

### Model Selection Impact

**ARIMA**
- Best for: Short-term predictions with linear auto-correlation.
- Strengths: Captures temporal dependencies and momentum.
- Recommendation: Tune (p, d, q) based on statistical stationarity properties.

**Decision Tree Regressor**
- Best for: Capturing distinct thresholds and distinct price breakpoints.
- Strengths: Useful for isolating distinct hierarchical data brackets without strictly adhering to linear trends.

## 📈 Interpreting Results

### Understanding the Chart Key
**Trace Colors & Markers:**
- **Electric Cyan (#00E5FF) Line:** The True Actual Historical Data (Both Train & Test period).
- **Neon Orange (#FF9100) Dashed Line:** Model Test Set Predictions (predicted across historical data unseen during training).
- **Translucent Orange Filling:** Confidence Interval for the Backtest/Test partition.
- **Bright Yellow (#FFD600) Dashed Line:** Future Forecast tracing 'X' days beyond the final recorded actual data.
- **Translucent Yellow Filling:** Confidence Interval calculated for Out-of-Sample Future predictions.
- **Light Slate Grey (#90A4AE) Vertical Line:** A clear marker splitting the end of the Train period and the start of the Test boundary.

### Backtesting Metrics Explanation

**MAE (Mean Absolute Error)**
- Average absolute difference between predicted and actual prices.
- Measured in USD.
- Lower is better; consider in context of current price.

**RMSE (Root Mean Squared Error)**
- Penalizes larger errors more heavily than MAE.
- Measured in USD.
- Useful for understanding worst-case prediction errors.

## 🛠️ Technical Architecture

### Dependencies
- **Streamlit**: Web app framework
- **Pandas**: Data manipulation and preprocessing
- **NumPy**: Numerical computations
- **Plotly**: Interactive visualizations
- **Statsmodels**: ARIMA implementation
- **Scikit-learn**: Decision Tree & Evaluation Metrics
- **SciPy**: Statistics computations

### Data Flow
1. User uploads CSV → Automatic format detection.
2. Data validation & chronological sorting.
3. User specifies Train/Test split split ratio.
4. Model trains on training partition.
5. Model generates predictions evaluating strictly to the end of the test dataset.
6. Evaluation metrics resolve against Train vs predictive output.
7. Interactive visualization is rendered reflecting testing overlap.

## 📋 System Requirements
- **OS**: Windows, macOS, Linux
- **Python**: 3.8 or higher
- **RAM**: 2+ GB recommended

## 📄 File Structure
```
BTC/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```
- [x] Plotly interactive visualizations
- [x] Error handling and validation
- [x] Technical indicator support
- [x] Confidence interval bands

---

**Last Updated**: April 24, 2026
**Version**: 1.0
