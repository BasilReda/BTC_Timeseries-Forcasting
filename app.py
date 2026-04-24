import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as stats

st.set_page_config(page_title="Bitcoin Data Analyzer", layout="wide")
st.title("Bitcoin Data Analyzer & Forecaster")
st.write("Upload your Bitcoin historical data to get started.")

@st.cache_data
def load_and_preprocess_data(file_obj):
    df = pd.read_csv(file_obj)
    df.columns = df.columns.str.strip()
    date_candidates = ["Date", "date", "Timestamp", "timestamp", "Open time", "Time", "time"]
    found_date_col = None
    for col in date_candidates:
        if col in df.columns:
            found_date_col = col
            break
            
    if found_date_col:
        df[found_date_col] = pd.to_datetime(df[found_date_col], errors="coerce").dt.date
        df = df.dropna(subset=[found_date_col])

    price_candidates = ["Open", "Close", "Low", "High", "Price", "price", "Adj Close", "Volume"]
    found_price_cols = []
    for col in price_candidates:
        if col in df.columns:
            found_price_cols.append(col)
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
    if found_price_cols:
        df = df.dropna(subset=found_price_cols, how='all')
        
    return df, found_date_col, found_price_cols

st.sidebar.header("1. Data & Model Setup")
uploaded_file = st.sidebar.file_uploader("Upload Bitcoin CSV file", type=["csv"])

if uploaded_file is not None:
    df, found_date_col, found_price_cols = load_and_preprocess_data(uploaded_file)
    
    if not found_date_col or not found_price_cols:
        st.sidebar.error("Incompatible CSV format: Could not find a recognizable Date/Time column (e.g., 'Date', 'Timestamp') or USD Price column (e.g., 'Open', 'Close', 'Price'). Please check your Kaggle data columns.")
        st.stop()
    df = df.dropna(subset=found_price_cols + [found_date_col], how='any')

    st.sidebar.subheader("Data Configuration")
    date_options = [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]) or col == found_date_col]
    if found_date_col in date_options:
        default_date_idx = date_options.index(found_date_col)
    else:
        default_date_idx = 0
        
    selected_date_col = st.sidebar.selectbox("Select Date Column (X-axis)", date_options, index=default_date_idx)
        
    target_options = found_price_cols
    selected_target_col = st.sidebar.selectbox("Select Target Column (Y-axis)", target_options, index=0)
    df = df.sort_values(by=selected_date_col)
    df[selected_date_col] = pd.to_datetime(df[selected_date_col])
    df = df.groupby(selected_date_col).last()
    df = df.resample('D').ffill().reset_index()
    
    st.sidebar.subheader("Model Configuration")
    model_choice = st.sidebar.selectbox("Select Prediction Model", ["ARIMA", "Decision Tree"])
    
    if model_choice == "ARIMA":
        st.sidebar.write("**ARIMA Order (p, d, q)**")
        p_col, d_col, q_col = st.sidebar.columns(3)
        with p_col: p_order = st.number_input("p", min_value=0, value=1, step=1)
        with d_col: d_order = st.number_input("d", min_value=0, value=1, step=1)
        with q_col: q_order = st.number_input("q", min_value=0, value=1, step=1)
        arima_order = (p_order, d_order, q_order)
    
    st.sidebar.subheader("Evaluation Settings")
    forecast_days = st.sidebar.slider("Forecast Horizon (Days into future)", min_value=7, max_value=90, value=30, step=1)
    conf_interval_pct = st.sidebar.slider("Confidence Interval (%)", min_value=50, max_value=99, value=95)
    alpha = 1.0 - (conf_interval_pct / 100.0) # Used for stats calculations
        
    st.sidebar.write("**Data Splitting**")
    train_size_pct = st.sidebar.slider("Training Data Split (%)", min_value=50, max_value=95, value=80)

    st.write("### Data Preview")
    st.dataframe(df.tail(5), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Latest Date", str(df[selected_date_col].dt.date.iloc[-1]))
    col3.metric(f"Current {selected_target_col}", f"${df[selected_target_col].iloc[-1]:,.2f}")
    
    st.write("---")

    st.write("### C. Forecasting Engine")
    if st.button("Generate Forecast", type="primary", use_container_width=True) or 'run_model' in st.session_state:
        st.session_state['run_model'] = True
        with st.spinner(f"Training {model_choice} model..."):
            y = df[selected_target_col].values
            dates = df[selected_date_col].values
            train_size = int(len(df) * (train_size_pct / 100.0))
            train_y, test_y = y[:train_size], y[train_size:]
            train_dates, test_dates = dates[:train_size], dates[train_size:]
            last_date = pd.to_datetime(dates[-1])
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
            all_forecast_dates = np.concatenate([test_dates, future_dates])

            if model_choice == "ARIMA":
                model = ARIMA(
                    train_y, 
                    order=arima_order, 
                    enforce_stationarity=False, 
                    enforce_invertibility=False
                )
                fitted_model = model.fit(method_kwargs={"maxiter": 200})
                forecast_result = fitted_model.get_forecast(steps=len(test_y) + forecast_days)
                forecast_values = forecast_result.predicted_mean
                conf_int = forecast_result.conf_int(alpha=alpha)
                
                test_predictions = forecast_values[:len(test_y)]
                future_predictions = forecast_values[len(test_y):]
                
                lower_bound = conf_int[:, 0]
                upper_bound = conf_int[:, 1]
                test_lower_bound = lower_bound[:len(test_y)]
                test_upper_bound = upper_bound[:len(test_y)]
                future_lower_bound = lower_bound[len(test_y):]
                future_upper_bound = upper_bound[len(test_y):]

            elif model_choice == "Decision Tree":
                X = np.arange(len(df)).reshape(-1, 1)
                train_X = X[:train_size]
                test_X = np.arange(train_size, len(df)).reshape(-1, 1)
                future_X = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
                dt_model = DecisionTreeRegressor(random_state=42)
                dt_model.fit(train_X, train_y)
                test_predictions = dt_model.predict(test_X)
                future_predictions = dt_model.predict(future_X)
                train_predictions_dt = dt_model.predict(train_X)
                residuals = train_y - train_predictions_dt
                std_resid = np.std(residuals)
                
                z_score = stats.norm.ppf(1 - alpha/2)
                margin_of_error = z_score * std_resid
                
                test_lower_bound = test_predictions - margin_of_error
                test_upper_bound = test_predictions + margin_of_error
                future_lower_bound = future_predictions - margin_of_error
                future_upper_bound = future_predictions + margin_of_error

            mae = mean_absolute_error(test_y, test_predictions)
            rmse = np.sqrt(mean_squared_error(test_y, test_predictions))
            
            st.write("#### Performance Metrics (Test Set)")
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("MAE (Mean Absolute Error)", f"${mae:,.2f}")
            m_col2.metric("RMSE (Root Mean Squared Error)", f"${rmse:,.2f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=y,
                mode='lines', name='Actual Data (Train & Test)',
                line=dict(color='#00E5FF', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=test_dates, y=test_predictions,
                mode='lines', name=f'{model_choice} Test Set Predictions',
                line=dict(color='#FF9100', width=2.5, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=np.concatenate([test_dates, test_dates[::-1]]),
                y=np.concatenate([test_upper_bound, test_lower_bound[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 145, 0, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f'{conf_interval_pct}% Confidence Interval (Test)'
            ))
            if len(future_dates) > 0:
                conn_dates = np.concatenate([[test_dates[-1]], future_dates])
                conn_values = np.concatenate([[test_predictions[-1]], future_predictions])
                
                fig.add_trace(go.Scatter(
                    x=conn_dates, y=conn_values,
                    mode='lines', name=f'{model_choice} (Future Forecast)',
                    line=dict(color='#FFD600', width=3, dash='dash')
                ))

                conn_upper = np.concatenate([[test_upper_bound[-1]], future_upper_bound])
                conn_lower = np.concatenate([[test_lower_bound[-1]], future_lower_bound])

                # 5. Plot Confidence Intervals for Future Forecast (Translucent Yellow)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([conn_dates, conn_dates[::-1]]),
                    y=np.concatenate([conn_upper, conn_lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255, 214, 0, 0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f'{conf_interval_pct}% Confidence Interval (Future)'
                ))
            fig.add_vline(x=test_dates[0], line_width=1, line_dash="dash", line_color="#90A4AE")
            fig.add_annotation(x=test_dates[0], y=1.05, yref="paper", text="Test Set Start", showarrow=False, font=dict(color="#90A4AE"))
            fig.update_layout(
                title=f"<b>Bitcoin Price Evaluation</b><br><sup>Model: {model_choice} | Target: {selected_target_col}</sup>",
                xaxis_title="",
                yaxis_title="<b>Price (USD)</b>",
                template="plotly_dark",
                hovermode="x unified",
                height=800,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="left",
                    x=0,
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(0,0,0,0)"
                ),
                margin=dict(l=40, r=40, t=140, b=40),
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                xaxis=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                    showline=False, zeroline=False,
                    showspikes=True, spikemode="across", spikethickness=1, spikedash="solid", spikecolor="#94a3b8"
                ),
                yaxis=dict(
                    showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickprefix="$",
                    showline=False, zeroline=False,
                    side="right",
                    showspikes=True, spikemode="across", spikethickness=1, spikedash="solid", spikecolor="#94a3b8"
                )
            )

            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Awaiting CSV file upload...")