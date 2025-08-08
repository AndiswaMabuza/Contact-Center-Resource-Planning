import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing

# --- Page Configuration ---
st.set_page_config(
    page_title="Mukuru Contact Center Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Contact Center Resource Planning Dashboard")
st.markdown("This dashboard showcases a comprehensive analysis of contact center data, including **forecasting**, **simulation**, and strategic **planning**.")

# --- Helper Functions ---
@st.cache_data
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2025-01-01', freq='H')
    num_intervals = len(dates)
    baseline_calls = 50
    weekly_seasonality = np.sin(np.linspace(0, 2 * np.pi, 24*7)) * 20
    daily_seasonality = np.sin(np.linspace(0, 2 * np.pi, 24)) * 10
    long_term_trend = np.linspace(0, 10, num_intervals)
    aht_baseline = 300
    calls_received = (baseline_calls + np.tile(weekly_seasonality, num_intervals // (24 * 7) + 1)[:num_intervals] * 1.5 + np.tile(daily_seasonality, num_intervals // 24 + 1)[:num_intervals] * 1.0 + long_term_trend + np.random.normal(0, 15, num_intervals)).astype(int)
    calls_received = np.maximum(0, calls_received)
    aht = (aht_baseline + np.random.normal(0, 30, num_intervals)).astype(int)
    aht = np.maximum(100, aht)
    df = pd.DataFrame({'timestamp': dates, 'calls_received': calls_received, 'aht_seconds': aht})
    return df

@st.cache_data
def perform_forecasting(df):
    forecast_periods = 90 * 24
    model_calls = ExponentialSmoothing(df['calls_received'], trend='add', seasonal='add', seasonal_periods=24*7).fit()
    forecast_calls = model_calls.forecast(forecast_periods)
    model_aht = ExponentialSmoothing(df['aht_seconds'], trend='add', seasonal='add', seasonal_periods=24).fit()
    forecast_aht = model_aht.forecast(forecast_periods)
    forecast_dates = pd.date_range(start=df['timestamp'].iloc[-1], periods=forecast_periods + 1, freq='H')[1:]
    forecast_df = pd.DataFrame({'timestamp': forecast_dates, 'forecasted_calls': np.maximum(0, forecast_calls).astype(int), 'forecasted_aht': np.maximum(100, forecast_aht).astype(int)})
    return forecast_df

def calculate_required_agents(calls, aht, shrinkage):
    erlangs = (calls * aht) / 3600
    agents_needed_raw = erlangs + (np.sqrt(erlangs) * (1 - 0.80)) # SLA is hardcoded for simplicity
    agents_needed_with_shrinkage = agents_needed_raw / (1 - shrinkage)
    return np.ceil(agents_needed_with_shrinkage)

def simulate_spike_by_number(forecast_df, start_date_str, spike_increase_calls, spike_duration_days):
    spiked_df = forecast_df.copy()
    start_date = pd.to_datetime(start_date_str)
    end_date = start_date + pd.Timedelta(days=spike_duration_days)
    spike_mask = (spiked_df['timestamp'] >= start_date) & (spiked_df['timestamp'] < end_date)
    if not spike_mask.any():
        st.warning("The spike date range is outside the forecast period.")
        return spiked_df
    spiked_df.loc[spike_mask, 'forecasted_calls'] = (spiked_df.loc[spike_mask, 'forecasted_calls'] + spike_increase_calls).astype(int)
    return spiked_df

def generate_insights(df_original, df_simulated=None, spike_increase_calls=None, spike_start=None, spike_duration=None):
    peak_agents_original = df_original['required_agents'].max()
    peak_date_original = df_original.loc[df_original['required_agents'].idxmax(), 'timestamp'].strftime('%Y-%m-%d')
    avg_agents_original = df_original['required_agents'].mean()
    insights = f"""
    ### Core Forecasting Insights
    - **Baseline Staffing:** The forecast suggests a baseline staffing need of approximately **{int(avg_agents_original)}** agents to meet service levels.
    - **Peak Demand:** The highest demand is projected to be around **{int(peak_agents_original)}** agents, occurring on or near **{peak_date_original}**. This is a critical period that requires careful attention to scheduling and adherence.
    """
    if df_simulated is not None:
        peak_agents_spiked = df_simulated['required_agents'].max()
        staffing_increase = peak_agents_spiked - peak_agents_original
        insights += f"""
    ### Spike Simulation: Key Takeaways
    - **Simulated Event:** A hypothetical event adding **{spike_increase_calls} extra calls per hour** over **{spike_duration} days**, starting **{spike_start}**, was simulated.
    - **Impact on Staffing:** This spike would require an additional **{int(staffing_increase)}** agents at its peak, reaching a new total peak of **{int(peak_agents_spiked)}** agents.
    ### Strategic Recommendations
    - **Proactive Scheduling:** For a planned event, pre-schedule shifts for the additional **{int(staffing_increase)}** agents.
    - **Real-Time Agility:** For an unplanned event, communicate with the Real-Time Analyst team to activate an emergency protocol.
    """
    return insights

# --- Main App Logic ---
df = generate_data()

# --- Section 1: Exploratory Data Analysis (EDA) ---
st.header("1. Exploratory Data Analysis (EDA)")
st.write("A deep dive into the historical data to understand key patterns and metrics.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Descriptive Statistics")
    st.write(df['calls_received'].describe().to_frame().T)
    st.write(df['aht_seconds'].describe().to_frame().T)

with col2:
    st.subheader("Data Distribution")
    fig_dist = make_subplots(rows=1, cols=2, subplot_titles=("Distribution of Calls Received", "Distribution of AHT"))
    fig_dist.add_trace(go.Histogram(x=df['calls_received'], name='Calls'), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=df['aht_seconds'], name='AHT'), row=1, col=2)
    fig_dist.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

st.subheader("Time Series Decomposition")
decomposition_calls = seasonal_decompose(df['calls_received'], model='additive', period=24*7)
fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Original Calls", "Trend", "Weekly Seasonality", "Residuals"))
fig_decomp.add_trace(go.Scatter(x=df['timestamp'], y=decomposition_calls.observed, name='Original'), row=1, col=1)
fig_decomp.add_trace(go.Scatter(x=df['timestamp'], y=decomposition_calls.trend, name='Trend'), row=2, col=1)
fig_decomp.add_trace(go.Scatter(x=df['timestamp'], y=decomposition_calls.seasonal, name='Seasonality'), row=3, col=1)
fig_decomp.add_trace(go.Scatter(x=df['timestamp'], y=decomposition_calls.resid, name='Residuals'), row=4, col=1)
fig_decomp.update_layout(height=800, title_text="Time Series Decomposition of Calls Received")
st.plotly_chart(fig_decomp, use_container_width=True)

st.subheader("Seasonality Analysis")
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()
avg_calls_by_hour = df.groupby('hour')['calls_received'].mean()
avg_calls_by_day = df.groupby('day_of_week')['calls_received'].mean()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
fig_season = make_subplots(rows=1, cols=2, subplot_titles=("Average Calls by Hour of Day", "Average Calls by Day of Week"))
fig_season.add_trace(go.Bar(x=avg_calls_by_hour.index, y=avg_calls_by_hour.values), row=1, col=1)
fig_season.add_trace(go.Bar(x=days_order, y=avg_calls_by_day.reindex(days_order).values), row=1, col=2)
fig_season.update_layout(title_text="Seasonality Analysis", height=400)
st.plotly_chart(fig_season, use_container_width=True)

st.markdown("---")

# --- Section 2: Forecasting ---
st.header("2. Forecasting")
st.write("Using a time series model to forecast future call volumes and AHT for the next quarter.")
forecast_df = perform_forecasting(df)
fig_forecast = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Calls Received Forecast", "AHT Forecast"))
fig_forecast.add_trace(go.Scatter(x=df['timestamp'], y=df['calls_received'], mode='lines', name='Historical Calls'), row=1, col=1)
fig_forecast.add_trace(go.Scatter(x=forecast_df['timestamp'], y=forecast_df['forecasted_calls'], mode='lines', name='Forecasted Calls'), row=1, col=1)
fig_forecast.add_trace(go.Scatter(x=df['timestamp'], y=df['aht_seconds'], mode='lines', name='Historical AHT'), row=2, col=1)
fig_forecast.add_trace(go.Scatter(x=forecast_df['timestamp'], y=forecast_df['forecasted_aht'], mode='lines', name='Forecasted AHT'), row=2, col=1)
fig_forecast.update_layout(height=600, title_text="Contact Center Forecasts")
st.plotly_chart(fig_forecast, use_container_width=True)

# --- Bug Fix: Calculate agents for original forecast once ---
original_forecast_df = forecast_df.copy()
original_forecast_df['required_agents'] = calculate_required_agents(original_forecast_df['forecasted_calls'], original_forecast_df['forecasted_aht'], shrinkage=0.3)

st.markdown("---")

# --- Section 3: Interactive Spike Simulation ---
st.header("3. Interactive Spike Simulation")
st.write("Model the impact of a sudden surge in demand to determine the required staffing changes.")
st.markdown("Use the controls in the sidebar to define the simulation event.")

with st.sidebar:
    st.header("Simulation Parameters")
    spike_start = st.date_input("Spike Start Date", pd.to_datetime(original_forecast_df['timestamp'].iloc[60]))
    spike_increase_calls = st.number_input("Extra Calls per Hour", min_value=0, max_value=200, value=50, step=10)
    spike_duration = st.slider("Spike Duration (days)", 1, 30, 7)

spiked_forecast_df = simulate_spike_by_number(original_forecast_df, str(spike_start), spike_increase_calls, spike_duration)

fig_spike = go.Figure()
fig_spike.add_trace(go.Scatter(x=original_forecast_df['timestamp'], y=original_forecast_df['required_agents'], mode='lines', name='Original Forecast'))
fig_spike.add_trace(go.Scatter(x=spiked_forecast_df['timestamp'], y=spiked_forecast_df['required_agents'], mode='lines', name='Spike Simulation'))
fig_spike.update_layout(title_text=f"Staffing Forecast with a Fixed Spike of {spike_increase_calls} Calls/Hour", xaxis_title="Date", yaxis_title="Number of Agents")
st.plotly_chart(fig_spike, use_container_width=True)

st.markdown("---")

# --- Section 4: Dynamic Insights and Recommendations ---
st.header("4. Strategic Insights & Recommendations")
st.write("Based on the analysis and simulation, here are the key insights and actionable recommendations.")
insights_text = generate_insights(original_forecast_df, spiked_forecast_df, spike_increase_calls, str(spike_start), spike_duration)
st.markdown(insights_text)
