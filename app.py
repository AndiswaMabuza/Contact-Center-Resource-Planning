import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing
import math
from typing import Tuple

# --- Page Configuration ---
st.set_page_config(
    page_title="Contact Center Resource Planning Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
FORECAST_DAYS = 90
HOURS_IN_DAY = 24
FORECAST_PERIODS = FORECAST_DAYS * HOURS_IN_DAY
WEEKLY_SEASONALITY_PERIOD = HOURS_IN_DAY * 7

# --- Helper Functions ---

@st.cache_data
def generate_data() -> pd.DataFrame:
    """
    Generates synthetic contact center data with trend and seasonality.

    Returns:
        pd.DataFrame: A DataFrame with 'timestamp', 'calls_received', and 'aht_seconds'.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2025-01-01', freq='H')
    num_intervals = len(dates)
    
    baseline_calls = 50
    weekly_seasonality = np.sin(np.linspace(0, 2 * np.pi, WEEKLY_SEASONALITY_PERIOD)) * 20
    daily_seasonality = np.sin(np.linspace(0, 2 * np.pi, HOURS_IN_DAY)) * 10
    long_term_trend = np.linspace(0, 10, num_intervals)
    aht_baseline = 300
    
    calls_received = (
        baseline_calls + 
        np.tile(weekly_seasonality, num_intervals // WEEKLY_SEASONALITY_PERIOD + 1)[:num_intervals] * 1.5 + 
        np.tile(daily_seasonality, num_intervals // HOURS_IN_DAY + 1)[:num_intervals] * 1.0 + 
        long_term_trend + 
        np.random.normal(0, 15, num_intervals)
    ).astype(int)
    calls_received = np.maximum(0, calls_received)
    
    aht = (aht_baseline + np.random.normal(0, 30, num_intervals)).astype(int)
    aht = np.maximum(100, aht)
    
    df = pd.DataFrame({'timestamp': dates, 'calls_received': calls_received, 'aht_seconds': aht})
    return df

@st.cache_data
def perform_forecasting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits an Exponential Smoothing model and forecasts future call volume and AHT.

    Args:
        df (pd.DataFrame): DataFrame with historical data.

    Returns:
        pd.DataFrame: DataFrame with forecasted data.
    """
    model_calls = ExponentialSmoothing(df['calls_received'], trend='add', seasonal='add', seasonal_periods=WEEKLY_SEASONALITY_PERIOD).fit()
    forecast_calls = model_calls.forecast(FORECAST_PERIODS)
    
    model_aht = ExponentialSmoothing(df['aht_seconds'], trend='add', seasonal='add', seasonal_periods=HOURS_IN_DAY).fit()
    forecast_aht = model_aht.forecast(FORECAST_PERIODS)
    
    forecast_dates = pd.date_range(start=df['timestamp'].iloc[-1], periods=FORECAST_PERIODS + 1, freq='H')[1:]
    forecast_df = pd.DataFrame({
        'timestamp': forecast_dates, 
        'forecasted_calls': np.maximum(0, forecast_calls).astype(int), 
        'forecasted_aht': np.maximum(100, forecast_aht).astype(int)
    })
    return forecast_df

def erlang_c(traffic_intensity: float, num_agents: int) -> float:
    """Calculates the probability of a call being queued (Erlang C formula)."""
    if traffic_intensity <= 0 or num_agents <= 0 or num_agents <= traffic_intensity:
        return 1.0
    
    erlang_b = traffic_intensity / num_agents
    for i in range(1, int(num_agents)):
        erlang_b = (traffic_intensity * erlang_b) / (i + traffic_intensity * erlang_b)
        
    prob_wait = (num_agents * erlang_b) / (num_agents - traffic_intensity * (1 - erlang_b))
    return prob_wait

def calculate_service_level(traffic_intensity: float, num_agents: int, aht_seconds: int, target_time_seconds: int) -> float:
    """Calculates the expected service level."""
    if num_agents <= traffic_intensity:
        return 0.0
    prob_wait = erlang_c(traffic_intensity, num_agents)
    service_level = 1.0 - (prob_wait * math.exp(-(num_agents - traffic_intensity) * (target_time_seconds / aht_seconds)))
    return service_level

def calculate_required_agents_erlang(calls_per_hour: int, aht_seconds: int, service_level_target: float, target_time_seconds: int) -> int:
    """Calculates required agents using the Erlang C formula by iterating to find the minimum agents."""
    if calls_per_hour == 0:
        return 0
        
    traffic_intensity = (calls_per_hour * aht_seconds) / 3600.0
    
    # Start with a base number of agents and iterate up
    required_agents = int(math.ceil(traffic_intensity))
    
    while True:
        current_sl = calculate_service_level(traffic_intensity, required_agents, aht_seconds, target_time_seconds)
        if current_sl >= service_level_target or required_agents > calls_per_hour * 2: # Safety break
            break
        required_agents += 1
    return required_agents

# --- Main App Logic ---
st.title("Contact Center Resource Planning Dashboard")
st.markdown("This dashboard showcases a comprehensive analysis of contact center data, including **forecasting**, **simulation**, and strategic **planning**.")

df = generate_data()

# --- Section 1: Exploratory Data Analysis (EDA) ---
st.header("1. Exploratory Data Analysis (EDA)")
st.write("A deep dive into the historical data to understand key patterns and metrics.")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Descriptive Statistics")
    st.write(df[['calls_received', 'aht_seconds']].describe())
with col2:
    st.subheader("Data Distribution")
    fig_dist = make_subplots(rows=1, cols=2, subplot_titles=("Distribution of Calls Received", "Distribution of AHT"))
    fig_dist.add_trace(go.Histogram(x=df['calls_received'], name='Calls'), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=df['aht_seconds'], name='AHT'), row=1, col=2)
    fig_dist.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)
st.subheader("Time Series Decomposition")
decomposition_calls = seasonal_decompose(df.set_index('timestamp')['calls_received'], model='additive', period=WEEKLY_SEASONALITY_PERIOD)
fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Observed", "Trend", "Seasonality", "Residuals"))
fig_decomp.add_trace(go.Scatter(x=decomposition_calls.observed.index, y=decomposition_calls.observed, name='Observed'), row=1, col=1)
fig_decomp.add_trace(go.Scatter(x=decomposition_calls.trend.index, y=decomposition_calls.trend, name='Trend'), row=2, col=1)
fig_decomp.add_trace(go.Scatter(x=decomposition_calls.seasonal.index, y=decomposition_calls.seasonal, name='Seasonality'), row=3, col=1)
fig_decomp.add_trace(go.Scatter(x=decomposition_calls.resid.index, y=decomposition_calls.resid, name='Residuals', mode='markers'), row=4, col=1)
fig_decomp.update_layout(height=800, title_text="Time Series Decomposition of Calls Received")
st.plotly_chart(fig_decomp, use_container_width=True)

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

st.markdown("---")

# --- Section 3: Interactive Simulation and Staffing ---
st.header("3. Interactive Staffing & Spike Simulation")
st.write("Model staffing requirements based on service level targets and simulate the impact of a sudden demand surge.")
st.markdown("Use the controls in the sidebar to define service goals and the simulation event.")

with st.sidebar:
    st.header("Planning Parameters")
    
    st.subheader("Service Level Goals")
    sl_target = st.slider("Service Level Target (%)", 50, 99, 80, 1) / 100.0
    time_target = st.slider("Answer Time Target (seconds)", 10, 60, 20, 5)

    st.subheader("Shrinkage")
    shrinkage_input = st.slider("Shrinkage (%)", 0, 50, 30, 5) / 100.0
    
    st.subheader("Simulation Parameters")
    min_date = forecast_df['timestamp'].iloc[0].date()
    max_date = forecast_df['timestamp'].iloc[-1].date()
    spike_start = st.date_input("Spike Start Date", min_value=min_date, max_value=max_date, value=min_date)
    spike_increase_calls = st.number_input("Extra Calls per Hour", min_value=0, max_value=500, value=50, step=10)
    spike_duration = st.slider("Spike Duration (days)", 1, 30, 7)

# Calculate agents needed for the original forecast
original_forecast_df = forecast_df.copy()
agents_raw = original_forecast_df.apply(
    lambda row: calculate_required_agents_erlang(
        row['forecasted_calls'], 
        row['forecasted_aht'], 
        sl_target, 
        time_target
    ), axis=1
)
original_forecast_df['required_agents'] = np.ceil(agents_raw / (1 - shrinkage_input))

# Simulate the spike
spiked_forecast_df = original_forecast_df.copy()
spike_start_ts = pd.to_datetime(spike_start)
spike_end_ts = spike_start_ts + pd.Timedelta(days=spike_duration)
spike_mask = (spiked_forecast_df['timestamp'] >= spike_start_ts) & (spiked_forecast_df['timestamp'] < spike_end_ts)

if spike_mask.any():
    spiked_forecast_df.loc[spike_mask, 'forecasted_calls'] += spike_increase_calls
    
    # Recalculate agents for the spiked period
    agents_raw_spiked = spiked_forecast_df.apply(
        lambda row: calculate_required_agents_erlang(
            row['forecasted_calls'], 
            row['forecasted_aht'], 
            sl_target, 
            time_target
        ), axis=1
    )
    spiked_forecast_df['required_agents'] = np.ceil(agents_raw_spiked / (1 - shrinkage_input))
else:
    st.warning("The selected spike date range is outside the forecast period.")

# Plotting the results
fig_spike = go.Figure()
fig_spike.add_trace(go.Scatter(x=original_forecast_df['timestamp'], y=original_forecast_df['required_agents'], mode='lines', name='Baseline Staffing'))
fig_spike.add_trace(go.Scatter(x=spiked_forecast_df['timestamp'], y=spiked_forecast_df['required_agents'], mode='lines', name='Spike Simulation Staffing', line=dict(dash='dot')))
fig_spike.update_layout(
    title_text=f"Staffing Forecast with Spike Simulation",
    xaxis_title="Date",
    yaxis_title="Number of Required Agents"
)
st.plotly_chart(fig_spike, use_container_width=True)

st.markdown("---")

# --- Section 4: Dynamic Insights and Recommendations ---
st.header("4. Strategic Insights & Recommendations")
st.write("Based on the analysis and simulation, here are the key insights and actionable recommendations.")

def generate_insights(df_original, df_simulated, **kwargs):
    peak_agents_original = df_original['required_agents'].max()
    peak_date_original = df_original.loc[df_original['required_agents'].idxmax(), 'timestamp'].strftime('%Y-%m-%d')
    avg_agents_original = df_original['required_agents'].mean()
    
    insights = f"""
    ### Core Planning Insights
    - **Service Goal:** The plan aims for **{int(kwargs['sl_target']*100)}%** of calls to be answered within **{kwargs['time_target']} seconds**.
    - **Baseline Staffing:** The forecast suggests a baseline staffing need of approximately **{int(avg_agents_original)} agents** on average.
    - **Peak Demand:** The highest demand is projected to require **{int(peak_agents_original)} agents**, occurring on or near **{peak_date_original}**.
    - **Shrinkage Assumption:** All calculations account for a shrinkage rate of **{int(kwargs['shrinkage'] * 100)}%**.
    """
    
    if not df_simulated.equals(df_original):
        peak_agents_spiked = df_simulated.loc[spike_mask, 'required_agents'].max()
        staffing_increase = peak_agents_spiked - df_original.loc[spike_mask, 'required_agents'].max()
        
        insights += f"""
        ### Spike Simulation: Key Takeaways
        - **Simulated Event:** A hypothetical event adding **{kwargs['spike_increase_calls']} extra calls per hour** over **{kwargs['spike_duration']} days**, starting **{kwargs['spike_start']}**.
        - **Impact on Staffing:** This spike would require up to **{int(staffing_increase)} additional agents** during the event, reaching a new peak of **{int(peak_agents_spiked)} agents**.
        
        ### Strategic Recommendations
        - **Proactive Scheduling:** For a planned marketing campaign or known event, pre-schedule shifts for the additional **~{int(staffing_increase)}** agents to maintain service levels.
        - **Contingency Planning:** For unplanned outages or viral events, have a contingency plan. This could involve cross-training staff from other departments or having on-call schedules ready to be activated.
        """
    return insights

insights_params = {
    'sl_target': sl_target,
    'time_target': time_target,
    'shrinkage': shrinkage_input,
    'spike_increase_calls': spike_increase_calls,
    'spike_duration': spike_duration,
    'spike_start': str(spike_start)
}
insights_text = generate_insights(original_forecast_df, spiked_forecast_df, **insights_params)
st.markdown(insights_text)

# --- Footer ---
st.markdown("---")
st.markdown("""
    <p style='text-align: center;'>
    Developed by Andiswa Mabuza | 
    <a href='mailto:Amabuza53@gmail.com'>Email</a> | 
    <a href='https://andiswamabuza.vercel.app' target='_blank'>Developer Site</a>
    </p>
    """, unsafe_allow_html=True)
