import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# 1. CONFIG & THEME
st.set_page_config(
    page_title="JetEngine AI - Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: #0a0e1a;
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #0f1419;
        border-right: 1px solid #1a2332;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    
    /* Logo area */
    .logo-container {
        padding: 1.5rem 1.5rem 2rem 1.5rem;
        border-bottom: 1px solid #1a2332;
        margin-bottom: 2rem;
    }
    
    .logo-title {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .logo-subtitle {
        color: #6b7280;
        font-size: 0.875rem;
        margin: 0.25rem 0 0 0;
        font-weight: 400;
    }
    
    /* Sidebar section headers */
    .sidebar-section {
        color: #6b7280;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 2rem 0 1rem 0;
        padding: 0 1.5rem;
    }
    
    /* Selectbox styling */
    [data-testid="stSidebar"] .stSelectbox > label {
        color: #9ca3af !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        background: #1a2332;
        border: 1px solid #2a3442;
        border-radius: 8px;
    }
    
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]:hover {
        border-color: #3a4452;
    }
    
    /* Sidebar info boxes */
    .sidebar-info {
        background: #1a2332;
        border: 1px solid #2a3442;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .sidebar-info-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.5rem 0;
    }
    
    .sidebar-info-label {
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .sidebar-info-value {
        color: #ffffff;
        font-size: 1.125rem;
        font-weight: 700;
    }
    
    /* Main content header */
    .main-header {
        padding: 0 0 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-title {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    
    .main-subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin: 0;
        font-weight: 400;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #6b7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="metric-container"] {
        background: #141b26;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #1e2937;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    
    /* Generate button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        color: white;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.875rem 1.5rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
        transition: all 0.2s ease;
        letter-spacing: 0.01em;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(6, 182, 212, 0.4);
        background: linear-gradient(135deg, #0891b2 0%, #2563eb 100%);
    }
    
    /* Status banner */
    .status-banner {
        background: linear-gradient(135deg, #854d0e 0%, #92400e 100%);
        border: 1px solid #a16207;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 2rem 0;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        box-shadow: 0 4px 12px rgba(133, 77, 14, 0.3);
    }
    
    .status-banner.healthy {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border-color: #059669;
        box-shadow: 0 4px 12px rgba(6, 95, 70, 0.3);
    }
    
    .status-banner.critical {
        background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%);
        border-color: #dc2626;
        box-shadow: 0 4px 12px rgba(153, 27, 27, 0.3);
    }
    
    .status-icon {
        font-size: 2rem;
    }
    
    .status-content {
        flex: 1;
    }
    
    .status-title {
        color: #fbbf24;
        font-size: 1.125rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        letter-spacing: 0.02em;
    }
    
    .status-banner.healthy .status-title {
        color: #34d399;
    }
    
    .status-banner.critical .status-title {
        color: #fca5a5;
    }
    
    .status-message {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.875rem;
        margin: 0;
    }
    
    /* Chart container */
    .chart-section {
        background: #141b26;
        border: 1px solid #1e2937;
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    
    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .chart-title {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0;
    }
    
    .chart-legend {
        display: flex;
        gap: 1.5rem;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        color: #9ca3af;
    }
    
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
    
    .legend-dot.healthy {
        background: #10b981;
    }
    
    .legend-dot.warning {
        background: #f59e0b;
    }
    
    .legend-dot.critical {
        background: #ef4444;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #141b26 !important;
        border: 1px solid #1e2937 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #2a3442 !important;
    }
    
    .streamlit-expanderContent {
        background: #0f1419 !important;
        border: 1px solid #1e2937 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        color: #9ca3af !important;
    }
    
    /* Success/Warning messages */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border-left: 4px solid #10b981 !important;
        border-radius: 8px !important;
        color: #d1fae5 !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 8px !important;
        color: #fef3c7 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #06b6d4 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Logo
st.sidebar.markdown("""
<div class="logo-container">
    <div class="logo-title">JetEngine AI</div>
    <div class="logo-subtitle">Predictive Maintenance</div>
</div>
""", unsafe_allow_html=True)

RUL_CAP = 125

# 2. LOAD PREDICTIONS (CACHED)
@st.cache_data
def load_pred_df():
    return joblib.load("pred_df.pkl")

pred_df = load_pred_df()

# 3. HEALTH COMPUTATION (FAST & CACHED)
@st.cache_data
def compute_health_series(engine_id):
    df_e = (
        pred_df[pred_df["engine_id"] == engine_id]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    if len(df_e) < 2:
        return None

    cycles = df_e["cycle"].values
    rul = np.minimum(df_e["pred_rul"].values, RUL_CAP)

    # STABLE health definition
    # Normalize relative to first visible cycle
    health = (rul / rul[0]) * 100
    health = np.clip(health, 0, 100)
    health = np.minimum.accumulate(health)

    # Enforce physical degradation
    health = np.minimum.accumulate(health)

    return cycles, rul, health

# 4. SIDEBAR
st.sidebar.markdown('<div class="sidebar-section">SELECT ENGINE</div>', unsafe_allow_html=True)

@st.cache_data
def get_valid_engines(min_points=5):
    return (
        pred_df.groupby("engine_id")
               .size()
               .loc[lambda s: s >= min_points]
               .index
               .tolist()
    )

valid_engines = get_valid_engines(min_points=5)

selected_id = st.sidebar.selectbox(
    "Engine ID",
    valid_engines,
    format_func=lambda x: f"Engine #{x}"
)

# Sidebar info
st.sidebar.markdown(f"""
<div class="sidebar-info">
    <div class="sidebar-info-item">
        <span class="sidebar-info-label">Total Engines:</span>
        <span class="sidebar-info-value">{len(valid_engines)}</span>
    </div>
    <div class="sidebar-info-item">
        <span class="sidebar-info-label">Analyzing:</span>
        <span class="sidebar-info-value">#{selected_id}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Performance metrics box
st.sidebar.markdown("""
<div class="sidebar-info" style="margin-top: 2rem;">
    <div style="color: #ffffff; font-weight: 600; margin-bottom: 1rem; font-size: 0.95rem;">Performance:</div>
    <div style="margin: 0.75rem 0;">
        <span style="color: #9ca3af; font-size: 0.875rem;">• LSTM RMSE: </span>
        <span style="color: #ffffff; font-weight: 600;">13.05</span>
    </div>
    <div style="margin: 0.75rem 0;">
        <span style="color: #9ca3af; font-size: 0.875rem;">• LSTM R²: </span>
        <span style="color: #ffffff; font-weight: 600;">0.8940</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# 5. MAIN LOGIC
# ==========================================================
result = compute_health_series(selected_id)

if result is None:
    st.warning("Not enough data to display this engine.")
    st.stop()

cycles, rul, health = result

current_cycle = int(cycles[-1])
current_rul = rul[-1]
current_health = health[-1]

# Main Header
st.markdown(f"""
<div class="main-header">
    <h1 class="main-title">Engine #{selected_id} Analysis</h1>
    <p class="main-subtitle">LSTM-based remaining useful life prediction</p>
</div>
""", unsafe_allow_html=True)

# 6. METRICS
col1, col2 = st.columns(2)
with col1:
    st.metric("PREDICTED RUL", f"{current_rul:.1f} cycles")
with col2:
    st.metric("HEALTH SCORE", f"{current_health:.1f} %")

# 7. GENERATE BUTTON
generate_button = st.button("▶  Generate Health Analysis", use_container_width=True)

# 8. STATUS INDICATOR & HEALTH GRAPH
if generate_button:
    with st.spinner("Generating health degradation analysis..."):
        # Status Banner
        if current_health >= 70:
            status_class = "healthy"
            status_icon = "✓"
            status_title = "HEALTHY"
            status_msg = "Engine operating within normal parameters"
        elif current_health >= 30:
            status_class = ""
            status_icon = "⚠"
            status_title = "MAINTENANCE REQUIRED"
            status_msg = "Schedule preventive maintenance soon"
        else:
            status_class = "critical"
            status_icon = "⚠"
            status_title = "CRITICAL FAILURE RISK"
            status_msg = "Immediate maintenance intervention required"
        
        st.markdown(f"""
        <div class="status-banner {status_class}">
            <div class="status-icon">{status_icon}</div>
            <div class="status-content">
                <div class="status-title">{status_title}</div>
                <div class="status-message">{status_msg}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chart Section
        st.markdown("""
        <div class="chart-section">
            <div class="chart-header">
                <div class="chart-title">Health Degradation Trend</div>
                <div class="chart-legend">
                    <div class="legend-item">
                        <div class="legend-dot healthy"></div>
                        <span>Healthy</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot warning"></div>
                        <span>Warning</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot critical"></div>
                        <span>Critical</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        plot_df = pd.DataFrame({
            "Cycle": cycles,
            "Health": health
        })

        # Background regions
        healthy_band = alt.Chart(
            pd.DataFrame({"y1": [70], "y2": [100]})
        ).mark_rect(
            opacity=0.15,
            color="#10b981"
        ).encode(
            y="y1:Q",
            y2="y2:Q"
        )

        warning_band = alt.Chart(
            pd.DataFrame({"y1": [30], "y2": [70]})
        ).mark_rect(
            opacity=0.15,
            color="#f59e0b"
        ).encode(
            y="y1:Q",
            y2="y2:Q"
        )

        critical_band = alt.Chart(
            pd.DataFrame({"y1": [0], "y2": [30]})
        ).mark_rect(
            opacity=0.15,
            color="#ef4444"
        ).encode(
            y="y1:Q",
            y2="y2:Q"
        )

        # Health line with gradient
        health_line = alt.Chart(plot_df).mark_line(
            color="#06b6d4",
            strokeWidth=3
        ).encode(
            x=alt.X("Cycle:Q", title="Cycle", axis=alt.Axis(
                labelColor='#9ca3af',
                titleColor='#9ca3af',
                gridColor='#1e2937',
                domainColor='#1e2937'
            )),
            y=alt.Y("Health:Q", title="Health (%)", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(
                labelColor='#9ca3af',
                titleColor='#9ca3af',
                gridColor='#1e2937',
                domainColor='#1e2937'
            ))
        )

        # Area under line
        health_area = alt.Chart(plot_df).mark_area(
            color="#06b6d4",
            opacity=0.2
        ).encode(
            x="Cycle:Q",
            y="Health:Q"
        )

        # Points on line
        health_points = alt.Chart(plot_df).mark_circle(
            size=60,
            color="#06b6d4",
            opacity=0.8
        ).encode(
            x="Cycle:Q",
            y="Health:Q",
            tooltip=[
                alt.Tooltip("Cycle:Q", title="Cycle"),
                alt.Tooltip("Health:Q", title="Health %", format=".1f")
            ]
        )

        chart = (
            healthy_band +
            warning_band +
            critical_band +
            health_area +
            health_line +
            health_points
        ).properties(
            height=400
        ).configure_view(
            strokeWidth=0,
            fill='#141b26'
        ).configure_axis(
            grid=True
        )

        st.altair_chart(chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.success("✓ Health analysis generated successfully!")

# 9. MODEL INFO
with st.expander("⚙ Model Architecture & Logic"):
    st.markdown("""
    **Architecture:**
    - Stacked LSTM (96 → 48 units)  
    - 15 sensors + degradation slopes  
    - Sliding window = 30 cycles  
    - RUL capped at 125 cycles  
    - Health derived from normalized degradation  
    - Inference precomputed (offline)
    
    **Performance:**
    - LSTM RMSE: 13.05
    - LSTM R²: 0.8940
    """)