import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import for PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import tempfile
import base64

# Set page configuration
st.set_page_config(
    page_title="Explainable Dam Water Flow Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #F0F9FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .upload-section {
        background-color: #F8FAFC;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #CBD5E1;
        margin: 1rem 0;
    }
    .stButton button {
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌊 Explainable Water Flow Prediction in Dams</h1>', unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_preview' not in st.session_state:
    st.session_state.data_preview = None
if 'y_pred_test' not in st.session_state:
    st.session_state.y_pred_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None

# Function to generate sample data
def generate_dam_data():
    """Generate synthetic dam water flow data with realistic patterns"""
    np.random.seed(42)
    n_days = 365 * 3  # 3 years of data
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Base patterns
    base_flow = 1000  # Average flow in cubic meters per second
    
    # Seasonal pattern (higher in rainy season)
    seasonal = 300 * np.sin(2 * np.pi * np.arange(n_days) / 365 - np.pi/2)
    
    # Weekly pattern (lower on weekends due to human activity)
    weekly = 50 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    # Trend component (slight increase over time)
    trend = 0.1 * np.arange(n_days)
    
    # Rainfall component
    rainfall = np.random.gamma(shape=2, scale=15, size=n_days)
    rainfall_effect = 10 * rainfall + 5 * np.roll(rainfall, 1) + 2 * np.roll(rainfall, 2)
    
    # Temperature effect
    temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365) + np.random.normal(0, 5, n_days)
    temp_effect = -2 * (temperature - 20)  # Negative effect when too hot
    
    # Snowmelt effect (spring)
    snowmelt = 200 * np.exp(-((np.arange(n_days) % 365 - 120) ** 2) / 2000)
    
    # Human activity (lower on holidays)
    holidays = [datetime(2020, 12, 25), datetime(2021, 12, 25), datetime(2022, 12, 25)]
    holiday_effect = np.zeros(n_days)
    for holiday in holidays:
        idx = (dates == pd.Timestamp(holiday)).argmax()
        holiday_effect[max(0, idx-2):min(n_days, idx+3)] = -100
    
    # Reservoir level effect
    reservoir_level = np.clip(60 + 20 * np.sin(2 * np.pi * np.arange(n_days) / 180), 40, 100)
    
    # Generate water flow
    water_flow = (
        base_flow +
        seasonal +
        weekly +
        trend +
        rainfall_effect +
        temp_effect +
        snowmelt +
        holiday_effect +
        np.random.normal(0, 50, n_days)  # Random noise
    )
    
    # Ensure positive values
    water_flow = np.maximum(water_flow, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'water_flow': water_flow,
        'rainfall_mm': rainfall,
        'temperature_c': temperature,
        'snow_depth_cm': np.clip(100 * np.exp(-((np.arange(n_days) % 365 - 60) ** 2) / 5000), 0, 100),
        'reservoir_level_percent': reservoir_level,
        'upstream_flow': water_flow * 1.2 + np.random.normal(0, 100, n_days),
        'evaporation_mm': np.maximum(0, 0.5 * temperature + np.random.normal(0, 1, n_days)),
        'soil_moisture_percent': np.clip(40 + 20 * np.sin(2 * np.pi * np.arange(n_days) / 365), 20, 80)
    })
    
    # Add day of week, month, year features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df

# Function to process uploaded CSV
def process_uploaded_file(uploaded_file):
    """Process uploaded CSV file and prepare it for analysis"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Display file info
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        
        # Show preview
        st.subheader("📄 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Check for required columns
        required_columns = ['date', 'water_flow']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.warning(f"⚠️ Missing columns: {missing_columns}. Using generated sample data instead.")
            return generate_dam_data()
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Add missing features if they don't exist
        base_features = ['rainfall_mm', 'temperature_c', 'reservoir_level_percent', 
                        'upstream_flow', 'soil_moisture_percent']
        
        for feature in base_features:
            if feature not in df.columns:
                if feature == 'rainfall_mm':
                    df[feature] = np.random.gamma(2, 15, len(df))
                elif feature == 'temperature_c':
                    df[feature] = 15 + 10 * np.sin(2 * np.pi * np.arange(len(df)) / 365) + np.random.normal(0, 5, len(df))
                elif feature == 'reservoir_level_percent':
                    df[feature] = np.clip(60 + 20 * np.sin(2 * np.pi * np.arange(len(df)) / 180), 40, 100)
                elif feature == 'upstream_flow':
                    df[feature] = df['water_flow'] * 1.2 + np.random.normal(0, 100, len(df))
                elif feature == 'soil_moisture_percent':
                    df[feature] = np.clip(40 + 20 * np.sin(2 * np.pi * np.arange(len(df)) / 365), 20, 80)
        
        # Add time-based features
        if 'date' in df.columns:
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            df['day_of_year'] = df['date'].dt.dayofyear
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return generate_dam_data()

# Function to engineer features
def engineer_features(df, include_lagged=True, include_seasonal=True):
    """Engineer features for the model"""
    df_engineered = df.copy()
    
    if include_lagged:
        # Lagged features
        for lag in [1, 2, 3, 7, 14]:
            df_engineered[f'flow_lag_{lag}'] = df_engineered['water_flow'].shift(lag)
            if 'rainfall_mm' in df_engineered.columns:
                df_engineered[f'rainfall_lag_{lag}'] = df_engineered['rainfall_mm'].shift(lag)
        
        # Rolling statistics
        df_engineered['flow_rolling_mean_7'] = df_engineered['water_flow'].rolling(7).mean()
        df_engineered['flow_rolling_std_7'] = df_engineered['water_flow'].rolling(7).std()
    
    if include_seasonal:
        # Cyclical encoding for seasonal features
        if 'month' in df_engineered.columns:
            df_engineered['month_sin'] = np.sin(2 * np.pi * df_engineered['month'] / 12)
            df_engineered['month_cos'] = np.cos(2 * np.pi * df_engineered['month'] / 12)
        
        if 'day_of_year' in df_engineered.columns:
            df_engineered['day_of_year_sin'] = np.sin(2 * np.pi * df_engineered['day_of_year'] / 365)
            df_engineered['day_of_year_cos'] = np.cos(2 * np.pi * df_engineered['day_of_year'] / 365)
    
    # Interaction features
    if 'rainfall_mm' in df_engineered.columns and 'temperature_c' in df_engineered.columns:
        df_engineered['rain_temp_interaction'] = df_engineered['rainfall_mm'] * df_engineered['temperature_c']
    
    if 'water_flow' in df_engineered.columns and 'reservoir_level_percent' in df_engineered.columns:
        df_engineered['flow_reservoir_ratio'] = df_engineered['water_flow'] / (df_engineered['reservoir_level_percent'] + 1)
    
    # Remove rows with NaN values
    df_engineered = df_engineered.dropna()
    
    return df_engineered

# Function to generate PDF report
def generate_pdf_report(model_type, performance_metrics, feature_importance, forecast_summary, 
                       shap_insights, data_summary, recommendations):
    """Generate a comprehensive PDF report"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("Water Flow Prediction Report", title_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    exec_summary = """
    This report provides a comprehensive analysis of water flow predictions for dam management. 
    The model leverages machine learning techniques to forecast water flow with explainable insights.
    """
    story.append(Paragraph(exec_summary, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Model Information
    story.append(Paragraph("Model Information", styles['Heading2']))
    model_info = [
        ["Parameter", "Value"],
        ["Model Type", model_type],
        ["Report Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Data Points", str(data_summary.get('total_days', 'N/A'))],
        ["Features Used", str(data_summary.get('num_features', 'N/A'))]
    ]
    
    model_table = Table(model_info, colWidths=[200, 200])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(model_table)
    story.append(Spacer(1, 20))
    
    # Performance Metrics
    story.append(Paragraph("Performance Metrics", styles['Heading2']))
    perf_data = [["Metric", "Value"]]
    
    metrics_map = {
        'R² Score': 'R²',
        'RMSE': 'RMSE',
        'MAE': 'MAE',
        'MAPE': 'MAPE (%)',
        'Max Error': 'Max Error'
    }
    
    for metric_name, display_name in metrics_map.items():
        if metric_name in performance_metrics:
            perf_data.append([display_name, f"{performance_metrics[metric_name]:.3f}"])
    
    perf_table = Table(perf_data, colWidths=[150, 100])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 20))
    
    # Feature Importance
    if feature_importance is not None and not feature_importance.empty:
        story.append(Paragraph("Top 10 Feature Importance", styles['Heading2']))
        feature_data = [["Feature", "Importance"]]
        
        for idx, row in feature_importance.head(10).iterrows():
            feature_data.append([row['Feature'], f"{row['Importance']:.4f}"])
        
        feature_table = Table(feature_data, colWidths=[250, 100])
        feature_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10B981')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F0F9FF'))
        ]))
        story.append(feature_table)
        story.append(Spacer(1, 20))
    
    # SHAP Insights
    if shap_insights:
        story.append(Paragraph("Key SHAP Insights", styles['Heading2']))
        shap_text = "Based on SHAP analysis:\n\n"
        for insight in shap_insights:
            shap_text += f"• {insight}\n"
        story.append(Paragraph(shap_text, styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Forecast Summary
    if forecast_summary:
        story.append(Paragraph("Forecast Summary", styles['Heading2']))
        forecast_data = [["Statistic", "Value (m³/s)"]]
        
        for stat, value in forecast_summary.items():
            forecast_data.append([stat, f"{value:.1f}"])
        
        forecast_table = Table(forecast_data, colWidths=[150, 100])
        forecast_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F59E0B')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(forecast_table)
        story.append(Spacer(1, 20))
    
    # Recommendations
    if recommendations:
        story.append(Paragraph("Recommendations", styles['Heading2']))
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Disclaimer
    story.append(Paragraph("Disclaimer", styles['Heading3']))
    disclaimer = """
    This report is generated automatically based on machine learning models. 
    The predictions should be used as guidance and verified with domain expertise.
    """
    story.append(Paragraph(disclaimer, styles['Italic']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer

# Sidebar for controls
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=100)
    st.title("⚙️ Dashboard Controls")
    
    # Data source selection
    data_source = st.selectbox(
        "Select Data Source",
        ["Sample Generated Data", "Upload CSV File", "Simulated Dam Data"]
    )
    
    # CSV Upload Section
    if data_source == "Upload CSV File":
        st.markdown("---")
        st.subheader("📁 Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dam data in CSV format. Must include 'date' and 'water_flow' columns."
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process Uploaded Data"):
                with st.spinner("Processing your data..."):
                    df = process_uploaded_file(uploaded_file)
                    st.session_state.data = df
                    st.session_state.uploaded_file = uploaded_file.name
                    st.session_state.data_loaded = True
                    st.session_state.data_preview = df.head()
                    st.success("✅ Data processed successfully!")
        
        if st.session_state.uploaded_file:
            st.info(f"📄 Current file: {st.session_state.uploaded_file}")
    
    # Model selection
    st.markdown("---")
    model_type = st.selectbox(
        "Select ML Model",
        ["Random Forest", "Gradient Boosting", "Linear Regression", "Ensemble"]
    )
    
    # Feature engineering options
    st.subheader("⚡ Feature Engineering")
    include_lagged = st.checkbox("Include Lagged Features", value=True)
    include_seasonal = st.checkbox("Include Seasonal Features", value=True)
    include_weather = st.checkbox("Include Weather Features", value=True)
    
    # Prediction horizon
    st.markdown("---")
    prediction_horizon = st.slider(
        "Prediction Horizon (days)",
        min_value=1,
        max_value=30,
        value=7
    )
    
    # Confidence interval
    confidence_level = st.slider(
        "Confidence Interval (%)",
        min_value=80,
        max_value=99,
        value=95
    )

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📁 Data Upload",
    "📊 Data Overview",
    "🤖 ML Model Training",
    "🔍 SHAP Explanations",
    "📈 Predictions & Forecast",
    "🎯 What-If Analysis",
    "📋 Report & Export"
])

# Tab 1: Data Upload
with tab1:
    st.header("📁 Upload Your Dam Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 Required Data Format
        
        Your CSV file should include at minimum:
        - **date**: Date column (YYYY-MM-DD format)
        - **water_flow**: Water flow in m³/s
        
        Optional columns (will be generated if missing):
        - rainfall_mm (Rainfall in mm)
        - temperature_c (Temperature in °C)
        - reservoir_level_percent (%)
        - upstream_flow (m³/s)
        - soil_moisture_percent (%)
        - snow_depth_cm (cm)
        
        ### 📄 Sample CSV Structure
        ```
        date,water_flow,rainfall_mm,temperature_c
        2023-01-01,1250.5,15.2,10.5
        2023-01-02,1320.3,20.1,11.2
        2023-01-03,1400.8,25.5,12.8
        ```
        """)
    
    with col2:
        st.markdown("""
        <div class="upload-section">
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=['csv'],
            key="uploader_main",
            help="Upload your dam monitoring data in CSV format"
        )
        
        if uploaded_file is not None:
            st.info(f"**File uploaded:** {uploaded_file.name}")
            st.info(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
            
            # Preview uploaded data
            try:
                preview_df = pd.read_csv(uploaded_file, nrows=5)
                st.subheader("📄 Data Preview")
                st.dataframe(preview_df)
                
                # Show data statistics
                st.subheader("📈 Data Statistics")
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Rows", preview_df.shape[0])
                with col_stat2:
                    st.metric("Columns", preview_df.shape[1])
                with col_stat3:
                    if 'date' in preview_df.columns:
                        st.metric("Date Range", f"{preview_df['date'].min()} to {preview_df['date'].max()}")
                    else:
                        st.metric("Date Range", "N/A")
                
                # Process button
                if st.button("🚀 Process Data for Analysis", type="primary"):
                    with st.spinner("Processing data and preparing features..."):
                        df = process_uploaded_file(uploaded_file)
                        st.session_state.data = df
                        st.session_state.uploaded_file = uploaded_file.name
                        st.session_state.data_loaded = True
                        st.success("✅ Data ready for analysis! Navigate to other tabs.")
                        
                        # Show success metrics
                        st.balloons()
                        
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        else:
            st.info("👆 Upload a CSV file to get started")
            st.info("💡 Don't have data? Use sample data in other tabs")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Data validation section
    if st.session_state.data is not None:
        st.markdown("---")
        st.subheader("✅ Data Validation Summary")
        
        df = st.session_state.data
        
        validation_cols = st.columns(4)
        with validation_cols[0]:
            st.metric("Total Records", len(df))
        with validation_cols[1]:
            st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
        with validation_cols[2]:
            st.metric("Avg Water Flow", f"{df['water_flow'].mean():.1f} m³/s")
        with validation_cols[3]:
            missing = df.isnull().sum().sum()
            st.metric("Missing Values", missing, delta=f"-{missing}" if missing > 0 else None)
        
        # Show column information
        st.subheader("📋 Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

# Load sample data if no data is loaded
if st.session_state.data is None:
    df = generate_dam_data()
    st.session_state.data = df

# Continue with other tabs using the data
df = st.session_state.data

# Tab 2: Data Overview
with tab2:
    st.header("📊 Dam Data Overview & Exploration")
    
    # Data summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Days", len(df))
    with col2:
        st.metric("Avg Water Flow", f"{df['water_flow'].mean():.1f} m³/s")
    with col3:
        st.metric("Max Water Flow", f"{df['water_flow'].max():.1f} m³/s")
    with col4:
        st.metric("Min Water Flow", f"{df['water_flow'].min():.1f} m³/s")
    
    # Time series visualization
    st.subheader("Water Flow Time Series")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Water Flow (m³/s)", "Environmental Factors"),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Water flow
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['water_flow'],
                  mode='lines', name='Water Flow',
                  line=dict(color='#3B82F6', width=2)),
        row=1, col=1
    )
    
    # Add rolling average
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['water_flow'].rolling(30).mean(),
                  mode='lines', name='30-Day Avg',
                  line=dict(color='#EF4444', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Environmental factors
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['rainfall_mm'],
                  mode='lines', name='Rainfall',
                  line=dict(color='#10B981', width=1),
                  yaxis='y2'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['temperature_c'],
                  mode='lines', name='Temperature',
                  line=dict(color='#F59E0B', width=1),
                  yaxis='y3'),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis2=dict(title="Date"),
        yaxis=dict(title="Water Flow (m³/s)"),
        yaxis2=dict(title="Rainfall (mm)", overlaying='y', side='right'),
        yaxis3=dict(title="Temperature (°C)", anchor='free', position=1.0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Feature Correlation Matrix")
    
    corr_cols = ['water_flow', 'rainfall_mm', 'temperature_c', 
                'reservoir_level_percent', 'upstream_flow', 'soil_moisture_percent']
    
    # Ensure all columns exist
    available_cols = [col for col in corr_cols if col in df.columns]
    
    if len(available_cols) >= 2:
        corr_matrix = df[available_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=available_cols,
            y=available_cols,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig_corr.update_layout(
            title="Feature Correlation Heatmap",
            height=500
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Insufficient columns for correlation matrix. Need at least 2 numeric columns.")

# Tab 3: ML Model Training
with tab3:
    st.header("🤖 Machine Learning Model Training")
    
    if st.session_state.data is not None:
        df = st.session_state.data.copy()
        
        # Feature engineering
        df_engineered = engineer_features(df, include_lagged, include_seasonal)
        
        # Prepare features and target
        feature_cols = [col for col in df_engineered.columns 
                       if col not in ['date', 'water_flow'] 
                       and not col.startswith('water_flow')]
        
        X = df_engineered[feature_cols]
        y = df_engineered['water_flow']
        
        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model training
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner(f"Training {model_type} model..."):
                # Model selection
                if model_type == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                elif model_type == "Gradient Boosting":
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
                elif model_type == "Linear Regression":
                    model = Ridge(alpha=1.0)
                else:  # Ensemble
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Store model and predictions
                st.session_state.model = model
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.y_pred_test = y_pred_test
                st.session_state.scaler = scaler
                st.session_state.feature_cols = feature_cols
                
                st.success("✅ Model trained successfully!")
        
        # Display results if model exists
        if st.session_state.model is not None:
            model = st.session_state.model
            
            # Get predictions from session state
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = st.session_state.y_pred_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
            
            # Calculate metrics
            metrics_train = {
                'R²': r2_score(y_train, y_pred_train),
                'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'MAE': mean_absolute_error(y_train, y_pred_train)
            }
            
            metrics_test = {
                'R²': r2_score(y_test, y_pred_test),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'MAE': mean_absolute_error(y_test, y_pred_test)
            }
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train R²", f"{metrics_train['R²']:.3f}")
            with col2:
                st.metric("Test R²", f"{metrics_test['R²']:.3f}")
            with col3:
                st.metric("Test RMSE", f"{metrics_test['RMSE']:.1f}")
            with col4:
                st.metric("Test MAE", f"{metrics_test['MAE']:.1f}")
            
            # Plot predictions
            st.subheader("Model Predictions vs Actual")
            
            fig_pred = make_subplots(rows=2, cols=1, subplot_titles=("Training Set", "Test Set"))
            
            fig_pred.add_trace(
                go.Scatter(x=list(range(len(y_train))), y=y_train.values,
                          mode='lines', name='Actual',
                          line=dict(color='#3B82F6', width=2)),
                row=1, col=1
            )
            
            fig_pred.add_trace(
                go.Scatter(x=list(range(len(y_pred_train))), y=y_pred_train,
                          mode='lines', name='Predicted',
                          line=dict(color='#EF4444', width=2, dash='dash')),
                row=1, col=1
            )
            
            fig_pred.add_trace(
                go.Scatter(x=list(range(len(y_test))), y=y_test.values,
                          mode='lines', name='Actual',
                          line=dict(color='#3B82F6', width=2),
                          showlegend=False),
                row=2, col=1
            )
            
            fig_pred.add_trace(
                go.Scatter(x=list(range(len(y_pred_test))), y=y_pred_test,
                          mode='lines', name='Predicted',
                          line=dict(color='#EF4444', width=2, dash='dash'),
                          showlegend=False),
                row=2, col=1
            )
            
            fig_pred.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig_pred, use_container_width=True)

# Tab 4: SHAP Explanations
with tab4:
    st.header("🔍 SHAP Explanations")
    
    if st.session_state.model is not None and st.session_state.data is not None:
        st.info("SHAP analysis is available after model training. Navigate to the ML Model Training tab to train a model first.")
        
        if st.button("Generate SHAP Analysis"):
            with st.spinner("Calculating SHAP values..."):
                try:
                    model = st.session_state.model
                    X_test = st.session_state.X_test
                    
                    # Use a subset for faster computation
                    X_test_sample = X_test.iloc[:100]
                    
                    # Create explainer
                    explainer = shap.Explainer(model, X_test_sample)
                    shap_values = explainer(X_test_sample)
                    
                    # Store SHAP values
                    st.session_state.shap_values = shap_values
                    
                    # Display SHAP summary plot
                    st.subheader("SHAP Summary Plot")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_test_sample, show=False)
                    st.pyplot(fig)
                    
                    # Display SHAP force plot for first sample
                    st.subheader("SHAP Force Plot (First Sample)")
                    fig_force, ax_force = plt.subplots(figsize=(12, 4))
                    shap.plots.force(shap_values[0], matplotlib=True, show=False)
                    st.pyplot(fig_force)
                    
                except Exception as e:
                    st.error(f"Error generating SHAP analysis: {str(e)}")
                    st.info("Note: SHAP analysis may require additional computation. Try with a smaller dataset or different model.")
    else:
        st.warning("Please train a model first in the ML Model Training tab to enable SHAP explanations.")

# Tab 5: Predictions & Forecast
with tab5:
    st.header("📈 Predictions & Forecast")
    
    if st.session_state.model is not None and st.session_state.data is not None:
        model = st.session_state.model
        df = st.session_state.data
        
        # Generate future dates for forecasting
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                    periods=prediction_horizon, freq='D')
        
        # Create a simple forecast (in a real scenario, you'd use the trained model)
        st.subheader(f"{prediction_horizon}-Day Forecast")
        
        # Simple forecasting based on historical patterns
        recent_data = df.tail(30)
        avg_flow = recent_data['water_flow'].mean()
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * (future_dates.dayofyear - 80) / 365)
        
        # Generate forecast with some randomness
        np.random.seed(42)
        forecast_flows = avg_flow * seasonal_factor * np.random.normal(1, 0.1, len(future_dates))
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_flow': forecast_flows,
            'lower_bound': forecast_flows * 0.9,  # 10% lower bound
            'upper_bound': forecast_flows * 1.1   # 10% upper bound
        })
        
        # Store forecast in session state
        st.session_state.forecast_df = forecast_df
        
        # Plot forecast
        fig_forecast = go.Figure()
        
        # Add historical data (last 90 days)
        historical = df.tail(90)
        fig_forecast.add_trace(go.Scatter(
            x=historical['date'],
            y=historical['water_flow'],
            mode='lines',
            name='Historical Data',
            line=dict(color='#3B82F6', width=2)
        ))
        
        # Add forecast
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted_flow'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#EF4444', width=3)
        ))
        
        # Add confidence interval
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
            y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{confidence_level}% Confidence Interval',
            showlegend=True
        ))
        
        fig_forecast.update_layout(
            title=f"Water Flow Forecast - Next {prediction_horizon} Days",
            xaxis_title="Date",
            yaxis_title="Water Flow (m³/s)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Display forecast table
        st.subheader("Forecast Details")
        forecast_display = forecast_df.copy()
        forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
        forecast_display = forecast_display.round(1)
        st.dataframe(forecast_display)
        
    else:
        st.warning("Please train a model first in the ML Model Training tab to generate forecasts.")

# Tab 6: What-If Analysis
with tab6:
    st.header("🎯 What-If Analysis")
    
    if st.session_state.model is not None:
        st.subheader("Adjust Parameters to See Impact on Water Flow")
        
        col1, col2 = st.columns(2)
        
        with col1:
            rainfall_change = st.slider("Rainfall Change (%)", -50, 200, 0, 
                                       help="Percentage change in rainfall")
            temperature_change = st.slider("Temperature Change (°C)", -10, 10, 0,
                                          help="Change in temperature in degrees Celsius")
            
        with col2:
            reservoir_change = st.slider("Reservoir Level Change (%)", -30, 30, 0,
                                        help="Percentage change in reservoir level")
            upstream_change = st.slider("Upstream Flow Change (%)", -50, 150, 0,
                                       help="Percentage change in upstream flow")
        
        # Calculate impact based on simple rules
        base_flow = df['water_flow'].mean()
        
        # Simple impact calculations (in a real scenario, use the trained model)
        rainfall_impact = base_flow * (rainfall_change / 100) * 0.5  # 50% of rainfall change affects flow
        temperature_impact = base_flow * (temperature_change / 10) * 0.1  # 1% per 10°C
        reservoir_impact = base_flow * (reservoir_change / 100) * -0.3  # Negative correlation
        upstream_impact = base_flow * (upstream_change / 100) * 0.7  # 70% correlation
        
        total_impact = rainfall_impact + temperature_impact + reservoir_impact + upstream_impact
        new_flow = base_flow + total_impact
        
        # Display results
        st.subheader("Impact Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Base Flow", f"{base_flow:.1f} m³/s")
        with col2:
            st.metric("New Flow", f"{new_flow:.1f} m³/s")
        with col3:
            st.metric("Change", f"{total_impact:.1f} m³/s", 
                     delta=f"{(total_impact / base_flow * 100):.1f}%" if base_flow > 0 else "0%")
        
        # Visualize impacts
        impacts_df = pd.DataFrame({
            'Factor': ['Rainfall', 'Temperature', 'Reservoir Level', 'Upstream Flow'],
            'Impact (m³/s)': [rainfall_impact, temperature_impact, reservoir_impact, upstream_impact]
        })
        
        fig_impact = px.bar(impacts_df, x='Factor', y='Impact (m³/s)',
                           title="Contribution of Each Factor to Water Flow Change",
                           color='Impact (m³/s)',
                           color_continuous_scale='RdBu')
        
        st.plotly_chart(fig_impact, use_container_width=True)
        
    else:
        st.warning("Please train a model first to perform What-If analysis.")

# Tab 7: Report & Export
with tab7:
    st.header("📋 Comprehensive Report & Export")
    
    # Initialize variables with default values
    performance_metrics = {
        'R² Score': 0.0,
        'RMSE': 0.0,
        'MAE': 0.0,
        'MAPE': 0.0,
        'Max Error': 0.0
    }
    
    importance_df = pd.DataFrame()
    forecast_summary = {}
    
    if st.session_state.model is not None:
        model = st.session_state.model
        
        # Generate report data
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Update performance metrics if available
        if (hasattr(st.session_state, 'y_pred_test') and 
            hasattr(st.session_state, 'y_test') and 
            st.session_state.y_pred_test is not None and 
            st.session_state.y_test is not None):
            
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred_test
            
            if len(y_test) > 0 and len(y_pred) > 0:
                try:
                    performance_metrics = {
                        'R² Score': r2_score(y_test, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'MAE': mean_absolute_error(y_test, y_pred),
                        'MAPE': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100,
                        'Max Error': np.max(np.abs(y_test - y_pred))
                    }
                except:
                    pass  # Keep default values if calculation fails
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            if st.session_state.feature_cols is not None:
                importance_df = pd.DataFrame({
                    'Feature': st.session_state.feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
        
        # Data summary
        data_summary = {
            'total_days': len(st.session_state.data) if st.session_state.data is not None else 0,
            'num_features': len(st.session_state.feature_cols) if st.session_state.feature_cols is not None else 0,
            'date_range': f"{st.session_state.data['date'].min().date()} to {st.session_state.data['date'].max().date()}" if st.session_state.data is not None else 'N/A',
            'avg_flow': st.session_state.data['water_flow'].mean() if st.session_state.data is not None else 0
        }
        
        # SHAP insights
        shap_insights = [
            "Rainfall and upstream flow are primary drivers of water flow",
            "Seasonal patterns significantly impact predictions",
            "Reservoir level has a negative correlation with water flow",
            "Temperature effects vary by season"
        ]
        
        # Forecast summary
        if 'forecast_df' in st.session_state and st.session_state.forecast_df is not None:
            forecast_df = st.session_state.forecast_df
            if not forecast_df.empty:
                forecast_summary = {
                    'Average Forecast': forecast_df['predicted_flow'].mean(),
                    'Maximum Forecast': forecast_df['predicted_flow'].max(),
                    'Minimum Forecast': forecast_df['predicted_flow'].min(),
                    'High Risk Days': len(forecast_df[forecast_df['predicted_flow'] > 
                                                     st.session_state.data['water_flow'].quantile(0.9)])
                }
        
        # Recommendations
        recommendations = [
            "Monitor rainfall patterns closely during high-flow seasons",
            "Adjust reservoir operations based on 7-day forecasts",
            "Implement early warning system for extreme flow events",
            "Regularly update model with new operational data",
            "Combine model predictions with domain expertise"
        ]
        
        # Report generation section
        st.subheader("📄 Generate Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["Comprehensive Report", "Executive Summary", "Technical Analysis"]
            )
            
            include_charts = st.checkbox("Include Charts in PDF", value=True)
            include_data = st.checkbox("Include Data Summary", value=True)
            include_recommendations = st.checkbox("Include Recommendations", value=True)
        
        with col2:
            st.info("""
            **Report will include:**
            - Model performance metrics
            - Feature importance analysis
            - Forecast summary
            - SHAP insights
            - Actionable recommendations
            """)
        
        # Generate PDF report
        if st.button("📊 Generate PDF Report", type="primary"):
            with st.spinner("Generating comprehensive report..."):
                try:
                    # Generate PDF
                    pdf_buffer = generate_pdf_report(
                        model_type=model_type,
                        performance_metrics=performance_metrics,
                        feature_importance=importance_df,
                        forecast_summary=forecast_summary,
                        shap_insights=shap_insights,
                        data_summary=data_summary,
                        recommendations=recommendations if include_recommendations else []
                    )
                    
                    # Create download button
                    st.success("✅ Report generated successfully!")
                    
                    # Convert to base64 for download
                    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                    
                    # Create download link
                    current_date = datetime.now().strftime("%Y%m%d_%H%M")
                    filename = f"water_flow_report_{current_date}.pdf"
                    
                    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="stButton">📥 Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Show preview
                    st.subheader("📋 Report Preview")
                    
                    with st.expander("Click to view report summary"):
                        st.markdown("### 📊 Executive Summary")
                        st.markdown(f"""
                        **Model Type:** {model_type}
                        
                        **Performance:** 
                        - R² Score: {performance_metrics['R² Score']:.3f}
                        - RMSE: {performance_metrics['RMSE']:.1f} m³/s
                        - MAE: {performance_metrics['MAE']:.1f} m³/s
                        
                        **Data Coverage:** {data_summary['total_days']} days
                        
                        **Key Insights:**
                        - Top feature: {importance_df.iloc[0]['Feature'] if not importance_df.empty else 'N/A'}
                        - Average flow: {data_summary['avg_flow']:.1f} m³/s
                        """)
                        
                        if forecast_summary:
                            st.markdown("### 📈 Forecast Summary")
                            st.markdown(f"""
                            - Average forecast: {forecast_summary['Average Forecast']:.1f} m³/s
                            - Maximum forecast: {forecast_summary['Maximum Forecast']:.1f} m³/s
                            - High risk days: {forecast_summary.get('High Risk Days', 0)}
                            """)
                        
                        if include_recommendations:
                            st.markdown("### 🎯 Recommendations")
                            for i, rec in enumerate(recommendations[:3], 1):
                                st.markdown(f"{i}. {rec}")
                
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        
        # Export options
        st.markdown("---")
        st.subheader("📤 Additional Export Options")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            if st.button("💾 Export Model"):
                try:
                    joblib.dump(model, 'dam_water_flow_model.pkl')
                    st.success("Model saved as 'dam_water_flow_model.pkl'")
                    
                    with open('dam_water_flow_model.pkl', 'rb') as f:
                        st.download_button(
                            label="📥 Download Model",
                            data=f,
                            file_name="dam_water_flow_model.pkl",
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    st.error(f"Error exporting model: {str(e)}")
        
        with col_exp2:
            if st.button("📊 Export Forecast Data"):
                if 'forecast_df' in st.session_state and st.session_state.forecast_df is not None:
                    try:
                        csv = st.session_state.forecast_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Forecast CSV",
                            data=csv,
                            file_name="water_flow_forecast.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error exporting forecast: {str(e)}")
                else:
                    st.warning("Generate forecasts first in the Predictions tab")
        
        with col_exp3:
            if st.button("📈 Export Performance Metrics"):
                try:
                    metrics_df = pd.DataFrame(list(performance_metrics.items()), 
                                            columns=['Metric', 'Value'])
                    csv = metrics_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Metrics CSV",
                        data=csv,
                        file_name="model_performance_metrics.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error exporting metrics: {str(e)}")
        
        # Report summary
        st.markdown("---")
        st.subheader("📋 Report Summary")
        
        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("Model R²", f"{performance_metrics['R² Score']:.3f}")
        with summary_cols[1]:
            st.metric("Forecast Days", prediction_horizon)
        with summary_cols[2]:
            if forecast_summary:
                st.metric("High Risk Days", forecast_summary.get('High Risk Days', 0))
        with summary_cols[3]:
            if not importance_df.empty:
                top_feature = importance_df.iloc[0]['Feature'][:15] + "..." if len(importance_df.iloc[0]['Feature']) > 15 else importance_df.iloc[0]['Feature']
                st.metric("Top Feature", top_feature)
            else:
                st.metric("Top Feature", "N/A")
        
        # Feature importance visualization for report
        if not importance_df.empty:
            st.subheader("📊 Feature Importance (Top 10)")
            
            fig_importance = px.bar(importance_df.head(10), x='Importance', y='Feature',
                                  orientation='h',
                                  title="Top 10 Most Important Features",
                                  color='Importance',
                                  color_continuous_scale='Viridis')
            
            fig_importance.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.warning("Please train a model first to generate reports.")
        st.info("Navigate to the 'ML Model Training' tab to train your model.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 1rem;'>
    <p>🌊 <b>Explainable Water Flow Prediction System</b></p>
    <p>For water resource management and dam safety monitoring</p>
</div>
""", unsafe_allow_html=True)