import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Prophet: House Price Analytics", page_icon="🏠", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# DATA ENGINE (With Caching)
# ===============================
@st.cache_data
def get_cleaned_data():
    if not os.path.exists("house_prices.csv"):
        return None
    
    df = pd.read_csv("house_prices.csv")
    
    def convert_price(price):
        if pd.isna(price) or not isinstance(price, str): return np.nan
        price = price.replace(",", "").strip()
        try:
            if "Cr" in price:
                return float(price.replace("Cr", "").strip()) * 10_000_000
            elif "Lac" in price:
                return float(price.replace("Lac", "").strip()) * 100_000
            return float(re.sub(r'[^\d.]', '', price))
        except:
            return np.nan

    # Cleaning pipeline
    df["Price"] = df["Amount(in rupees)"].apply(convert_price)
    df["Carpet Area"] = df["Carpet Area"].str.extract(r"(\d+)").astype(float)
    df["Bathroom"] = pd.to_numeric(df["Bathroom"], errors="coerce").fillna(0)
    df["Balcony"] = pd.to_numeric(df["Balcony"], errors="coerce").fillna(0)
    
    return df.dropna(subset=["Price", "Carpet Area"])

# ===============================
# ML ENGINE (Cache the Model!)
# ===============================
@st.cache_resource
def train_model(data):
    features = ["Carpet Area", "Bathroom", "Balcony"]
    X = data[features]
    y = data["Price"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate stats for the ML page
    preds = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds)
    }
    return model, metrics

# --- LOAD DATA ---
df = get_cleaned_data()

if df is None:
    st.error("⚠️ 'house_prices.csv' not found. Please ensure the file is in the directory.")
    st.stop()

model, model_metrics = train_model(df)

# ===============================
# SIDEBAR NAVIGATION
# ===============================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/609/609803.png", width=100)
    st.title("Prophet AI")
    page = st.radio("Navigation", ["Overview", "Analytics", "Prediction Engine"])
    st.info("This tool uses a Random Forest Regressor to estimate property values based on historical data.")

# ===============================
# PAGE: OVERVIEW
# ===============================
if page == "Overview":
    st.title("🏠 Real Estate Dashboard")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Listings", f"{len(df):,}")
    col2.metric("Avg. Price", f"₹{int(df['Price'].mean()):,}")
    col3.metric("Avg. Area", f"{int(df['Carpet Area'].mean())} sqft")

    st.subheader("Recent Listings")
    st.dataframe(df.head(10), use_container_width=True)

# ===============================
# PAGE: ANALYTICS
# ===============================
elif page == "Analytics":
    st.title("📊 Market Insights")
    
    tab1, tab2 = st.tabs(["Distributions", "Correlations"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig_price = px.box(df, y="Price", title="Price Range Outliers", color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig_price, use_container_width=True)
        with c2:
            fig_area = px.scatter(df, x="Carpet Area", y="Price", trendline="ols", title="Price vs Area (with Trendline)")
            st.plotly_chart(fig_area, use_container_width=True)

    with tab2:
        st.subheader("Feature Correlation")
        # 
        corr = df[["Price", "Carpet Area", "Bathroom", "Balcony"]].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)

# ===============================
# PAGE: PREDICTION ENGINE
# ===============================
elif page == "Prediction Engine":
    st.title("💰 Smart Price Estimator")
    
    col_input, col_stats = st.columns([1, 1])
    
    with col_input:
        st.subheader("Property Details")
        area = st.number_input("Carpet Area (sqft)", 100, 10000, 1200)
        bath = st.slider("Bathrooms", 1, 10, 2)
        balc = st.slider("Balconies", 0, 10, 1)
        
        if st.button("Calculate Valuation", type="primary"):
            input_df = pd.DataFrame([[area, bath, balc]], columns=["Carpet Area", "Bathroom", "Balcony"])
            prediction = model.predict(input_df)[0]
            
            st.success(f"### Estimated Value: ₹{int(prediction):,}")
            st.balloons()

    with col_stats:
        st.subheader("Model Reliability")
        st.write(f"**Accuracy (R²):** {model_metrics['r2']:.2%}")
        st.write(f"**Avg. Error:** ₹{int(model_metrics['mae']):,}")
        st.caption("Note: Accuracy depends on data quality and location nuances.")