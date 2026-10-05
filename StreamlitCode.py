import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet

# =====================================================================
# 1. PAGE SETUP & USER INTERFACE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="eThekwini Electoral Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 South African Municipal Elections: Predictive Dashboard")
st.markdown("### Case Study: eThekwini Metropolitan Municipality (2011 - 2021)")

# Sidebar Uploader (User Friendly)
st.sidebar.header("📁 Step 1: Data Collection")
uploaded_files = st.sidebar.file_uploader(
    "Upload your 3 eThekwini CSV files here:", 
    type="csv", 
    accept_multiple_files=True
)

# =====================================================================
# 2. DATA PROCESSING PIPELINE
# =====================================================================
if len(uploaded_files) == 3:
    st.sidebar.success("All 3 files uploaded successfully!")
    
    # Load files safely
    dfs = []
    for file in uploaded_files:
        df = pd.read_csv(file, encoding='latin1')
        # Simple rule to dynamically assign the correct year based on row counts/contents
        if len(df) < 20000:
            df['election_year'] = 2011
        elif "2016" in str(df.iloc[0]):
            df['election_year'] = 2016
        else:
            df['election_year'] = 2021
        dfs.append(df)
        
    # Stack the files (Data Ingestion)
    ethekwini_ts = pd.concat(dfs, ignore_index=True)
    ethekwini_ts.columns = ethekwini_ts.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Clean blank row outlines
    ethekwini_ts = ethekwini_ts.dropna(subset=['ward']).drop_duplicates().reset_index(drop=True)
    
    # --- TABS FOR USER-FRIENDLY NAVIGATION ---
    tab1, tab2, tab3 = st.tabs(["📈 Exploratory Data Analysis", "🌲 Machine Learning (Gradient Boosting)", "🔮 Time-Series Forecasting (Prophet)"])
    
    with tab1:
        st.header("Exploratory Data Analysis")
        
        # Metric Cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data Points Encompassed", f"{len(ethekwini_ts):,}")
        col2.metric("Unique Wards Monitored", f"{ethekwini_ts['ward'].nunique()}")
        col3.metric("Contesting Parties Logged", f"{ethekwini_ts['partyname'].nunique()}")
        
        # Interactive Bubble Chart
        st.subheader("Turnout vs Population Scale (Bubble Chart)")
        fig_bubble = px.scatter(
            ethekwini_ts,
            x="registeredvoters",
            y="totalvalidvotes",
            size="spoiltvotes",
            color="election_year",
            hover_name="ward",
            title="eThekwini Voting Dynamics over Time",
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    # =====================================================================
    # 3. FEATURE ENGINEERING & MACHINE LEARNING RESHAPING
    # =====================================================================
    pr_ballots = ethekwini_ts[ethekwini_ts['ballottype'] == 'PR'].copy()
    
    pivot_matrix = pr_ballots.pivot_table(
        index=['election_year', 'ward'],
        columns='partyname',
        values='totalvalidvotes',
        aggfunc='sum'
    ).fillna(0).reset_index()
    
    pivot_matrix['total_votes_cast'] = pivot_matrix.drop(columns=['election_year', 'ward'], errors='ignore').sum(axis=1)
    pivot_matrix['target_party_share'] = (pivot_matrix['AFRICAN NATIONAL CONGRESS'] / pivot_matrix['total_votes_cast']) * 100
    pivot_matrix['target_party_share'] = pivot_matrix['target_party_share'].replace([np.inf, -np.inf], np.nan).fillna(0)

    with tab2:
        st.header("Machine Learning Predictive Solution")
        st.write("Training model to evaluate party share on historical datasets (2011, 2016) against unseen future points (2021)...")
        
        # Features & Targets Setup
        X_features = pivot_matrix.drop(columns=['election_year', 'ward', 'AFRICAN NATIONAL CONGRESS', 'total_votes_cast'], errors='ignore')
        
        train_mask = pivot_matrix['election_year'] < 2021
        test_mask = pivot_matrix['election_year'] == 2021
        
        X_train, y_train = X_features[train_mask], pivot_matrix.loc[train_mask, 'target_party_share']
        X_test, y_test = X_features[test_mask], pivot_matrix.loc[test_mask, 'target_party_share']
        
        # Train Model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict & Evaluate
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        # Performance Evaluation Cards
        ml_col1, ml_col2 = st.columns(2)
        ml_col1.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
        ml_col2.metric("R² Prediction Accuracy Score", f"{r2:.2f}")
        
        # Feature Importance Chart
        st.subheader("Top 10 Most Important Features (Parties) Driving the Forecast")
        feat_imp = pd.DataFrame({'Party': X_train.columns, 'Importance': model.feature_importances_})
        top_feats = feat_imp.sort_values(by='Importance', ascending=False).head(10)
        fig_importance = px.bar(top_feats, x='Importance', y='Party', orientation='h', color='Importance', color_continuous_scale='Magma')
        st.plotly_chart(fig_importance, use_container_width=True)

    with tab3:
        st.header("Time-Series Forecasting Framework")
        st.write("Collapsing geographical records into a true chronological sequence to feed the Prophet model without compiler issues.")
        
        # Consolidate timeline globally across the metro
        yearly_totals = pr_ballots.groupby(['election_year'])['totalvalidvotes'].sum().reset_index()
        anc_totals = pr_ballots[pr_ballots['partyname'] == 'AFRICAN NATIONAL CONGRESS'].groupby(['election_year'])['totalvalidvotes'].sum().reset_index().rename(columns={'totalvalidvotes': 'anc_votes'})
        
        merged_timeline = pd.merge(yearly_totals, anc_totals, on='election_year')
        merged_timeline['target_party_share'] = (merged_timeline['anc_votes'] / merged_timeline['totalvalidvotes']) * 100
        
        # Set Prophet Layout Structure
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = pd.to_datetime(merged_timeline['election_year'].astype(str) + '-11-01')
        prophet_df['y'] = merged_timeline['target_party_share']
        
        # Fit Prophet
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        m.fit(prophet_df)
        
        # Forecast 5 years into the future (2026 Target Frame)
        future = m.make_future_dataframe(periods=1, freq='5Y')
        forecast = m.predict(future)
        
        # Interactive Line Graph of the Timeline Forecast
        st.subheader("Historical Trajectory Share Curve & 2026 Estimate Output")
        fig_forecast = px.line(forecast, x='ds', y='yhat', markers=True, title="ANC Vote Share Trend over Time")
        fig_forecast.update_layout(xaxis_title="Timeline Interval", yaxis_title="Vote Share %")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        projected_2026 = forecast.iloc[-1]['yhat']
        st.info(f"🔮 **The Prophet Model Estimates overall Party Support Share in 2026 to point at roughly: {projected_2026:.2f}%**")

else:
    st.info("ℹ️ Please upload exactly 3 CSV datasets via the left sidebar to generate the web dashboard analysis pipeline.")
