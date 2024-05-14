import streamlit as st
import os
import subprocess
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.training import RevenueModel, CogsModel, UnitCasesModel

st.title('Coca-Cola ASP OU Black Box')

tab1, tab2, tab3 = st.tabs(["How to Use", "Model Dashboard", "Case Documentation"])

with tab1:
    st.header('Welcome 👋!')
    st.markdown('''
        <style>
            .important {
                font-weight: bold;
                color: #e31e2e;
            }
        </style>
        <div>
            <p>This tool is a simple web-based interface to run models for predicting <span class="important">Revenue</span>, <span class="important">COGS</span>, and <span class="important">Unit Cases</span> for Coca-Cola's ASP OU.</p>
            <p>It allows you to <span class="important">upload a CSV file</span> containing the input feature data for a specific year and then predict the revenue, COGS, and unit cases for that year and location.</p>
            <p>You can also <span class="important">download the completed data</span> with model predictions integrated as a CSV file.</p>
            <p>Additionally, <span class="important">statistical metrics</span> evaluating each model's predictions are provided.</p>
        </div>
        ''', unsafe_allow_html=True)
    st.subheader('Enjoy🎉!')
    st.text('- Harvard Undergraduate Data Analytics Group')

with tab2: 
    with st.spinner('Initializing...'):
        # Function to install packages
        def install(package):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to install {package}. Error: {str(e)}")

        # List of required packages
        required_packages = ['pandas', 'numpy', 'streamlit', 'torch', 'torchvision', 'joblib']
        for package in required_packages:
            install(package)

        # Load models
        model_revenue = RevenueModel()
        model_cogs = CogsModel()
        model_uc = UnitCasesModel()

        model_revenue.load_state_dict(torch.load('models/model_revenue.pth'))
        model_cogs.load_state_dict(torch.load('models/model_cogs.pth'))
        model_uc.load_state_dict(torch.load('models/model_uc.pth'))
        scaler = joblib.load('models/scaler.pkl')

        def load_data(file):
            data = pd.read_csv(file)
            return data

        def process_data(data):
            return data

        def calculate_statistics(data):
            features = data.select_dtypes(include=["number"])
            features = features[['Year', 'DME', 'Industry Value Final', 'DfR',
            'Population in urban agglomerations of more than 1 million (% of total population)',
            'Labor force, total_valmultiplied',
            'Population in the largest city (% of urban population)_valmultiplied',
            'Unemployment, youth female (% of female labor force ages 15-24) (modeled ILO estimate)_valmultiplied',
            'Population in the largest city (% of urban population)_volume_multiplied',
            'Population in urban agglomerations of more than 1 million_volume_multiplied',
            'Population in urban agglomerations of more than 1 million (% of total population)_volume_multiplied']]
            
            if not set(features.columns).issubset(data.columns):
                st.error('Uploaded data does not contain the required features.')
                return pd.DataFrame()

            X = scaler.transform(features)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                model_revenue.eval()
                model_cogs.eval()
                model_uc.eval()
                y_pred_revenue = model_revenue(X_tensor).numpy().flatten()
                y_pred_cogs = model_cogs(X_tensor).numpy().flatten()
                y_pred_uc = model_uc(X_tensor).numpy().flatten()

            mse_revenue = mean_squared_error(data['Revenue'], y_pred_revenue)
            mae_revenue = mean_absolute_error(data['Revenue'], y_pred_revenue)
            r2_revenue = r2_score(data['Revenue'], y_pred_revenue)
            test_loss_revenue = mse_revenue  # Simplified test loss as MSE for example

            mse_cogs = mean_squared_error(data['COGS'], y_pred_cogs)
            mae_cogs = mean_absolute_error(data['COGS'], y_pred_cogs)
            r2_cogs = r2_score(data['COGS'], y_pred_cogs)
            test_loss_cogs = mse_cogs  # Simplified test loss as MSE for example

            mse_uc = mean_squared_error(data['Unit Cases'], y_pred_uc)
            mae_uc = mean_absolute_error(data['Unit Cases'], y_pred_uc)
            r2_uc = r2_score(data['Unit Cases'], y_pred_uc)
            test_loss_uc = mse_uc  # Simplified test loss as MSE for example

            # Create results DataFrame
            results_df = pd.DataFrame({
                'Actual Revenue': data['Revenue'],
                'Predicted Revenue': y_pred_revenue,
                'Revenue Residual': data['Revenue'] - y_pred_revenue,
                'Actual COGS': data['COGS'],
                'Predicted COGS': y_pred_cogs,
                'COGS Residual': data['COGS'] - y_pred_cogs,
                'Actual Unit Cases': data['Unit Cases'],
                'Predicted Unit Cases': y_pred_uc,
                'Unit Cases Residual': data['Unit Cases'] - y_pred_uc,
            })

            error_metrics_df = pd.DataFrame({
                'Test Loss Revenue': [test_loss_revenue],
                'MSE Revenue': [mse_revenue],
                'MAE Revenue': [mae_revenue],
                'R² Revenue': [r2_revenue],
                'Test Loss COGS': [test_loss_cogs],
                'MSE COGS': [mse_cogs],
                'MAE COGS': [mae_cogs],
                'R² COGS': [r2_cogs],
                'Test Loss Unit Cases': [test_loss_uc],
                'MSE Unit Cases': [mse_uc],
                'MAE Unit Cases': [mae_uc],
                'R² Unit Cases': [r2_uc]
            })

            return results_df, error_metrics_df

        uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            if 'Revenue' in data.columns and 'COGS' in data.columns and 'Unit Cases' in data.columns:
                st.write('Data Preview:')
                editable_data = st.data_editor(data)
                
                if st.button('Process Data & Predict'):
                    result = process_data(editable_data)
                    results_df, error_metrics_df = calculate_statistics(data)
                    
                    with st.spinner("Generating predictions..."):
                        st.write('Model Predictions:')
                        st.write(results_df)

                        with st.sidebar:
                            st.header("Revenue Model Statistics")
                            st.metric("Test Loss", f"{error_metrics_df['Test Loss Revenue'].iloc[0]:.3f}")
                            st.metric("MSE", f"{error_metrics_df['MSE Revenue'].iloc[0]:.3f}")
                            st.metric("MAE", f"{error_metrics_df['MAE Revenue'].iloc[0]:.3f}")
                            st.metric("R²", f"{error_metrics_df['R² Revenue'].iloc[0]:.3f}")

                            st.header("COGS Model Statistics")
                            st.metric("Test Loss", f"{error_metrics_df['Test Loss COGS'].iloc[0]:.3f}")
                            st.metric("MSE", f"{error_metrics_df['MSE COGS'].iloc[0]:.3f}")
                            st.metric("MAE", f"{error_metrics_df['MAE COGS'].iloc[0]:.3f}")
                            st.metric("R²", f"{error_metrics_df['R² COGS'].iloc[0]:.3f}")

                            st.header("Unit Cases Model Statistics")
                            st.metric("Test Loss", f"{error_metrics_df['Test Loss Unit Cases'].iloc[0]:.3f}")
                            st.metric("MSE", f"{error_metrics_df['MSE Unit Cases'].iloc[0]:.3f}")
                            st.metric("MAE", f"{error_metrics_df['MAE Unit Cases'].iloc[0]:.3f}")
                            st.metric("R²", f"{error_metrics_df['R² Unit Cases'].iloc[0]:.3f}")

            else:
                st.error('Uploaded data does not contain the required columns: Revenue, COGS, Unit Cases')

with tab3:
    st.page_link("http://www.google.com", label="Documentation", icon="🌎")