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
import models.training as training

PASSWORD = st.secrets["password"]

st.title('Coca-Cola ASP OU Black Box')

# Password input and validation
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    def password_entered():
        if st.session_state["password"] == PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False
            st.toast("Incorrect password. Please try again!", icon="❌")

    if not st.session_state["password_correct"]:
        with st.form(key='password_form'):
            st.text_input("Password", type="password", key="password")
            submit_button = st.form_submit_button("Enter")
            if submit_button:
                password_entered()
        return False
    else:
        return True
    
if check_password():

    tab1, tab2, tab3 = st.tabs(["How to Use", "Model Dashboard", "Case Documentation"])

    data_files = [f for f in os.listdir('data') if f.endswith('.csv')]

    # Basic instructions for application usage.
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

    # Model Dashboard
    with tab2: 
        tabs1, tabs2 = st.tabs(["Model Testing", "Generate Forecasts"])

        # Required features for model prediction
        req_features = ['Year', 'DME', 'Industry Value Final', 'DfR',
                'Population in urban agglomerations of more than 1 million (% of total population)',
                'Labor force, total_valmultiplied',
                'Population in the largest city (% of urban population)_valmultiplied',
                'Unemployment, youth female (% of female labor force ages 15-24) (modeled ILO estimate)_valmultiplied',
                'Population in the largest city (% of urban population)_volume_multiplied',
                'Population in urban agglomerations of more than 1 million_volume_multiplied',
                'Population in urban agglomerations of more than 1 million (% of total population)_volume_multiplied']

        with st.spinner('Initalizing...'):
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
        st.sidebar.success('Packages installed.')

        def load_data(file):
            data = pd.read_csv(file)
            return data

        def process_data(data):
            return data
        

        # Run predictive tasks -> return results and error metrics
        def calculate_statistics(data, return_error_metrics=True):
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

            # Create results dataframe
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

            integrated_df = data.copy()
            integrated_df['Revenue'] = y_pred_revenue
            integrated_df['COGS'] = y_pred_cogs
            integrated_df['Unit Cases'] = y_pred_uc

            if return_error_metrics:
                # Create error statistics dataframe
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
                return results_df, integrated_df, error_metrics_df
            else:
                return results_df, integrated_df

        with st.spinner('Initializing models...'):
            model_revenue = RevenueModel()
            model_cogs = CogsModel()
            model_uc = UnitCasesModel()

            model_revenue.load_state_dict(torch.load('models/model_revenue.pth'))
            model_cogs.load_state_dict(torch.load('models/model_cogs.pth'))
            model_uc.load_state_dict(torch.load('models/model_uc.pth'))
            scaler = joblib.load('models/scaler.pkl')
        st.sidebar.success('Models loaded successfully.')

        with st.spinner('Preparing prediction pipeline...'):
            st.sidebar.subheader('Retrain Models')
            new_training_file = st.sidebar.file_uploader("Choose a CSV file to retrain. Otherwise, the default configuration will be used.", type='csv')
            if new_training_file is not None:
                training.retrain_models(load_data(new_training_file))
                st.sidebar.success("Models retrained successfully!")
            else:
                default_data = load_data('data/prop_wdi_final_data.csv')
                training.retrain_models(default_data)

        # Model testing implementation (not for generating forecasts)
        with tabs1:
            with st.spinner('Loading...'):
                st.subheader('Testing')
                selected_file = st.selectbox("Choose a CSV file to perform a test with the current configuration.", data_files)

                if selected_file:
                    data = load_data(os.path.join('data', selected_file))

                    if 'Revenue' in data.columns and 'COGS' in data.columns and 'Unit Cases' in data.columns and set(req_features).issubset(data.columns):
                        st.write('Data Preview:')
                        editable_data = st.data_editor(data, key="data_editor_testing")

                        if st.button('Process Data & Predict', key="predict_button"):
                            result = process_data(editable_data)
                            results_df, integrated_df, error_metrics_df = calculate_statistics(data, return_error_metrics=True)
                            
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
                                st.sidebar.success('Prediction statistics generated.')

                    else:
                        st.error('Uploaded data does not contain the required target variables (Revenue, COGS, Unit Cases) or the required input features.')

        # Generate forecasts implementation (not for testing)
        with tabs2:
            st.subheader('Forecasts')
            
            # Option to select a file from the existing data directory
            selected_file = st.selectbox("Choose a CSV file (with Input Features)", data_files)
            
            # Option to upload an external file
            uploaded_file = st.file_uploader("Or upload a CSV file", type='csv')
            
            if selected_file or uploaded_file:
                if uploaded_file:
                    data = load_data(uploaded_file)
                else:
                    data = load_data(os.path.join('data', selected_file))

                if set(req_features).issubset(data.columns):
                    st.write('Input Data Preview:')
                    editable_inputs = st.data_editor(data, key="data_editor_forecasts")

                    if st.button('Process Data & Predict', key="predict_button_forecasts"):
                        result = process_data(editable_inputs)
                        results_df, integrated_df = calculate_statistics(data, return_error_metrics=False)
                        
                        with st.spinner("Generating predictions..."):
                            st.write('Model Predictions:')
                            st.write(results_df)

                            csv_results = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Model Predictions as CSV",
                                data=csv_results,
                                file_name='model_predictions.csv',
                                mime='text/csv',
                            )

                            st.write('Integrated Data with Forecasts:')
                            st.write(integrated_df)

                            csv_integrated = integrated_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Integrated Data as CSV",
                                data=csv_integrated,
                                file_name='integrated_data.csv',
                                mime='text/csv',
                            )

                            st.success('Forecasts generated.')

                else:
                    st.error('Uploaded data does not contain one or more of the required input features.')

    with tab3:
        st.page_link("https://docs.google.com/document/d/1QV_2c4kduPLH3IRlfrMHrrNBEVo8iLF9Eshe_rwvKM4/edit?usp=sharingm", label="Documentation and Case Report", icon="🌎")