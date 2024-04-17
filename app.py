import streamlit as st
import pandas as pd
import numpy as np

def load_data(file):
    data = pd.read_csv(file)
    return data

def process_data(data):
    # Dummy processing: just return the first 5 rows
    return data.head(5)

def calculate_statistics(data):
    # Dummy statistics
    mae = np.random.uniform(0, 2)
    r_squared = np.random.uniform(0, 1)
    return mae, r_squared

st.title('Simple Model Prediction App')

uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write('Data Preview:')
    st.write(data)
    
    if st.button('Process Data'):
        result = process_data(data)
        mae, r_squared = calculate_statistics(data)
        
        st.write('Processed Data Preview:')
        st.write(result)
        
        st.write('Model Statistics:')
        st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
        st.write(f'R-squared: {r_squared:.2f}')
