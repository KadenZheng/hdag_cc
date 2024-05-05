import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib

prop_wdi_final_data = pd.read_csv('prop_wdi_final_data.csv')
print("data received")

prop_wdi_final_data = prop_wdi_final_data[prop_wdi_final_data['Year'] == 2023]
prop_wdi_final_data.to_csv('prop_wdi_final_data_2023.csv', index=False)
print("data saved")
