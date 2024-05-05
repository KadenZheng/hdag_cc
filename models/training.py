import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load your data here
prop_wdi_final_data = pd.read_csv('../data/prop_wdi_final_data.csv')
print("data received")

# CODE BEFORE NARROWING DOWN FEATURES
# Remove non-numeric columns for simplicity
features = prop_wdi_final_data.select_dtypes(include=["number"])
features = features[['Year', 'DME', 'Industry Value Final', 'DfR',
       'Population in urban agglomerations of more than 1 million (% of total population)',
       'Labor force, total_valmultiplied',
       'Population in the largest city (% of urban population)_valmultiplied',
       'Unemployment, youth female (% of female labor force ages 15-24) (modeled ILO estimate)_valmultiplied',
       'Population in the largest city (% of urban population)_volume_multiplied',
       'Population in urban agglomerations of more than 1 million_volume_multiplied',
       'Population in urban agglomerations of more than 1 million (% of total population)_volume_multiplied']]

# features = features.drop(['Gross Profit', 'Brand Contribution', 'Revenue', 'COGS', 'Operating Expenses'], axis=1)
targets = prop_wdi_final_data[['Revenue', 'COGS', 'Unit Cases']]
print("features split")

# Split the data into training and testing sets
X_train = features[features['Year'] != 2023]
X_test = features[features['Year'] == 2023]
y_train = targets[features['Year'] != 2023]
y_test = targets[features['Year'] == 2023]
print("data split")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')
print("scaled")

# Neural network for Revenue
print("building model revenue")
model_revenue = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1)
])
model_revenue.compile(optimizer=Adam(learning_rate=0.003), loss='mse')

# Neural network for COGS
print("building model cogs")
model_cogs = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1)
])
model_cogs.compile(optimizer=Adam(learning_rate=0.003), loss='mse')

# Neural network for Unit Cases
print("building model unit cases")
model_uc = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
model_uc.compile(optimizer=Adam(learning_rate=0.1), loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Add your training code here
print("training revenue")
model_revenue.fit(X_train_scaled, y_train['Revenue'], validation_split=0.2, epochs=100, callbacks=[early_stopping], verbose=0)
print("training cogs")
model_cogs.fit(X_train_scaled, y_train['COGS'], validation_split=0.2, epochs=100, callbacks=[early_stopping], verbose=0)
print("training unit cases")
model_uc.fit(X_train_scaled, y_train['Unit Cases'], validation_split=0.2, epochs=100, callbacks=[early_stopping], verbose=0)

# Save models
print("saving models")
model_revenue.save('model_revenue.h5')
model_cogs.save('model_cogs.h5')
model_uc.save('model_uc.h5')