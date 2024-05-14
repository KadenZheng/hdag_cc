import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Load your data here
prop_wdi_final_data = pd.read_csv('./data/prop_wdi_final_data.csv')
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

# Convert numpy arrays to torch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

# Define models using PyTorch
class RevenueModel(nn.Module):
    def __init__(self):
        super(RevenueModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(X_train_scaled.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)

class CogsModel(nn.Module):
    def __init__(self):
        super(CogsModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(X_train_scaled.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)

class UnitCasesModel(nn.Module):
    def __init__(self):
        super(UnitCasesModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(X_train_scaled.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Initialize models
model_revenue = RevenueModel()
model_cogs = CogsModel()
model_uc = UnitCasesModel()

# Define optimizers and loss function
optimizer_revenue = optim.Adam(model_revenue.parameters(), lr=0.003)
optimizer_cogs = optim.Adam(model_cogs.parameters(), lr=0.003)
optimizer_uc = optim.Adam(model_uc.parameters(), lr=0.1)
criterion = nn.MSELoss()

# Training loop for each model
for epoch in range(100):
    model_revenue.train()
    model_cogs.train()
    model_uc.train()
    for inputs, targets in DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True):
        # Revenue model training
        optimizer_revenue.zero_grad()
        outputs = model_revenue(inputs)
        loss = criterion(outputs, targets[:, 0].unsqueeze(1))  # Assuming Revenue is the first column
        loss.backward()
        optimizer_revenue.step()

        # cogs model training
        optimizer_cogs.zero_grad()
        outputs_cogs = model_cogs(inputs)
        loss_cogs = criterion(outputs_cogs, targets[:, 1].unsqueeze(1))  # Assuming COGS is the second column
        loss_cogs.backward()
        optimizer_cogs.step()

        # unit cases model training
        optimizer_uc.zero_grad()
        outputs_uc = model_uc(inputs)
        loss_uc = criterion(outputs_uc, targets[:, 2].unsqueeze(1))  # Assuming Unit Cases is the third column
        loss_uc.backward()
        optimizer_uc.step()

    print(f"Epoch {epoch+1}: Revenue Loss={loss.item():.4f}, COGS Loss={loss_cogs.item():.4f}, Unit Cases Loss={loss_uc.item():.4f}")

# Save the models
torch.save(model_revenue.state_dict(), 'models/model_revenue.pth')
torch.save(model_cogs.state_dict(), 'models/model_cogs.pth')
torch.save(model_uc.state_dict(), 'models/model_uc.pth')
print("Models saved successfully.")

