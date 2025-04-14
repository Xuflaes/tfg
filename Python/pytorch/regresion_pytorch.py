import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import time

# 1. Cargar y preparar datos
df = pd.read_csv(r"C:\Users\User\OneDrive\Escritorio\TFG\csv\housing.csv").dropna()
df = pd.get_dummies(df, columns=["ocean_proximity"])

y = df["median_house_value"].values.astype(np.float32)
X = df.drop(columns=["median_house_value"])
X = ((X - X.mean()) / X.std()).astype(np.float32).values

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).view(-1, 1)

# 2. Definir modelo de red neuronal para regresión
class Regressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# 3. Validación cruzada K-Fold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
all_mae, all_mse, all_times = [], [], []
start_total = time.time()

print("📌 Modelo: Red Neuronal (PyTorch)")
for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor), 1):
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

    model = Regressor(X_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

    start = time.time()
    model.train()
    for epoch in range(1000):  # Misma cantidad que scikit (se puede ajustar)
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = loss_fn(predictions, y_test).item()
        mae = nn.L1Loss()(predictions, y_test).item()

    elapsed = time.time() - start
    print(f"🔁 Fold {fold} - MSE: {mse:.2f}, MAE: {mae:.2f}, Tiempo: {elapsed:.2f}s")

    all_mse.append(mse)
    all_mae.append(mae)
    all_times.append(elapsed)

# 4. Resultados finales
end_total = time.time()
print("\n📊 Resultados Promedio (Red Neuronal - PyTorch):")
print(f"📐 MSE Promedio: {np.mean(all_mse):.2f}")
print(f"📐 MAE Promedio: {np.mean(all_mae):.2f}")
print(f"⏱️ Tiempo promedio por fold: {np.mean(all_times):.2f} segundos")
print(f"⏲️ Tiempo total: {end_total - start_total:.4f} segundos")
