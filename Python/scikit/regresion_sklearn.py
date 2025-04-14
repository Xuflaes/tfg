import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Cargar y preparar datos
df = pd.read_csv(r"C:\Users\User\OneDrive\Escritorio\TFG\csv\housing.csv").dropna()

# Codificar variable categórica en variables dummy (one-hot)
df = pd.get_dummies(df, columns=["ocean_proximity"])

# Separar características y variable objetivo
X = df.drop(columns=["median_house_value"]).values
y = df["median_house_value"].values

# 2. Normalización de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Configurar validación cruzada (5-fold)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 4. Inicializar listas para guardar resultados
all_mse, all_mae, all_times = [], [], []

# 5. Bucle de entrenamiento y evaluación por fold
print("📌 Modelo: LinearRegression")
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = LinearRegression()

    start = time.time()
    model.fit(X_train, y_train)              # Entrenar
    y_pred = model.predict(X_test)           # Predecir
    elapsed = time.time() - start            # Tiempo

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"🔁 Fold {fold} - MSE: {mse:.2f}, MAE: {mae:.2f}, Tiempo: {elapsed:.2f}s")

    all_mse.append(mse)
    all_mae.append(mae)
    all_times.append(elapsed)

# 6. Resultados finales promedio
print("\n📊 Resultados Promedio (LinearRegression):")
print(f"📐 MSE Promedio: {np.mean(all_mse):.2f}")
print(f"📐 MAE Promedio: {np.mean(all_mae):.2f}")
print(f"⏱️ Tiempo promedio por fold: {np.mean(all_times):.2f} segundos")
print(f"⏲️ Tiempo total: {np.sum(all_times):.4f} segundos")
