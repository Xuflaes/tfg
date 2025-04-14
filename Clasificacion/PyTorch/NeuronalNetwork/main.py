import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# 1. Cargar y preparar los datos
df = pd.read_csv(r"C:\Users\User\OneDrive\Escritorio\TFG\csv\bank-full.csv", sep=';')
df = shuffle(df, random_state=42)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# 2. Separar características y etiquetas
X = df.drop(columns=["y"])
y = df["y"].values

# 3. One-hot encoding
X_encoded = pd.get_dummies(X)

# 4. Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Convertir a float32 para PyTorch
X_scaled = X_scaled.astype(np.float32)
y = y.astype(np.float32)

# 5. Red neuronal simple (MLP)
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 6. Validación cruzada
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies, train_times, pred_times = [], [], []
start_total = time.time()

for train_index, test_index in kf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Convertir a tensores
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).unsqueeze(1)

    # Inicializar modelo, pérdida, optimizador
    model = MLP(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenamiento
    start_train = time.time()
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    end_train = time.time()

    # Predicción
    start_pred = time.time()
    model.eval()
    with torch.no_grad():
        y_pred_probs = model(X_test_tensor).numpy().flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)
    end_pred = time.time()

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    train_times.append(end_train - start_train)
    pred_times.append(end_pred - start_pred)

# 7. Resultados finales
end_total = time.time()
print("==== Red Neuronal (PyTorch - bank-full.csv) ====")
print(f"Accuracy promedio (10-fold): {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Tiempo entrenamiento medio (s): {np.mean(train_times):.4f}")
print(f"Tiempo predicción medio (s):    {np.mean(pred_times):.4f}")
print(f"Tiempo total del programa (s): {end_total - start_total:.4f}")

