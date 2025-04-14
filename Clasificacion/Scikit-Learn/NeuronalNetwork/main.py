import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# 1. Cargar datos
df = pd.read_csv(r"C:\Users\User\OneDrive\Escritorio\TFG\csv\bank-full.csv", sep=';')
df = shuffle(df, random_state=42)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# 2. Separar características y etiquetas
X = df.drop(columns=["y"])
y = df["y"].values

# 3. One-hot encoding + normalización
X_encoded = pd.get_dummies(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# 4. Validación cruzada
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies, train_times, pred_times = [], [], []
start_total = time.time()

for train_index, test_index in kf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    start_pred = time.time()
    y_pred = model.predict(X_test)
    end_pred = time.time()

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    train_times.append(end_train - start_train)
    pred_times.append(end_pred - start_pred)

# 5. Resultados
end_total = time.time()
print("==== Red Neuronal (MLPClassifier - bank-full.csv) ====")
print(f"Accuracy promedio (10-fold): {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Tiempo entrenamiento medio (s): {np.mean(train_times):.4f}")
print(f"Tiempo predicción medio (s):    {np.mean(pred_times):.4f}")
print(f"Tiempo total del programa (s): {end_total - start_total:.4f}")
