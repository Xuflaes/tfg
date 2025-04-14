import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Cargar dataset
df = pd.read_csv(r"C:\Users\User\OneDrive\Escritorio\TFG\csv\bank-full.csv", sep=';')

# 2. Codificar variables categóricas
for column in df.select_dtypes(include='object').columns:
    df[column] = LabelEncoder().fit_transform(df[column])

# 3. Separar características y etiquetas
X = df.drop(columns=["y"]).values
y = df["y"].values

# 4. Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Validación cruzada
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies, train_times, pred_times = [], [], []

start_total = time.time()

for train_index, test_index in kf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

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

end_total = time.time()

# 6. Resultados
print("==== Random Forest (scikit-learn - bank-full.csv) ====")
print(f"Accuracy promedio (10-fold): {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Tiempo entrenamiento medio (s): {np.mean(train_times):.4f}")
print(f"Tiempo predicción medio (s):    {np.mean(pred_times):.4f}")
print(f"Tiempo total del programa (s): {end_total - start_total:.4f}")
