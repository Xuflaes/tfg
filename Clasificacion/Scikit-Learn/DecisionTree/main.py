import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import time

# Cargar datos
df = pd.read_csv(r"C:\Users\User\OneDrive\Escritorio\TFG\csv\bank-full.csv", sep=';')

# Codificar variable objetivo
df["y"] = LabelEncoder().fit_transform(df["y"])  # yes → 1, no → 0

# One-hot encoding de variables categóricas
df = pd.get_dummies(df)

X = df.drop(columns=["y"])
y = df["y"]

# ===============================
# Entrenar modelo y visualizar árbol
# ===============================
model = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limitamos profundidad
model.fit(X, y)

plt.figure(figsize=(20, 10))
plot_tree(model, 
          feature_names=X.columns, 
          class_names=["No", "Yes"], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Árbol de Decisión - bank-full.csv (profundidad 3)")
plt.show()

# ===============================
# Validación cruzada con métricas
# ===============================
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracies, train_times, pred_times = [], [], []
start_total = time.time()

for train_idx, test_idx in kf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = DecisionTreeClassifier(random_state=42)
    
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    
    start_pred = time.time()
    y_pred = model.predict(X_test)
    end_pred = time.time()
    
    accuracies.append(accuracy_score(y_test, y_pred))
    train_times.append(end_train - start_train)
    pred_times.append(end_pred - start_pred)

end_total = time.time()

# Resultados
print("==== Árbol de Decisión (scikit-learn - bank-full.csv) ====")
print(f"Accuracy promedio (10-fold): {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Tiempo entrenamiento medio (s): {np.mean(train_times):.4f}")
print(f"Tiempo predicción medio (s):    {np.mean(pred_times):.4f}")
print(f"Tiempo total del programa (s): {end_total - start_total:.4f}")
