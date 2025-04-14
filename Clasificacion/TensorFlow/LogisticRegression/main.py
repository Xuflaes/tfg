import pandas as pd
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Cargar y preparar datos
df = pd.read_csv(r"C:\Users\User\OneDrive\Escritorio\TFG\csv\bank-full.csv", sep=';')
df = shuffle(df, random_state=42)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# 2. Separar características y etiquetas
X = df.drop(columns=["y"])
y = df["y"]

# 3. Codificar variables categóricas (OneHot)
X_encoded = pd.get_dummies(X)

# 4. Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# 5. Validación cruzada
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
start_total = time.time()
accuracies, train_times, pred_times = [], [], []

for train_index, test_index in kf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 6. Modelo de regresión logística = red neuronal sin capa oculta
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Solo una neurona, sin capas ocultas
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    start_train = time.time()
    model.fit(X_train, y_train, epochs=100, batch_size=256, verbose=0)
    end_train = time.time()

    start_pred = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    end_pred = time.time()

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    train_times.append(end_train - start_train)
    pred_times.append(end_pred - start_pred)

# 7. Resultados
end_total = time.time()
print("==== Regresión Logística (TensorFlow - bank-full.csv) ====")
print(f"Accuracy promedio (10-fold): {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Tiempo entrenamiento medio (s): {np.mean(train_times):.4f}")
print(f"Tiempo predicción medio (s):    {np.mean(pred_times):.4f}")
print(f"Tiempo total del programa (s): {end_total - start_total:.4f}")
