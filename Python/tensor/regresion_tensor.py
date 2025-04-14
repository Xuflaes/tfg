# Importar librerías necesarias
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import KFold

# 1. Cargar el dataset desde archivo CSV
df = pd.read_csv(r"C:\Users\User\OneDrive\Escritorio\TFG\csv\housing.csv")

# Eliminar filas con valores nulos (muy importante para evitar errores durante el entrenamiento)
df = df.dropna()

# 2. Codificar la columna categórica "ocean_proximity" en variables dummy (one-hot encoding)
df = pd.get_dummies(df, columns=["ocean_proximity"])

# 3. Separar las características (X) y la variable objetivo (y)
X = df.drop("median_house_value", axis=1)              # Todas las columnas excepto la variable objetivo
y = df["median_house_value"].values.astype(np.float32) # Convertir la salida a float32

# Convertir X a un array NumPy de tipo float32
X = X.astype(np.float32).to_numpy()

# 4. Normalización de características con z-score: (valor - media) / desviación típica
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# 5. Configurar validación cruzada (KFold con 5 particiones aleatorias)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Listas para guardar resultados por fold
all_mae = []   # Mean Absolute Error
all_loss = []  # Mean Squared Error (loss)
all_times = [] # Tiempo de entrenamiento + evaluación

# Bucle para entrenar y evaluar el modelo en cada fold
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    print(f"\n🔁 Fold {fold}/5")

    # Dividir datos de entrenamiento y prueba para este fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Crear datasets de TensorFlow para entrenamiento y prueba
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(256).shuffle(1000)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(256)

    # Definir la arquitectura del modelo (red neuronal)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)), # Capa oculta con 64 neuronas
        tf.keras.layers.Dense(32, activation="relu"),                                  # Otra capa oculta con 32 neuronas
        tf.keras.layers.Dense(1)                                                       # Capa de salida (regresión)
    ])

    # Compilar el modelo con optimizador Adam, pérdida MSE, y métrica MAE
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Medir el tiempo de entrenamiento y evaluación
    start_time = time.time()
    model.fit(train_ds, epochs=1000, verbose=0)               # Entrenar el modelo (silenciosamente)
    loss, mae = model.evaluate(test_ds, verbose=0)          # Evaluar el modelo
    elapsed = time.time() - start_time                      # Tiempo total del fold

    # Mostrar resultados del fold
    print(f"✅ Fold {fold} - MSE: {loss:.2f}, MAE: {mae:.2f}, Tiempo: {elapsed:.2f} s")

    # Guardar resultados
    all_loss.append(loss)
    all_mae.append(mae)
    all_times.append(elapsed)

# 6. Mostrar resultados finales después de todos los folds
print("\n📊 Resultados Finales:")
print(f"📐 MSE Promedio: {np.mean(all_loss):.2f}")
print(f"📐 MAE Promedio: {np.mean(all_mae):.2f}")
print(f"⏱️ Tiempo promedio por fold: {np.mean(all_times):.2f} segundos")
print(f"⏲️ Tiempo total: {np.sum(all_times):.2f} segundos")
