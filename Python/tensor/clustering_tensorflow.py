import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. Cargar datos
df = pd.read_csv(r"C:\Users\User\OneDrive\Escritorio\TFG\csv\housing.csv")
df = df.dropna()  # Eliminar filas con valores nulos

# 2. Preprocesamiento
# Codificamos la variable categórica 'ocean_proximity' y eliminamos la etiqueta (no supervisado)
df = df.drop(columns=[
    "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income",
    "median_house_value", "ocean_proximity"
])  # Eliminamos la variable objetivo

# 3. Normalización (z-score)
X = (df - df.mean()) / df.std()
X_tensor = tf.convert_to_tensor(X.values, dtype=tf.float32)

# 🔢 Guardar número de dimensiones para el título
num_vars = X.shape[1]

# 4. Parámetros del clustering
k = 5 # Número de clusters
num_iters = 10  # Número de iteraciones del algoritmo
num_points = X_tensor.shape[0]
dims = X_tensor.shape[1]

# 5. Inicialización aleatoria de centroides
indices = tf.random.shuffle(tf.range(num_points))[:k]
centroids = tf.gather(X_tensor, indices)

# ⏱️ Iniciar temporizador del bucle KMeans
start_kmeans = time.time()

# 6. Algoritmo K-Means (bucle de entrenamiento)
for i in range(num_iters):
    # Expandir dimensiones para calcular distancias entre puntos y centroides
    expanded_points = tf.expand_dims(X_tensor, 1)       # [N, 1, D]
    expanded_centroids = tf.expand_dims(centroids, 0)   # [1, K, D]

    # Calcular distancias cuadradas y asignar al centroide más cercano
    distances = tf.reduce_sum(tf.square(expanded_points - expanded_centroids), axis=2)  # [N, K]
    assignments = tf.argmin(distances, axis=1)  # [N]

    # Recalcular centroides
    new_centroids = []
    for c in range(k):
        mask = tf.equal(assignments, c)
        cluster_points = tf.boolean_mask(X_tensor, mask)
        new_centroids.append(tf.reduce_mean(cluster_points, axis=0))
    
    centroids = tf.stack(new_centroids)

    #print(f"Iteración {i+1} completada.")

# ⏱️ Tiempo total de clustering
kmeans_time = time.time() - start_kmeans

# 7. Mostrar el número de puntos por cluster
unique, counts = np.unique(assignments.numpy(), return_counts=True)
for u, c in zip(unique, counts):
    print(f"Cluster {u}: {c} puntos")

# 8. Visualización (2 primeras variables)
X_np = X_tensor.numpy()
assign_np = assignments.numpy()

plt.figure(figsize=(8, 6))
for i in range(k):
    plt.scatter(X_np[assign_np == i, 0], X_np[assign_np == i, 1], label=f'Cluster {i}', alpha=0.5)

# Título con valores dinámicos
plt.title(f"Clustering con KMeans (k={k}) - {num_vars} variables (2D projection)")
plt.xlabel("Variable 1 (normalizada)")
plt.ylabel("Variable 2 (normalizada)")
plt.legend()
plt.tight_layout()

# Guardar imagen
plt.savefig("Figure_Clustering_TensorFlow.png")

# ⏱️ Tiempo total de ejecución (hasta antes del .show())
print(f"\n⏱️ Tiempo clustering: {kmeans_time:.4f} segundos")

