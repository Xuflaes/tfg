import pandas as pd
import torch
import matplotlib.pyplot as plt
import time

# ⏱️ Tiempo total
start_total = time.time()

# 1. Cargar y preparar datos
df = pd.read_csv(r"C:\Users\User\OneDrive\Escritorio\TFG\csv\housing.csv").dropna()
df = df.drop(columns=[
    "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income",
    "median_house_value", "ocean_proximity"
])  # Eliminar columnas no deseadas

# 2. Normalización z-score
X = (df - df.mean()) / df.std()
X_tensor = torch.tensor(X.values, dtype=torch.float32)

# 🔢 Variables para título y nombre de archivo
num_vars = X_tensor.shape[1]

# 3. Inicialización de KMeans
k = 3
n_iters = 10
n_samples, n_features = X_tensor.shape

# Selección aleatoria de centroides
indices = torch.randperm(n_samples)[:k]
centroids = X_tensor[indices]

# ⏱️ Tiempo clustering
start_kmeans = time.time()

for i in range(n_iters):
    expanded_points = X_tensor.unsqueeze(1)       # [N, 1, D]
    expanded_centroids = centroids.unsqueeze(0)   # [1, K, D]
    distances = torch.sum((expanded_points - expanded_centroids) ** 2, dim=2)
    assignments = torch.argmin(distances, dim=1)

    new_centroids = []
    for j in range(k):
        cluster_points = X_tensor[assignments == j]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(dim=0))
        else:
            new_centroids.append(X_tensor[torch.randint(0, n_samples, (1,)).item()])
    centroids = torch.stack(new_centroids)

# ⏱️ Fin tiempo clustering
kmeans_time = time.time() - start_kmeans

# 4. Tamaño de clusters
unique, counts = torch.unique(assignments, return_counts=True)
for u, c in zip(unique.tolist(), counts.tolist()):
    print(f"Cluster {u}: {c} puntos")

# 5. Visualización
X_np = X_tensor.numpy()
assign_np = assignments.numpy()

plt.figure(figsize=(8, 6))
for i in range(k):
    plt.scatter(X_np[assign_np == i, 0], X_np[assign_np == i, 1], label=f'Cluster {i}', alpha=0.5)

# Título y nombre del archivo dinámico
plt.title(f"Clustering con PyTorch (k={k}) - {num_vars} variables (2D)")
plt.xlabel("Variable 1 (normalizada)")
plt.ylabel("Variable 2 (normalizada)")
plt.legend()
plt.tight_layout()

# 🖼️ Guardar imagen
filename = f"pytorch_k{k}_n{num_vars}.png"
plt.savefig(filename)

# ⏱️ Tiempo total
total_time = time.time() - start_total

# 🕒 Mostrar tiempos
print(f"\n⏱️ Tiempo clustering: {kmeans_time:.4f} segundos")
print(f"✅ Tiempo total de ejecución: {total_time:.4f} segundos")
print(f"📁 Imagen guardada como {filename}")

