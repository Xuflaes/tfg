import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import time

# 1. Cargar y preparar datos
# Leemos el CSV, eliminamos valores nulos y nos quedamos con 2 columnas numéricas
df = pd.read_csv(r"C:\Users\User\OneDrive\Escritorio\TFG\csv\housing.csv").dropna()
df = df.drop(columns=[
    "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income",
    "median_house_value", "ocean_proximity"
])   # Quedarán solo 'longitude' y 'latitude'
num_vars = df.shape[1]

# 2. Normalización de los datos
# Estandarizamos las dos variables restantes (media 0, varianza 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Clustering con KMeans
# Aplicamos KMeans para agrupar en k clusters
start_total = time.time()
k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
end_total = time.time()
# 4. Mostrar número de muestras por cluster
# Contamos cuántos puntos hay en cada cluster
unique, counts = np.unique(clusters, return_counts=True)
for i, c in zip(unique, counts):
    print(f"Cluster {i}: {c} muestras")

# 5. Visualización 2D
# Dibujamos un scatter plot con las dos variables normalizadas
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='tab10', alpha=0.5)
plt.title(f"Clustering con KMeans (k={k}) - {num_vars} variables")
plt.xlabel("Feature 1 (normalizada)")
plt.ylabel("Feature 2 (normalizada)")
plt.tight_layout()
plt.savefig("scikit_clustering_k_n")

# Mostrar tiempos
print(f"\n✅ Tiempo total de ejecución: {end_total - start_total:.4f} segundos")

