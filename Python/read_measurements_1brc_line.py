import time

file_path = r"C:\Users\User\OneDrive\Escritorio\TFG\csv\measurements-1000000000.txt"

# Diccionario para almacenar resultados parciales
resultado = {}

start_time = time.time()

# Leer línea por línea
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        try:
            estacion, temperatura = line.strip().split(";")
            temperatura = float(temperatura)

            if estacion in resultado:
                resultado[estacion][0] = min(resultado[estacion][0], temperatura)  # Min
                resultado[estacion][1] = max(resultado[estacion][1], temperatura)  # Max
                resultado[estacion][2] += temperatura  # Sum
                resultado[estacion][3] += 1  # Contador
            else:
                resultado[estacion] = [temperatura, temperatura, temperatura, 1]

        except ValueError:
            continue  # Saltar líneas corruptas

# Calcular promedio final
for estacion, valores in resultado.items():
    valores.append(valores[2] / valores[3])  # Agregar promedio

end_time = time.time()
elapsed_time = end_time - start_time  # Tiempo en segundos

# Imprimir resultados
print(f"\n✅ Lectura completada en {elapsed_time:.2f} segundos.")
print(f"Total de estaciones procesadas: {len(resultado)}")

# Mostrar solo algunas estaciones de ejemplo
for estacion, valores in list(resultado.items())[:5]:
    print(f"{estacion}: Min={valores[0]}, Max={valores[1]}, Prom={valores[4]:.3f}")
