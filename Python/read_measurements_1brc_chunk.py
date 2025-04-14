import time

file_path = r"C:\Users\User\OneDrive\Escritorio\TFG\csv\measurements-1000000000.txt"
chunk_size = 10_000_000  # Número de líneas por chunk

resultado = {}
start_time = time.time()

with open(file_path, "r", encoding="utf-8") as file:
    chunk_counter = 0
    current_chunk = []

    for line_number, line in enumerate(file, start=1):
        current_chunk.append(line)

        # Cuando se completa un chunk, procesarlo
        if line_number % chunk_size == 0:
            chunk_counter += 1
            print(f"📦 Procesando chunk {chunk_counter}...")

            for entry in current_chunk:
                try:
                    estacion, temperatura = entry.strip().split(";")
                    temperatura = float(temperatura)

                    if estacion in resultado:
                        resultado[estacion][0] = min(resultado[estacion][0], temperatura)
                        resultado[estacion][1] = max(resultado[estacion][1], temperatura)
                        resultado[estacion][2] += temperatura
                        resultado[estacion][3] += 1
                    else:
                        resultado[estacion] = [temperatura, temperatura, temperatura, 1]
                except ValueError:
                    continue

            current_chunk.clear()

    # Procesar líneas restantes que no llenaron un chunk completo
    if current_chunk:
        chunk_counter += 1
        print(f"📦 Procesando último chunk {chunk_counter}...")
        for entry in current_chunk:
            try:
                estacion, temperatura = entry.strip().split(";")
                temperatura = float(temperatura)

                if estacion in resultado:
                    resultado[estacion][0] = min(resultado[estacion][0], temperatura)
                    resultado[estacion][1] = max(resultado[estacion][1], temperatura)
                    resultado[estacion][2] += temperatura
                    resultado[estacion][3] += 1
                else:
                    resultado[estacion] = [temperatura, temperatura, temperatura, 1]
            except ValueError:
                continue

# Calcular promedios
for valores in resultado.values():
    valores.append(valores[2] / valores[3])

end_time = time.time()

# Resultados
print(f"\n✅ Lectura completada en {end_time - start_time:.2f} segundos.")
print(f"Total de estaciones procesadas: {len(resultado)}")
for estacion, valores in list(resultado.items())[:5]:
    print(f"{estacion}: Min={valores[0]}, Max={valores[1]}, Prom={valores[4]:.3f}")
