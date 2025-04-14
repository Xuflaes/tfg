use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "C:/Users/User/OneDrive/Escritorio/TFG/csv/measurements-1000000000.txt";

    let start_time = Instant::now();
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut resultado: HashMap<String, (f64, f64, f64, u64)> = HashMap::new();

    // Leer línea por línea
    for line in reader.lines() {
        if let Ok(row) = line {
            if let Some((estacion, temp_str)) = row.split_once(';') {
                if let Ok(temperatura) = temp_str.parse::<f64>() {
                    let entry = resultado.entry(estacion.to_string()).or_insert((temperatura, temperatura, 0.0, 0));


                    // Actualizar valores
                    entry.0 = entry.0.min(temperatura);  // Min
                    entry.1 = entry.1.max(temperatura);  // Max
                    entry.2 += temperatura;  // Sum
                    entry.3 += 1;  // Contador
                }
            }
        }
    }

    let duration = start_time.elapsed();

    // Calcular promedio
    let mut resultado_promedio: HashMap<String, (f64, f64, f64)> = HashMap::new();
    for (estacion, (min, max, suma, count)) in &resultado {
        resultado_promedio.insert(estacion.clone(), (*min, *max, suma / *count as f64));
    }

    println!("\n✅ Lectura completada en {:?}.", duration);
    println!("Total de estaciones procesadas: {}", resultado_promedio.len());

    // Mostrar algunos resultados
    for (estacion, (min, max, prom)) in resultado_promedio.iter().take(5) {
        println!("{}: Min={:.2}, Max={:.2}, Prom={:.3}", estacion, min, max, prom);
    }

    Ok(())
}