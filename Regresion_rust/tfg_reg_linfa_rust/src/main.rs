use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2, Axis, s, concatenate};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use csv::ReaderBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    println!("📥 Cargando housing.csv...");

    // 1. Leer y limpiar el CSV
    let file = File::open("C:/Users/User/OneDrive/Escritorio/TFG/csv/housing.csv")?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(file));

    let mut features = Vec::new();
    let mut targets = Vec::new();
    let mut count = 0;

    for result in rdr.records() {
        let record = result?;
        if record.len() != 10 || record.iter().any(|v| v.trim().is_empty()) {
            continue;
        }

        let mut row = Vec::new();
        let mut valid = true;

        for i in 0..8 {
            match record[i].parse::<f64>() {
                Ok(val) => row.push(val),
                Err(_) => {
                    valid = false;
                    break;
                }
            }
        }

        if !valid {
            continue;
        }

        let target = match record[8].parse::<f64>() {
            Ok(v) => v,
            Err(_) => continue,
        };

        features.extend(row);
        targets.push(target);
        count += 1;
    }

    println!("📊 Cargadas {count} muestras");

    // 2. Convertir a arrays de ndarray
    let x_all = Array2::from_shape_vec((count, 8), features)?;
    let y_all = Array1::from_vec(targets);
    let n_samples = x_all.nrows();

    // 3. Validación cruzada manual
    let n_folds = 10;
    let fold_size = n_samples / n_folds;

    let mut all_mse = Vec::new();
    let mut all_mae = Vec::new();
    let mut all_times = Vec::new();

    for fold in 0..n_folds {
        println!("\n🔁 Fold {}/{}", fold + 1, n_folds);

        let start = fold * fold_size;
        let end = if fold == n_folds - 1 {
            n_samples
        } else {
            (fold + 1) * fold_size
        };

        // 4. Separar datos de test
        let x_test = x_all.slice(s![start..end, ..]).to_owned();
        let y_test = y_all.slice(s![start..end]).to_owned();

        // 5. Construir datos de entrenamiento concatenando top + bottom
        let x_train_top = x_all.slice(s![..start, ..]).to_owned();
        let x_train_bottom = x_all.slice(s![end.., ..]).to_owned();
        let x_train = concatenate(Axis(0), &[x_train_top.view(), x_train_bottom.view()])?;

        let y_train_top = y_all.slice(s![..start]).to_owned();
        let y_train_bottom = y_all.slice(s![end..]).to_owned();
        let y_train = concatenate(Axis(0), &[y_train_top.view(), y_train_bottom.view()])?;

        // 6. Crear datasets
        let train = Dataset::new(x_train, y_train);
        let test = Dataset::new(x_test, y_test);

        // 7. Entrenar y medir tiempo
        let timer = Instant::now();
        let model = LinearRegression::default().fit(&train)?;
        let pred = model.predict(&test);
        let elapsed = timer.elapsed().as_secs_f64();

        // 8. Calcular métricas
        let mse = pred.mean_squared_error(&test.targets()).unwrap();
        let mae = pred.mean_absolute_error(&test.targets()).unwrap();

        println!("✅ MSE: {:.2}, MAE: {:.2}, Tiempo: {:.2}s", mse, mae, elapsed);

        all_mse.push(mse);
        all_mae.push(mae);
        all_times.push(elapsed);
    }

    // 9. Mostrar resultados finales
    let avg_mse = all_mse.iter().sum::<f64>() / all_mse.len() as f64;
    let avg_mae = all_mae.iter().sum::<f64>() / all_mae.len() as f64;
    let total_time = all_times.iter().sum::<f64>();
    let avg_time = total_time / all_times.len() as f64;

    println!("\n📊 Resultados Finales (linfa - LinearRegression):");
    println!("📐 MSE Promedio: {:.2}", avg_mse);
    println!("📐 MAE Promedio: {:.2}", avg_mae);
    println!("⏱️ Tiempo promedio por fold: {:.2} segundos", avg_time);
    println!("⏲️ Tiempo total: {:.4} segundos", total_time);

    Ok(())
}
