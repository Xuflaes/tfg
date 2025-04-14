use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::*;
use smartcore::metrics::{mean_absolute_error, mean_squared_error};
use smartcore::linalg::basic::arrays::Array2; // Necesario para usar `.get_row()`
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use csv::ReaderBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    // ▶️ Leer y limpiar el CSV
    let file = File::open("C:/Users/User/OneDrive/Escritorio/TFG/csv/housing.csv")?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(BufReader::new(file));

    let mut features = Vec::new();
    let mut targets = Vec::new();
    let mut num_features = 0;

    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        if record.iter().any(|v| v.trim().is_empty()) {
            continue;
        }

        if i == 0 {
            num_features = record.len() - 2; // Asumimos que la penúltima es la target
        }

        for j in 0..num_features {
            features.push(record[j].parse::<f64>()?);
        }

        targets.push(record[num_features].parse::<f64>()?);
    }

    let n_samples = targets.len();
    println!("📊 Dataset cargado: {n_samples} muestras, {num_features} características");

    // ▶️ Crear matriz X y vector Y
    let x_all = DenseMatrix::new(n_samples, num_features, features.clone(), false);
    let y_all = targets;

    // ▶️ Configuración de validación cruzada
    let n_folds = 10;
    let fold_size = n_samples / n_folds;

    let mut all_mse = Vec::new();
    let mut all_mae = Vec::new();
    let mut all_times = Vec::new();

    // ▶️ Validación cruzada manual
    for fold in 0..n_folds {
        println!("\n🔁 Fold {}/{}", fold + 1, n_folds);

        let start = fold * fold_size;
        let end = if fold == n_folds - 1 { n_samples } else { (fold + 1) * fold_size };

        // ▶️ Dividir en train y test
        let mut x_train_data = Vec::new();
        let mut y_train = Vec::new();
        let mut x_test_data = Vec::new();
        let mut y_test = Vec::new();

        for i in 0..n_samples {
            let row = x_all.get_row(i).iterator(0).map(|v| *v).collect::<Vec<f64>>();
            if i >= start && i < end {
                x_test_data.extend(row);
                y_test.push(y_all[i]);
            } else {
                x_train_data.extend(row);
                y_train.push(y_all[i]);
            }
        }

        let x_train = DenseMatrix::new(y_train.len(), num_features, x_train_data, false);
        let x_test = DenseMatrix::new(y_test.len(), num_features, x_test_data, false);

        // ▶️ Entrenamiento y predicción
        let timer = Instant::now();
        let model = LinearRegression::fit(&x_train, &y_train, Default::default())?;
        let y_pred = model.predict(&x_test)?;
        let elapsed = timer.elapsed().as_secs_f64();

        // ▶️ Métricas
        let mse = mean_squared_error(&y_test, &y_pred);
        let mae = mean_absolute_error(&y_test, &y_pred);

        println!("✅ MSE: {:.2}, MAE: {:.2}, Tiempo: {:.2}s", mse, mae, elapsed);

        all_mse.push(mse);
        all_mae.push(mae);
        all_times.push(elapsed);
    }

    // ▶️ Promedios finales
    let avg = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
    let total_time: f64 = all_times.iter().sum();

    println!("\n📊 Resultados Finales (smartcore - LinearRegression):");
    println!("📐 MSE Promedio: {:.2}", avg(&all_mse));
    println!("📐 MAE Promedio: {:.2}", avg(&all_mae));
    println!("⏱️ Tiempo promedio por fold: {:.2}s", avg(&all_times));
    println!("⏲️ Tiempo total: {:.4}s", total_time);

    Ok(())
}
