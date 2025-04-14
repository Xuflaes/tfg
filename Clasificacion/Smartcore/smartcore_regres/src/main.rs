use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;
use smartcore::metrics::accuracy;
use smartcore::linear::logistic_regression::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use rand::seq::SliceRandom;
use rand::thread_rng;

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("C:/Users/User/OneDrive/Escritorio/TFG/csv/bank-full.csv")?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Leer encabezado
    let header = lines.next().unwrap()?;
    let headers: Vec<&str> = header.split(';').collect();
    let feature_len = headers.len() - 1;

    let mut data: Vec<(Vec<f64>, f64)> = Vec::new();

    // Inicializar codificadores
    let mut encoders: Vec<HashMap<String, f64>> = vec![HashMap::new(); headers.len()];

    for line in lines {
        let line = line?;
        let parts: Vec<&str> = line.trim().split(';').collect();
        if parts.len() != headers.len() {
            continue;
        }

        let mut row = Vec::new();
        for i in 0..feature_len {
            let value = parts[i].trim_matches('"');
            let encoded = if let Ok(v) = value.parse::<f64>() {
                v
            } else {
                let encoder = &mut encoders[i];
                if !encoder.contains_key(value) {
                    let new_val = encoder.len() as f64;
                    encoder.insert(value.to_string(), new_val);
                }
                *encoder.get(value).unwrap()
            };
            row.push(encoded);
        }

        let label_str = parts[feature_len].trim_matches('"');
        let encoder = &mut encoders[feature_len];
        if !encoder.contains_key(label_str) {
            let new_val = encoder.len() as f64;
            encoder.insert(label_str.to_string(), new_val);
        }
        let label = *encoder.get(label_str).unwrap();

        data.push((row, label));
    }

    // Barajar y separar
    let mut rng = thread_rng();
    data.shuffle(&mut rng);

    let (features, labels): (Vec<_>, Vec<_>) = data.into_iter().unzip();
    let nrows = labels.len();
    let ncols = feature_len;
    let flat_x: Vec<f64> = features.into_iter().flatten().collect();
    let x_matrix = DenseMatrix::from_array(nrows, ncols, &flat_x);
    let y = labels;

    // Validación cruzada 10-fold
    let k = 10;
    let fold_size = y.len() / k;
    let mut scores = Vec::new();
    let start_total = Instant::now();

    for i in 0..k {
        let start = i * fold_size;
        let end = if i == k - 1 { y.len() } else { (i + 1) * fold_size };

        let mut x_test_data = vec![];
        for idx in start..end {
            x_test_data.extend_from_slice(&x_matrix.get_row(idx));
        }
        let x_test = DenseMatrix::from_array(end - start, ncols, &x_test_data);
        let y_test = &y[start..end];

        let mut x_train_data = vec![];
        let mut y_train = vec![];
        for idx in 0..y.len() {
            if idx < start || idx >= end {
                x_train_data.extend_from_slice(&x_matrix.get_row(idx));
                y_train.push(y[idx]);
            }
        }

        let x_train = DenseMatrix::from_array(y_train.len(), ncols, &x_train_data);

        // Crear y ajustar el modelo de regresión logística
        let model = LogisticRegression::fit(
            &x_train,
            &y_train,
            LogisticRegressionParameters::default(),
        )?;
        let y_pred = model.predict(&x_test)?;
        let acc = accuracy(&y_test.to_vec(), &y_pred);
        scores.push(acc);
    }

    let duration = start_total.elapsed();
    let avg_acc = scores.iter().copied().sum::<f64>() / scores.len() as f64;

    println!("==== Regresión Logística (SmartCore - bank-full.csv) ====");
    println!("Accuracy promedio (10-fold): {:.4}", avg_acc);
    println!("Tiempo total de ejecución: {:.2?}", duration);

    Ok(())
}
