use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() {
    // Abrir el archivo CSV
    let file = File::open(r"C:\Users\User\OneDrive\Escritorio\TFG\csv\bank-full.csv")
    .expect("No se puede abrir el archivo");
    let reader = BufReader::new(file);

    // Variables para las características y etiquetas
    let mut features = Vec::new();
    let mut targets = Vec::new();

    // Mapas para codificar las características y etiquetas
    let mut encoders: Vec<HashMap<String, usize>> = vec![HashMap::new(); 17];

    for (line_idx, line) in reader.lines().enumerate() {
        if line_idx == 0 {
            // Saltar la cabecera
            continue;
        }

        if let Ok(line) = line {
            let parts: Vec<&str> = line.split(';').collect();
            if parts.len() != 17 {
                continue;
            }

            let mut row = Vec::new();
            for i in 0..16 {
                let value = parts[i].trim_matches('"').to_string();
                let map = &mut encoders[i];
                let next_index = map.len();
                let encoded = *map.entry(value).or_insert(next_index);
                row.push(encoded as f64);
            }

            let label = parts[16].trim_matches('"').to_string();
            let map = &mut encoders[16];
            let next_index = map.len();
            let encoded_label = *map.entry(label).or_insert(next_index);
            features.push(row);
            targets.push(encoded_label);
        }
    }

    // Convertir a matrices ndarray
    let x = Array2::from_shape_vec((features.len(), 16), features.concat()).unwrap();
    let y = Array1::from(targets);

    // Validación cruzada (10-fold)
    let k = 10;
    let fold_size = x.nrows() / k;
    let mut accuracies = Vec::new();
    let start_total = Instant::now();

    for i in 0..k {
        let test_start = i * fold_size;
        let test_end = if i == k - 1 { x.nrows() } else { (i + 1) * fold_size };

        let mut x_train = Vec::new();
        let mut y_train = Vec::new();
        let mut x_test = Vec::new();
        let mut y_test = Vec::new();

        for idx in 0..x.nrows() {
            let row = x.row(idx).to_vec();
            if idx >= test_start && idx < test_end {
                x_test.extend(row);
                y_test.push(y[idx]);
            } else {
                x_train.extend(row);
                y_train.push(y[idx]);
            }
        }

        let x_train_arr = Array2::from_shape_vec((y_train.len(), 16), x_train).unwrap();
        let x_test_arr = Array2::from_shape_vec((y_test.len(), 16), x_test).unwrap();
        let y_train_arr = Array1::from(y_train);
        let y_test_arr = Array1::from(y_test);

        let train_ds = Dataset::new(x_train_arr, y_train_arr);
        let test_ds = Dataset::new(x_test_arr, y_test_arr);

        let model = DecisionTree::params().fit(&train_ds).unwrap();
        let pred = model.predict(&test_ds);
        let acc = pred.confusion_matrix(&test_ds).unwrap().accuracy();
        accuracies.push(acc);
    }

    let mean_acc: f32 = accuracies.iter().sum::<f32>() / accuracies.len() as f32;
    let duration = start_total.elapsed();

    println!("==== Árbol de Decisión (Linfa - bank-full.csv) ====");
    println!("Accuracy medio (10-fold): {:.2}%", mean_acc * 100.0);
    println!("Tiempo total: {:.2?}", duration);
}
