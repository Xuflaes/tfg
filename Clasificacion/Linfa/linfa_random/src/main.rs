use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn majority_vote(predictions: &[usize]) -> usize {
    let mut counts = HashMap::new();
    for &pred in predictions {
        *counts.entry(pred).or_insert(0) += 1;
    }
    *counts.iter().max_by_key(|&(_, count)| count).unwrap().0
}

fn main() {
    // Abrir el archivo CSV
    let file = File::open(r"C:\Users\User\OneDrive\Escritorio\TFG\csv\bank-full.csv")
    .expect("No se puede abrir el archivo");
    let reader = BufReader::new(file);

    let mut features = Vec::new();
    let mut targets = Vec::new();

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

    let k = 10;
    let fold_size = x.nrows() / k;
    let mut accuracies = Vec::new();
    let num_trees = 100;
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

        let _train_ds = Dataset::new(x_train_arr.clone(), y_train_arr.clone());

        // Entrenar múltiples árboles
        let mut tree_preds: Vec<Vec<usize>> = vec![Vec::new(); x_test_arr.nrows()];
        let mut rng = thread_rng();

        for _ in 0..num_trees {
            // Bootstrapping
            let sample_indices: Vec<usize> = (0..x_train_arr.nrows())
                .collect::<Vec<_>>()
                .choose_multiple(&mut rng, x_train_arr.nrows())
                .cloned()
                .collect();

            let sample_x = Array2::from_shape_vec(
                (sample_indices.len(), 16),
                sample_indices
                    .iter()
                    .flat_map(|&i| x_train_arr.row(i).to_vec())
                    .collect(),
            )
            .unwrap();
            let sample_y = Array1::from(sample_indices.iter().map(|&i| y_train_arr[i]).collect::<Vec<_>>());
            let sample_ds = Dataset::new(sample_x, sample_y);

            // Entrenar árbol y predecir
            let model = DecisionTree::params().fit(&sample_ds).unwrap();
            let preds = model.predict(&Dataset::new(x_test_arr.clone(), y_test_arr.clone()));

            for (j, pred) in preds.iter().enumerate() {
                tree_preds[j].push(*pred);
            }
        }

        // Votación mayoritaria
        let final_preds: Vec<usize> = tree_preds
            .iter()
            .map(|votes| majority_vote(votes))
            .collect();

        let correct = final_preds
            .iter()
            .zip(y_test_arr.iter())
            .filter(|(pred, actual)| *pred == *actual)
            .count();

        let accuracy = correct as f32 / y_test_arr.len() as f32;
        accuracies.push(accuracy);
    }

    let mean_acc: f32 = accuracies.iter().sum::<f32>() / accuracies.len() as f32;
    let duration = start_total.elapsed();

    println!("==== Random Forest (Linfa simulado - bank-full.csv) ====");
    println!("Accuracy medio (10-fold): {:.2}%", mean_acc * 100.0);
    println!("Tiempo total: {:.2}ms", duration.as_secs_f64() * 1000.0);
}
