use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Array2, Axis, s};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() {
    let file_path = r"C:\Users\User\OneDrive\Escritorio\TFG\csv\bank-full.csv";
    let file = File::open(file_path).expect("No se puede abrir el archivo");
    let reader = BufReader::new(file);

    let mut features = Vec::new();
    let mut targets = Vec::new();
    let mut encoders: Vec<HashMap<String, usize>> = vec![HashMap::new(); 17];

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split(';').collect();

        if line_num == 0 || parts.len() != 17 {
            continue;
        }

        let mut row = Vec::new();
        for i in 0..16 {
            let value = parts[i].trim_matches('"');
            let encoded = value.parse::<f64>().unwrap_or_else(|_| {
                let encoder = &mut encoders[i];
                let len = encoder.len();
                *encoder.entry(value.to_string()).or_insert(len) as f64
            });
            row.push(encoded);
        }        

        let label_str = parts[16].trim_matches('"');
        let label = match label_str {
            "yes" => 1,
            "no" => 0,
            _ => continue,
        };

        features.push(row);
        targets.push(label);
    }

    let n_samples = features.len();
    let n_features = features[0].len();

    let flat_features: Vec<f64> = features.into_iter().flatten().collect();
    let x = Array2::from_shape_vec((n_samples, n_features), flat_features).unwrap();
    let y = Array1::from(targets);

    let k = 10;
    let fold_size = n_samples / k;
    let mut accuracies: Vec<f64> = Vec::new();
    let start_total = Instant::now();

    for i in 0..k {
        let test_start = i * fold_size;
        let test_end = if i == k - 1 { n_samples } else { (i + 1) * fold_size };

        let x_train = ndarray::concatenate(
            Axis(0),
            &[x.slice(s![0..test_start, ..]), x.slice(s![test_end.., ..])]
        ).unwrap();

        let y_train = Array1::from_iter(
            y.slice(s![0..test_start])
                .iter()
                .chain(y.slice(s![test_end..]).iter())
                .cloned(),
        );

        let x_test = x.slice(s![test_start..test_end, ..]).to_owned();
        let y_test = y.slice(s![test_start..test_end]).to_owned();

        let train_ds = Dataset::new(x_train, y_train);
        let test_ds = Dataset::new(x_test, y_test);

        let model = LogisticRegression::default().fit(&train_ds).unwrap();
        let pred = model.predict(&test_ds);
        let acc = pred.confusion_matrix(&test_ds).unwrap().accuracy();
        accuracies.push(acc as f64);
    }

    let mean_acc: f64 = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
    let duration = start_total.elapsed();

    println!("==== Regresión Logística (Linfa - bank-full.csv) ====");
    println!("Accuracy medio (10-fold): {:.2}%", mean_acc * 100.0);
    println!("Tiempo total: {:.2?}", duration);
}
