use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::accuracy;
use smartcore::svm::svc::{SVC, SVCParameters};
use smartcore::svm::LinearKernel;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};
use rand::seq::SliceRandom;
use rand::thread_rng;

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("C:/Users/User/OneDrive/Escritorio/TFG/csv/bank-full.csv")?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let header = lines.next().unwrap()?;
    let headers: Vec<&str> = header.split(';').collect();
    let feature_len = headers.len() - 1;

    let mut data: Vec<(Vec<f64>, usize)> = Vec::new();
    let mut encoders: Vec<HashMap<String, usize>> = vec![HashMap::new(); headers.len()];

    for line in lines {
        let line = line?;
        let parts: Vec<&str> = line.trim().split(';').collect();
        if parts.len() != headers.len() {
            continue;
        }

        let mut row = Vec::new();
        for i in 0..feature_len {
            let value = parts[i].trim_matches('"');
            let encoder = &mut encoders[i];
            let new_val = encoder.len();
            let encoded = if let Ok(v) = value.parse::<f64>() {
                v
            } else {
                *encoder.entry(value.to_string()).or_insert(new_val) as f64
            };
            row.push(encoded);
        }

        let label_str = parts[feature_len].trim_matches('"');
        let encoder = &mut encoders[feature_len];
        let new_label = encoder.len();
        let label = *encoder.entry(label_str.to_string()).or_insert(new_label);

        data.push((row, label));
    }

    let mut rng = thread_rng();
    data.shuffle(&mut rng);

    let k = 10;
    let fold_size = data.len() / k;
    let mut accuracies = Vec::new();
    let mut train_durations = Vec::new();
    let mut pred_durations = Vec::new();
    let start_total = Instant::now();

    for i in 0..k {
        let start = i * fold_size;
        let end = if i == k - 1 { data.len() } else { (i + 1) * fold_size };

        let (test_set, train_set): (Vec<_>, Vec<_>) = data.iter().cloned().enumerate()
            .partition(|(idx, _)| *idx >= start && *idx < end);
        let test_set: Vec<_> = test_set.into_iter().map(|(_, v)| v).collect();
        let train_set: Vec<_> = train_set.into_iter().map(|(_, v)| v).collect();

        let (x_train, y_train): (Vec<_>, Vec<_>) = train_set.into_iter().unzip();
        let (x_test, y_test): (Vec<_>, Vec<_>) = test_set.into_iter().unzip();

        // Verificar que es un problema binario
        let mut classes = y_train.clone();
        classes.sort();
        classes.dedup();

        if classes.len() != 2 {
            panic!("SVM binario solo admite dos clases. Encontradas: {:?}", classes);
        }

        let class_a = classes[0];
        let class_b = classes[1];

        // Remapear etiquetas a {-1.0, 1.0}
        let y_train_mapped: Vec<i32> = y_train.iter().map(|&y| if y == class_a { -1 } else { 1 }).collect();
        let y_test_mapped: Vec<i32> = y_test.iter().map(|&y| if y == class_a { -1 } else { 1 }).collect();


        let train_matrix = DenseMatrix::from_2d_vec(&x_train);
        let test_matrix = DenseMatrix::from_2d_vec(&x_test);

        let params = SVCParameters::default().with_c(1.0).with_kernel(LinearKernel);

        let start_train = Instant::now();
        let model = SVC::fit(&train_matrix, &y_train_mapped, &params)?;
        let train_time = start_train.elapsed();

        let start_pred = Instant::now();
        let y_pred_raw: Vec<i32> = model.predict(&test_matrix)?
            .into_iter()
            .map(|v| v as i32)
            .collect();
        let pred_time = start_pred.elapsed();

        // Convertir las predicciones {-1.0, 1.0} de vuelta a {clase_a, clase_b}
        let y_pred: Vec<usize> = y_pred_raw
            .iter()
            .map(|&val| if val < 0 { class_a } else { class_b })
            .collect();

        let acc = accuracy(&y_test, &y_pred);

        accuracies.push(acc);
        train_durations.push(train_time);
        pred_durations.push(pred_time);
    }

    let duration = start_total.elapsed();
    let avg_acc: f64 = accuracies.iter().sum::<f64>() / k as f64;
    let avg_train: Duration = train_durations.iter().sum::<Duration>() / k as u32;
    let avg_pred: Duration = pred_durations.iter().sum::<Duration>() / k as u32;

    println!("==== SVM (SmartCore - bank-full.csv, 10-fold CV) ====");
    println!("Accuracy promedio (10-fold): {:.4}", avg_acc);
    println!("Tiempo entrenamiento medio: {:.4?}", avg_train);
    println!("Tiempo predicción medio: {:.4?}", avg_pred);
    println!("Tiempo total ejecucion: {:.4?}", duration);

    Ok(())
}
