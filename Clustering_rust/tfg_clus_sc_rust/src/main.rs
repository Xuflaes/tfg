use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use csv::ReaderBuilder;
use plotters::prelude::*;
use itertools::izip;

fn main() -> Result<(), Box<dyn Error>> {
    // ⏱️ Inicio del tiempo total
    let start_total = Instant::now();

    let k = 5; // Número de clusters
    let selected_indices = 0..7; // Cambia para usar 2, 5 o 7 variables

    // ▶️ Leer CSV
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(File::open("C:/Users/User/OneDrive/Escritorio/TFG/csv/housing.csv")?));

    let mut features = Vec::new();
    let mut count = 0;

    for result in reader.records() {
        let record = result?;
        if record.len() != 10 || record.iter().any(|v| v.trim().is_empty()) {
            continue;
        }

        let row: Result<Vec<f64>, _> = selected_indices.clone().map(|i| record[i].parse::<f64>()).collect();
        match row {
            Ok(r) => {
                features.extend(r);
                count += 1;
            }
            Err(_) => continue,
        }
    }

    let num_features = selected_indices.len();
    let num_samples = count;
    println!("📊 Dataset cargado: {num_samples} muestras, {num_features} variables");

    let x = DenseMatrix::new(num_samples, num_features, features.clone(), false);

    // ▶️ Clustering con KMeans
    let start_kmeans = Instant::now();

    let model = KMeans::<f64, usize, DenseMatrix<f64>, Vec<usize>>::fit(
        &x,
        KMeansParameters::default().with_k(k),
    )?;
    let labels = model.predict(&x)?;

    let elapsed_kmeans = start_kmeans.elapsed().as_secs_f64();
    println!("✅ Clustering completado en {:.3} segundos", elapsed_kmeans);

    // ▶️ Visualización 2D con las 2 primeras variables
    let x_vals: Vec<f64> = features.iter().step_by(num_features).cloned().collect();
    let y_vals: Vec<f64> = features.iter().skip(1).step_by(num_features).cloned().collect();

    let filename = format!("smartcore_k{}_n{}.png", k, num_features);
    println!("🖼️ Generando gráfico {}...", filename);

    let x_min = x_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let root = BitMapBackend::new(&filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("KMeans Clustering (k={}, {} variables)", k, num_features), ("sans-serif", 25))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    for (x, y, label) in izip!(x_vals, y_vals, labels) {
        chart.draw_series(PointSeries::of_element(
            vec![(x, y)],
            3,
            &Palette99::pick(label),
            &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
        ))?;
    }

    println!("✅ Gráfico guardado como {}", filename);

    // ⏱️ Tiempo total
    let elapsed_total = start_total.elapsed().as_secs_f64();
    println!("\n⏱️ Tiempo clustering: {:.3} segundos", elapsed_kmeans);
    println!("🔄 Ejecutando KMeans con k = {}...", k);
    println!("✅ Tiempo total de ejecución: {:.3} segundos", elapsed_total);

    Ok(())
}
