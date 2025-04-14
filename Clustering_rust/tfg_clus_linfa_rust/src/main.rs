use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::{Array1, Array2};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader};
use std::time::Instant;
use csv::ReaderBuilder;
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("C:/Users/User/OneDrive/Escritorio/TFG/csv/housing.csv")?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(BufReader::new(file));

    let mut features = Vec::new();
    let mut count = 0;

    for result in rdr.records() {
        let record = result?;
        if record.len() != 10 || record.iter().any(|v| v.trim().is_empty()) {
            continue;
        }

        let row: Result<Vec<f64>, _> = (0..2).map(|i| record[i].parse::<f64>()).collect();
        match row {
            Ok(r) => {
                features.extend(r);
                count += 1;
            }
            Err(_) => continue,
        }
    }

    // ▶️ Crear matriz con n variables
    let x = Array2::from_shape_vec((count, 2), features)?;
    let dummy_labels = Array1::<usize>::zeros(x.nrows());
    let dataset = Dataset::new(x.clone(), dummy_labels);

    // ▶️ Clustering
    let k = 3;
    println!("🔄 Ejecutando KMeans con k={}...",k);
    let start = Instant::now();
    let model = KMeans::params(k).max_n_iterations(100).fit(&dataset)?;
    let elapsed = start.elapsed().as_secs_f64();
    let labels = model.predict(&x);

    println!("✅ Clustering completado en {:.2} segundos", elapsed);

    let x_vals = x.column(0);
    let y_vals = x.column(1);
    let num_vars = x.ncols();

    let title = format!("KMeans Clustering (k={}, {} variables)", k, num_vars);

    let x_min = x_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let filename = format!("linfa_k{}_n{}.png", k, num_vars);
    let root = BitMapBackend::new(&filename, (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 25))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    for i in 0..x.nrows() {
        let x_val = x[[i, 0]];
        let y_val = x[[i, 1]];
        let label = labels[i];

        chart.draw_series(PointSeries::of_element(
            vec![(x_val, y_val)],
            3,
            &Palette99::pick(label),
            &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
        ))?;
    }

    Ok(())
}
