use cadeia::engine::{Data, Engine, FitCfg};
use cadeia::factory::MlpCfg;
use cadeia::matrix::{MatrixOwned, MatrixViewRead, MatrixViewWrite};

use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;
const IMAGE_PATH: &str = "data/data/mnist/X_train.npy";
const LABEL_PATH: &str = "data/data/mnist/y_train.npy";
const TEST_IMAGE_PATH: &str = "data/data/mnist/X_test.npy";
const TEST_LABEL_PATH: &str = "data/data/mnist/y_test.npy";

struct NpyHeader {
    descr: String,
    shape: Vec<usize>,
    data_offset: u64,
}

fn read_header(file: &mut File) -> io::Result<NpyHeader> {
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    if &prefix[..6] != b"\x93NUMPY" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a NumPy .npy file",
        ));
    }
    let (header_len, header_prefix) = match prefix[6] {
        1 => {
            let mut len = [0; 2];
            file.read_exact(&mut len)?;
            (u16::from_le_bytes(len) as usize, 10u64)
        }
        2 | 3 => {
            let mut len = [0; 4];
            file.read_exact(&mut len)?;
            (u32::from_le_bytes(len) as usize, 12u64)
        }
        version => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported .npy version {version}"),
            ));
        }
    };
    let mut header_bytes = vec![0; header_len];
    file.read_exact(&mut header_bytes)?;
    let header = String::from_utf8(header_bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 in .npy header"))?;
    let descr = header_value(&header, "descr")?;
    let shape_start = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing shape in .npy header")
        })?;
    let open = header[shape_start..]
        .find('(')
        .map(|i| i + shape_start)
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "invalid shape in .npy header")
        })?;
    let close = header[open..].find(')').map(|i| i + open).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "invalid shape in .npy header")
    })?;
    let shape = header[open + 1..close]
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect();
    Ok(NpyHeader {
        descr,
        shape,
        data_offset: header_prefix + header_len as u64,
    })
}

fn header_value(header: &str, key: &str) -> io::Result<String> {
    let marker = format!("'{key}'");
    let start = header
        .find(&marker)
        .or_else(|| header.find(&format!("\"{key}\"")))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("missing {key} in .npy header"),
            )
        })?;
    let colon = header[start..]
        .find(':')
        .map(|i| start + i)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid .npy header"))?;
    let rest = header[colon + 1..].trim_start();
    let quote = rest
        .chars()
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid dtype"))?;
    if quote != '\'' && quote != '"' {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid dtype in .npy header",
        ));
    }
    let end = rest[1..].find(quote).map(|i| i + 1).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "invalid dtype in .npy header")
    })?;
    Ok(rest[1..end].to_owned())
}

fn read_prefix(
    path: &Path,
    count: usize,
    expected_shape_tail: &[usize],
    kind: char,
) -> Result<(NpyHeader, Vec<u8>), Box<dyn Error>> {
    let mut file = File::open(path)?;
    let header = read_header(&mut file)?;
    if header.shape.len() != expected_shape_tail.len() + 1
        || header.shape[1..] != *expected_shape_tail
    {
        return Err(format!("{} has unexpected shape {:?}", path.display(), header.shape).into());
    }
    if count > header.shape[0] {
        return Err(format!(
            "requested {count} samples, but {} contains only {}",
            path.display(),
            header.shape[0]
        )
        .into());
    }
    let dtype = header.descr.as_str();
    if dtype.len() < 3
        || !matches!(dtype.as_bytes()[0], b'<' | b'|' | b'=')
        || dtype.as_bytes()[1] != kind as u8
    {
        return Err(format!("{} has unsupported dtype {}", path.display(), header.descr).into());
    }
    let item_size: usize = dtype[2..].parse()?;
    if (kind == 'f' && item_size != 4) || (kind == 'u' && !matches!(item_size, 1 | 2 | 4 | 8)) {
        return Err(format!("{} has unsupported dtype {}", path.display(), header.descr).into());
    }
    let item_count = count
        .checked_mul(expected_shape_tail.iter().product::<usize>())
        .ok_or("sample count overflow")?;
    let byte_count = item_count
        .checked_mul(item_size)
        .ok_or("data size overflow")?;
    file.seek(SeekFrom::Start(header.data_offset))?;
    let mut data = vec![0; byte_count];
    file.read_exact(&mut data)?;
    Ok((header, data))
}

fn label_at(bytes: &[u8], n_classes: usize) -> usize {
    (0..n_classes)
        .max_by(|&lhs, &rhs| {
            let lhs = f32::from_le_bytes(bytes[lhs * 4..lhs * 4 + 4].try_into().unwrap());
            let rhs = f32::from_le_bytes(bytes[rhs * 4..rhs * 4 + 4].try_into().unwrap());
            lhs.total_cmp(&rhs)
        })
        .unwrap()
}

fn f32_le_values(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
}

fn argmax_row(values: &impl MatrixViewRead<K = f32>, row: usize) -> usize {
    (0..values.layout().cols())
        .max_by(|&lhs, &rhs| values.at(row, lhs).total_cmp(&values.at(row, rhs)))
        .unwrap()
}

fn softmax_probability(values: &impl MatrixViewRead<K = f32>, row: usize, class: usize) -> f32 {
    let max_logit = (0..values.layout().cols())
        .map(|col| values.at(row, col))
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = (0..values.layout().cols())
        .map(|col| (values.at(row, col) - max_logit).exp())
        .sum();
    (values.at(row, class) - max_logit).exp() / exp_sum
}

fn shade(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn render(image: &[u8]) {
    for row in 0..14 {
        for col in 0..28 {
            let top = shade(f32::from_le_bytes(
                image[((row * 2) * 28 + col) * 4..][..4].try_into().unwrap(),
            ));
            let bottom = if row * 2 + 1 < 28 {
                shade(f32::from_le_bytes(
                    image[((row * 2 + 1) * 28 + col) * 4..][..4]
                        .try_into()
                        .unwrap(),
                ))
            } else {
                0
            };
            print!("\x1b[38;2;{top};{top};{top}m\x1b[48;2;{bottom};{bottom};{bottom}m▀");
        }
        println!("\x1b[0m");
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    const BATCH_SIZE: u32 = 60;
    const DEFAULT_SAMPLE: usize = 60000;
    const N_INPUT: usize = 28 * 28;
    const N_OUTPUT: usize = 10;
    const FULL_TRAIN_COUNT: usize = 60_000;
    const FULL_TEST_COUNT: usize = 10_000;
    const N_EPOCHS: u32 = 10;
    let hidden_layers = vec![50, 50];

    let count = match env::args().nth(1) {
        Some(value) => {
            let parsed: usize = value
                .parse()
                .map_err(|_| "sample count must be a positive integer")?;
            if parsed < BATCH_SIZE as usize {
                return Err(
                    format!("sample count must be at least the batch size {BATCH_SIZE}").into(),
                );
            }
            parsed
        }
        None => DEFAULT_SAMPLE,
    };
    let (images_header, images) = read_prefix(Path::new(IMAGE_PATH), count, &[28, 28], 'f')?;
    let (labels_header, labels) = read_prefix(Path::new(LABEL_PATH), count, &[10], 'f')?;
    if images_header.shape[0] != labels_header.shape[0] {
        return Err("image and label datasets have different lengths".into());
    }
    println!("Loaded {count} MNIST samples.");
    render(&images[..N_INPUT * 4]);
    println!("Label: {}", label_at(&labels, labels_header.shape[1]));

    let model = MlpCfg {
        n_inputs: N_INPUT,
        n_output: N_OUTPUT,
        hidden_layers,
    }
    .build();
    let mut engine = Engine::new(model.graph, model.prediction, model.cost, 1);
    let mut rng = StdRng::seed_from_u64(7);

    let image_matrix = MatrixOwned::from_vec(f32_le_values(&images).collect(), count, N_INPUT);
    let labels_matrix = MatrixOwned::from_vec(f32_le_values(&labels).collect(), count, N_OUTPUT);
    drop(images);
    drop(labels);
    let mut training_inputs = HashMap::new();
    training_inputs.insert(model.input, image_matrix.as_view());
    training_inputs.insert(model.target, labels_matrix.as_view());
    let training_data = Data::new(training_inputs);
    let tic = Instant::now();
    engine.fit(
        &training_data,
        FitCfg {
            n_epochs: N_EPOCHS,
            batch_size: BATCH_SIZE,
            params_initializer: Box::new(move |params| {
                // Good heurisitc
                let nrows = params.layout().rows();
                if nrows == 1 {
                    params.fill_zero();
                } else {
                    let aux = (6.0_f32 / nrows as f32).sqrt();
                    params.fill_with(|| rng.random_range(-aux..aux));
                }
            }),
            learning_rate: 0.01,
        },
    );
    let fit_time = Instant::now() - tic;
    drop(training_data);
    drop(image_matrix);
    drop(labels_matrix);

    let test_count = FULL_TEST_COUNT * count / FULL_TRAIN_COUNT;
    let (test_images_header, test_images) =
        read_prefix(Path::new(TEST_IMAGE_PATH), test_count, &[28, 28], 'f')?;
    let (test_labels_header, test_labels) =
        read_prefix(Path::new(TEST_LABEL_PATH), test_count, &[N_OUTPUT], 'f')?;
    if test_images_header.shape[0] != test_labels_header.shape[0] {
        return Err("test image and label datasets have different lengths".into());
    }
    let test_image_matrix =
        MatrixOwned::from_vec(f32_le_values(&test_images).collect(), test_count, N_INPUT);
    let test_label_matrix =
        MatrixOwned::from_vec(f32_le_values(&test_labels).collect(), test_count, N_OUTPUT);
    let test_images_view = test_image_matrix.as_view();
    let test_labels_view = test_label_matrix.as_view();

    let batch_size = BATCH_SIZE as usize;
    let mut correct = 0;
    let mut sample_predictions = Vec::new();
    for start in (0..test_count).step_by(batch_size) {
        let rows = (test_count - start).min(batch_size);
        let mut batch_images = MatrixOwned::zeros(batch_size, N_INPUT);
        {
            let mut batch_view = batch_images.as_view_mut();
            for row in 0..rows {
                for col in 0..N_INPUT {
                    *batch_view.get_mut(row, col) = test_images_view.at(start + row, col);
                }
            }
        }
        let test_batch = Data::from_input(model.input, batch_images.as_view());
        let predictions = engine.predict(&test_batch).unwrap();
        for row in 0..rows {
            let predicted = argmax_row(&predictions, row);
            let actual = argmax_row(&test_labels_view, start + row);
            correct += usize::from(predicted == actual);

            let sample = start + row;
            if sample < 3 {
                let probability = softmax_probability(&predictions, row, predicted);
                sample_predictions.push((sample, predicted, probability, actual));
            }
        }
        print!(
            "\r\x1b[2Kevaluating test set: {}/{}",
            start + rows,
            test_count
        );
        io::stdout().flush()?;
    }
    println!();
    for (sample, predicted, probability, actual) in sample_predictions {
        println!(
            "Test image {sample}: predicted {predicted} (probability {probability:.4}), actual {actual}"
        );
        let image_start = sample * N_INPUT * 4;
        render(&test_images[image_start..image_start + N_INPUT * 4]);
    }
    println!("Fit time {:?}", fit_time);
    println!(
        "Test accuracy: {correct}/{test_count} ({:.2}%)",
        100.0 * correct as f64 / test_count as f64
    );

    Ok(())
}
