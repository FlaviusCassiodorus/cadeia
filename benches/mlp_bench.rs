use std::collections::HashMap;

use cadeia::engine::{Data, Engine, FitCfg, ParamsInitializer};
use cadeia::factory::MlpCfg;
use cadeia::matrix::{MatrixOwned, MatrixViewRead, MatrixViewWrite};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_fit(c: &mut Criterion) {
    const BATCH: usize = 16;
    let model = MlpCfg {
        n_inputs: 784,
        n_output: 10,
        hidden_layers: vec![64, 32],
    }
    .build();
    let inputs = MatrixOwned::full(BATCH, 784, 0.01_f32);
    let targets = MatrixOwned::full(BATCH, 10, 0.1_f32);
    let mut data = HashMap::new();
    data.insert(model.input, inputs.as_view());
    data.insert(model.target, targets.as_view());
    let data = Data::new(data);

    let mut engine = Engine::new(model.graph, model.prediction, model.cost, 0);
    c.bench_function("mlp/fit_batch_16_784-64-32-10_f32", |b| {
        b.iter(|| {
            let initializer: ParamsInitializer<f32> = Box::new(|params| {
                let (rows, cols) = (params.layout().rows(), params.layout().cols());
                for row in 0..rows {
                    for col in 0..cols {
                        *params.get_mut(row, col) = ((row * 17 + col * 13) % 101) as f32 / 1000.0;
                    }
                }
            });
            engine.fit(
                &data,
                FitCfg {
                    n_epochs: 1,
                    batch_size: BATCH as u32,
                    params_initializer: initializer,
                    learning_rate: 0.01_f32,
                },
            );
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5))
        .sample_size(30);
    targets = bench_fit
);
criterion_main!(benches);
