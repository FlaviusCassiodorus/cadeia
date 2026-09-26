use std::collections::HashMap;

use rand::{Rng, SeedableRng, rngs::StdRng};

use cadeia::engine::{Data, Engine, FitCfg};
use cadeia::graph::{Graph, NodeId};
use cadeia::matrix::{MatrixOwned, MatrixViewRead, MatrixViewWrite};

pub struct LassoCfg {
    pub lambda: f64,
    pub has_intercept: bool,
    pub n_features: usize,
}
impl Default for LassoCfg {
    fn default() -> Self {
        Self {
            lambda: 0.05,
            has_intercept: true,
            n_features: 3,
        }
    }
}

impl LassoCfg {
    fn build(&self) -> (Graph, NodeId, NodeId, NodeId, NodeId) {
        let mut graph = Graph::default();
        let input = graph.add_input(cadeia::graph::Shape::batch_rows(self.n_features));
        let weights = graph.add_param(cadeia::graph::Shape::fixed(self.n_features, 1));
        let prediction = graph.add_op(cadeia::graph::Operation::MatMul(input, weights));
        let prediction = if self.has_intercept {
            let bias = graph.add_param(cadeia::graph::Shape::fixed(1, 1));
            graph.add_op(cadeia::graph::Operation::Add(prediction, bias))
        } else {
            prediction
        };
        let target = graph.add_input(cadeia::graph::Shape::batch_rows(1));
        let fit_loss = graph.add_op(cadeia::graph::Operation::LossL2(prediction, target));
        let penalty = graph.add_op(cadeia::graph::Operation::L1(weights, self.lambda));
        let cost = graph.add_op(cadeia::graph::Operation::Add(fit_loss, penalty));
        (graph, input, target, prediction, cost)
    }
}

const TRUE_WEIGHTS: [f64; 3] = [1.5, 0.0, -2.0];
const TRUE_INTERCEPT: f64 = 0.75;
const N_SAMPLES: usize = 8;

fn format_coefficients(coefficients: &[f64]) -> String {
    let coefficients = coefficients
        .iter()
        .map(|coefficient| format!("{coefficient:.3}"))
        .collect::<Vec<_>>();
    format!("[{}]", coefficients.join(", "))
}

fn make_data(feature_scale: f64) -> (MatrixOwned<f64>, MatrixOwned<f64>) {
    let mut inputs = MatrixOwned::zeros(N_SAMPLES, TRUE_WEIGHTS.len());
    let mut targets = MatrixOwned::zeros(N_SAMPLES, 1);
    let mut inputs_view = inputs.as_view_mut();
    let mut targets_view = targets.as_view_mut();

    for row in 0..N_SAMPLES {
        let mut target = TRUE_INTERCEPT;
        let mut interaction = 1.0;
        for (feature, weight) in TRUE_WEIGHTS.iter().enumerate() {
            let sign = if row & (1 << feature) == 0 { -1.0 } else { 1.0 };
            let value = sign * feature_scale;
            *inputs_view.get_mut(row, feature) = value;
            target += value * weight;
            interaction *= value;
        }
        target += 0.02 * interaction;
        *targets_view.get_mut(row, 0) = target;
    }

    (inputs, targets)
}

fn main() {
    let cfg = LassoCfg::default();
    let (graph, input_node, target_node, prediction_node, cost_node) = cfg.build();
    let mut engine = Engine::new(graph, prediction_node, cost_node);
    let mut rng = StdRng::seed_from_u64(7);

    // The generated rule uses features 0 and 2; feature 1 is irrelevant.
    let (training_input, training_target) = make_data(1.0);
    let mut training_inputs = HashMap::new();
    training_inputs.insert(input_node, training_input.as_view());
    training_inputs.insert(target_node, training_target.as_view());
    let training_data = Data::new(training_inputs);
    let (prediction_input, expected_target) = make_data(0.5);

    engine.fit(
        &training_data,
        FitCfg {
            n_epochs: 100,
            batch_size: N_SAMPLES as u32,
            params_initializer: Box::new(move |params| {
                params.fill_with(|| rng.random_range(-1.0..1.0));
            }),
            learning_rate: 0.05,
        },
    );

    let mut coefficient_input = MatrixOwned::zeros(N_SAMPLES, TRUE_WEIGHTS.len());
    for feature in 0..TRUE_WEIGHTS.len() {
        *coefficient_input
            .as_view_mut()
            .get_mut(feature + 1, feature) = 1.0;
    }
    let coefficient_data = Data::from_input(input_node, coefficient_input.as_view());
    let coefficient_predictions = engine
        .predict(&coefficient_data)
        .expect("engine was fitted");
    let learned_intercept = coefficient_predictions.at(0, 0);
    let learned_weights: Vec<_> = (0..TRUE_WEIGHTS.len())
        .map(|feature| coefficient_predictions.at(feature + 1, 0) - learned_intercept)
        .collect();
    println!("True intercept: {TRUE_INTERCEPT:.3}");
    println!("Learned intercept: {learned_intercept:.3}");
    println!("True coefficients: {}", format_coefficients(&TRUE_WEIGHTS));
    println!(
        "Learned coefficients: {}",
        format_coefficients(&learned_weights)
    );

    let prediction_data = Data::from_input(input_node, prediction_input.as_view());
    let prediction = engine.predict(&prediction_data).expect("engine was fitted");
    println!("Target rule: y = 0.75 + 1.5*x0 + 0.0*x1 - 2.0*x2 + small noise");
    println!("First four prediction rows:");
    for row in 0..4 {
        println!(
            "  row {row}: target = {:.4}, prediction = {:.4}",
            expected_target.as_view().at(row, 0),
            prediction.at(row, 0)
        );
    }
}
