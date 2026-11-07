use std::collections::{HashMap, HashSet};
use std::io::{self, Write};

use crate::graph::{
    Graph, Node, NodeId, NodeStorage, NodeStorageGradView, NodeStorageParamView,
    NodeStorageValueView,
};
use crate::matrix::{
    Element, MatrixOwned, MatrixView, MatrixViewMut, MatrixViewRead, MatrixViewWrite,
};

/// Scalar types supported by the engine's fitting and prediction APIs.
pub trait FloatElement: Element + std::fmt::Display {}
impl FloatElement for f32 {}
impl FloatElement for f64 {}

pub type ParamsInitializer<K = f64> = Box<dyn FnMut(&mut MatrixViewMut<'_, K>)>;

pub struct FitCfg<K = f64> {
    pub n_epochs: u32,
    pub batch_size: u32,
    pub params_initializer: ParamsInitializer<K>,
    pub learning_rate: K,
}
pub struct Engine<K = f64> {
    graph: Graph,
    node_out: NodeId,
    node_cost: NodeId,
    verbose: u8,
    state: Option<FitState<K>>,
}

pub struct Data<'a, K = f64> {
    inputs: HashMap<NodeId, MatrixView<'a, K>>,
    n: usize,
}
impl<'a, K: FloatElement> Data<'a, K> {
    pub fn new(inputs: HashMap<NodeId, MatrixView<'a, K>>) -> Self {
        let n = inputs
            .values()
            .next()
            .map_or(0, |view| view.layout().rows());
        assert!(
            inputs.values().all(|view| view.layout().rows() == n),
            "all inputs must have the same row count"
        );
        Self { inputs, n }
    }
    pub fn from_input(node: NodeId, input: MatrixView<'a, K>) -> Self {
        let mut inputs = HashMap::new();
        inputs.insert(node, input);
        Self::new(inputs)
    }
    pub fn rows(&self) -> usize {
        self.n
    }
    pub fn slice(&self, start: usize, end: usize) -> Data<'a, K> {
        assert!(start <= end && end <= self.n, "data slice out of bounds");
        let inputs = self
            .inputs
            .iter()
            .map(|(&node, view)| (node, view.sub_rows(start..end)))
            .collect();
        Data {
            inputs,
            n: end - start,
        }
    }
}

struct Storage<K> {
    values: Vec<MatrixOwned<K>>,
    grads: Vec<MatrixOwned<K>>,
}

struct FitState<K> {
    storage: Storage<K>,
    positions: HashMap<NodeId, usize>,
    ops: HashSet<NodeId>,
    params: Vec<bool>,
    batch_size: usize,
}
impl<K: FloatElement> FitState<K> {
    fn new(graph: &Graph, batch_size: usize) -> Self {
        let mut values = Vec::new();
        let mut grads = Vec::new();
        let mut positions = HashMap::new();
        let mut ops = HashSet::new();
        let mut params = Vec::new();
        for (node_id, node) in graph.traverse(None) {
            let (rows, cols) = graph.shape_of(*node_id).rows_cols(batch_size);
            positions.insert(*node_id, values.len());
            values.push(MatrixOwned::zeros(rows, cols));
            grads.push(MatrixOwned::zeros(rows, cols));
            if matches!(node, Node::Operation(_)) {
                ops.insert(*node_id);
            }
            if matches!(node, Node::Param(_)) {
                params.push(true);
            } else {
                params.push(false);
            }
        }
        Self {
            storage: Storage { values, grads },
            positions,
            ops,
            params,
            batch_size,
        }
    }
    fn params_init(&mut self, mut initializer: ParamsInitializer<K>) {
        for (node_id, matrix) in self.storage.values.iter_mut().enumerate() {
            if self.params[node_id] {
                initializer(&mut matrix.as_view_mut());
            }
        }
    }
    fn clear_grads(&mut self) {
        for grad in self.storage.grads.iter_mut() {
            grad.as_view_mut().fill_zero();
        }
    }
    fn clear_ops(&mut self) {
        let ops: Vec<NodeId> = self.ops.iter().copied().collect();
        for node_id in ops {
            self.get_mut_val(node_id).fill_zero();
        }
    }
}

fn batch_indices(n: usize, b: usize) -> impl Iterator<Item = (usize, usize)> {
    assert!(b > 0, "batch size must be greater than zero");
    (0..n)
        .step_by(b)
        .filter_map(move |start| (start + b <= n).then_some((start, start + b)))
}

impl<K: FloatElement> NodeStorage for FitState<K> {
    type K = K;
    fn get_grad(&self, node_id: NodeId) -> MatrixView<'_, K> {
        self.storage.grads[self.positions[&node_id]].as_view()
    }
    fn get_mut_grad(&mut self, node_id: NodeId) -> MatrixViewMut<'_, K> {
        let position = self.positions[&node_id];
        self.storage.grads[position].as_view_mut()
    }
    fn get_val(&self, node_id: NodeId) -> MatrixView<'_, K> {
        self.storage.values[self.positions[&node_id]].as_view()
    }
    fn get_mut_val(&mut self, node_id: NodeId) -> MatrixViewMut<'_, K> {
        let position = self.positions[&node_id];
        self.storage.values[position].as_view_mut()
    }
    fn get_mut_val_and_grad(&mut self, node_id: NodeId) -> NodeStorageParamView<'_, K> {
        let position = self.positions[&node_id];
        let Storage { values, grads } = &mut self.storage;
        (values[position].as_view_mut(), grads[position].as_view())
    }
    fn get_val_output_and_inputs(
        &mut self,
        output: NodeId,
        input_ids: &[NodeId],
    ) -> NodeStorageValueView<'_, Self::K> {
        let output_index = self.positions[&output];
        assert!(
            input_ids
                .iter()
                .all(|node_id| self.positions[node_id] < output_index)
        );

        let (inputs_storage, output_storage) = self.storage.values.split_at_mut(output_index);
        let output = output_storage
            .first_mut()
            .expect("operation output must be allocated")
            .as_view_mut();
        let inputs = std::array::from_fn(|index| {
            input_ids
                .get(index)
                // We cannot use get_val because it borrows all of self
                .map(|node_id| inputs_storage[self.positions[node_id]].as_view())
        });
        (output, inputs)
    }

    fn get_grad_and_vals(
        &mut self,
        input_grad: NodeId,
        out_grad: NodeId,
        inputs: &[NodeId],
    ) -> NodeStorageGradView<'_, Self::K> {
        let (grad_inputs, grad_outputs) =
            self.storage.grads.split_at_mut(self.positions[&out_grad]);
        let grad_input = grad_inputs
            .get_mut(self.positions[&input_grad])
            .expect("gradient must be allocated")
            .as_view_mut();
        let grad_output = grad_outputs
            .first()
            .expect("gradient must be allocated")
            .as_view();
        // Inputs come from the values so there is no aliasing issue
        let inputs = std::array::from_fn(|index| {
            inputs
                .get(index)
                // We cannot use get_val because it borrows all of self
                .map(|node_id| self.storage.values[self.positions[node_id]].as_view())
        });
        (grad_input, grad_output, inputs)
    }
}

impl<K: FloatElement> Engine<K> {
    pub fn new(graph: Graph, node_out: NodeId, node_cost: NodeId, verbose: u8) -> Self {
        Self {
            graph,
            node_out,
            node_cost,
            verbose,
            state: None,
        }
    }
    fn forward_pass(&mut self, data: &Data<'_, K>, out_id: NodeId) {
        let expected_rows = self.state.as_ref().unwrap().batch_size;
        assert_eq!(
            data.n, expected_rows,
            "batch row count does not match engine state"
        );
        let state = self.state.as_mut().unwrap();
        for (node_id, node) in self.graph.traverse(Some(out_id)) {
            match node {
                Node::Operation(op) => op.apply(state, *node_id),
                Node::Input(_) => {
                    let input = data
                        .inputs
                        .get(node_id)
                        .unwrap_or_else(|| panic!("missing input for graph node"));
                    let mut target = state.get_mut_val(*node_id);
                    assert!(
                        target.same_shape(input),
                        "input shape does not match graph shape"
                    );
                    target.assign_with(input, input, |value, _| value);
                }
                Node::Param(_) => {}
            }
        }
    }
    fn backward_pass(&mut self, cost_id: NodeId) {
        // Iterate over the operations in reverse topological order
        let state = self.state.as_mut().unwrap();
        for (node_id, node) in self.graph.reverse(Some(cost_id)) {
            if let Node::Operation(op) = node {
                op.grad(state, *node_id);
            };
        }
    }

    fn apply_grad(&mut self, learning_rate: K) {
        let state = self.state.as_mut().unwrap();
        for (node_id, node) in self.graph.traverse(None) {
            if matches!(node, Node::Param(_)) {
                let (mut parameter, gradient) = state.get_mut_val_and_grad(*node_id);
                parameter.sub_assign_scaled(&gradient, learning_rate);
            }
        }
    }
    pub fn fit(&mut self, input: &Data<'_, K>, cfg: FitCfg<K>) {
        assert!(cfg.batch_size > 0, "batch size must be greater than zero");
        let batch_size = cfg.batch_size as usize;
        let batches_per_epoch = input.n / batch_size;
        let mut state = FitState::new(&self.graph, cfg.batch_size as usize);
        state.params_init(cfg.params_initializer);
        self.state = Some(state);
        for epoch in 0..cfg.n_epochs {
            for (start, end) in batch_indices(input.n, batch_size) {
                // Take only the data that corresponds to this batch
                let batch = input.slice(start, end);
                // Clear operations and gradients; initialize the cost gradient with 1
                {
                    let state = self.state.as_mut().unwrap();
                    state.clear_ops();
                    state.clear_grads();
                    state.get_mut_grad(self.node_cost).fill(Element::one());
                }
                // Perform the forward pass UP TO cost node -- fills operation buffers
                self.forward_pass(&batch, self.node_cost);
                let loss = self
                    .state
                    .as_ref()
                    .unwrap()
                    .get_val(self.node_cost)
                    .at(0, 0);
                if self.verbose > 0 {
                    print!(
                        "\r\x1b[2Kepoch {}/{} batch {}/{} loss {:.6}",
                        epoch + 1,
                        cfg.n_epochs,
                        start / batch_size + 1,
                        batches_per_epoch,
                        loss
                    );
                    io::stdout()
                        .flush()
                        .expect("failed to flush training progress");
                }
                // Perform the barckward pass FROM the cost node -- fills gradient buffers
                self.backward_pass(self.node_cost);
                // Change the params based on the gradients
                self.apply_grad(cfg.learning_rate);
            }
        }
        if self.verbose > 0 && cfg.n_epochs > 0 && batches_per_epoch > 0 {
            println!();
        }
    }
    pub fn predict(&mut self, input: &Data<'_, K>) -> Option<MatrixView<'_, K>> {
        self.state.as_ref()?;
        self.state.as_mut().unwrap().clear_ops();
        let node_out = self.node_out;
        self.forward_pass(input, node_out);
        Some(self.state.as_ref().unwrap().get_val(node_out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_batches_are_dropped() {
        assert_eq!(
            batch_indices(5, 2).collect::<Vec<_>>(),
            vec![(0, 2), (2, 4)]
        );
    }

    #[test]
    fn apply_grad_updates_parameters_only() {
        let mut graph = Graph::default();
        let parameter_id = graph.add_param(crate::graph::Shape::fixed(1, 2));
        let input_id = graph.add_input(crate::graph::Shape::fixed(1, 2));
        let mut engine = Engine::new(graph, input_id, input_id, 0);
        engine.state = Some(FitState::new(&engine.graph, 1));

        let state = engine.state.as_mut().unwrap();
        {
            let mut parameter = state.get_mut_val(parameter_id);
            *parameter.get_mut(0, 0) = 3.0;
            *parameter.get_mut(0, 1) = -2.0;
        }
        {
            let mut gradient = state.get_mut_grad(parameter_id);
            *gradient.get_mut(0, 0) = 0.5;
            *gradient.get_mut(0, 1) = -1.0;
        }
        {
            let mut input = state.get_mut_val(input_id);
            *input.get_mut(0, 0) = 7.0;
            *input.get_mut(0, 1) = 8.0;
        }
        {
            let mut gradient = state.get_mut_grad(input_id);
            *gradient.get_mut(0, 0) = 4.0;
            *gradient.get_mut(0, 1) = 5.0;
        }

        engine.apply_grad(0.1);

        let state = engine.state.as_ref().unwrap();
        let parameter = state.get_val(parameter_id);
        assert_eq!(parameter.at(0, 0), 2.95);
        assert_eq!(parameter.at(0, 1), -1.9);
        let input = state.get_val(input_id);
        assert_eq!(input.at(0, 0), 7.0);
        assert_eq!(input.at(0, 1), 8.0);
    }

    #[test]
    fn forward_pass_matches_matrix_operations() {
        let mut graph = Graph::default();
        let input_id = graph.add_input(crate::graph::Shape::batch_rows(2));
        let weights_id = graph.add_param(crate::graph::Shape::fixed(2, 2));
        let matmul_id = graph.add_op(crate::graph::Operation::MatMul(input_id, weights_id));
        let bias_id = graph.add_input(crate::graph::Shape::batch_rows(2));
        let output_id = graph.add_op(crate::graph::Operation::Add(matmul_id, bias_id));
        let cost_id = graph.add_op(crate::graph::Operation::NormL1(output_id, 0.0));

        let input = MatrixOwned::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let bias = MatrixOwned::from_vec(vec![0.5, -1.0, 1.5, 2.0], 2, 2);
        let mut data_inputs = HashMap::new();
        data_inputs.insert(input_id, input.as_view());
        data_inputs.insert(bias_id, bias.as_view());
        let data = Data::new(data_inputs);

        let weights = MatrixOwned::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let mut expected = MatrixOwned::zeros(2, 2);
        expected
            .as_view_mut()
            .matmul_assign(&input.as_view(), &weights.as_view());
        expected.as_view_mut().add_assign(&bias.as_view());

        let params_initializer: ParamsInitializer = Box::new(move |params| {
            params.assign_with(&weights.as_view(), &weights.as_view(), |value, _| value);
        });
        let mut engine = Engine::new(graph, output_id, cost_id, 0);
        engine.fit(
            &data,
            FitCfg {
                n_epochs: 0,
                batch_size: 2,
                params_initializer,
                learning_rate: 0.0,
            },
        );
        let actual = engine.predict(&data).expect("engine was fitted");

        assert!(expected.as_view().eq_matrix(&actual));
    }

    fn fit_predict_float<K: FloatElement>() {
        let mut graph = Graph::default();
        let input_id = graph.add_input(crate::graph::Shape::batch_rows(1));
        let weight_id = graph.add_param(crate::graph::Shape::fixed(1, 1));
        let prediction_id = graph.add_op(crate::graph::Operation::MatMul(input_id, weight_id));
        let target_id = graph.add_input(crate::graph::Shape::batch_rows(1));
        let cost_id = graph.add_op(crate::graph::Operation::LossL2(prediction_id, target_id));

        let input = MatrixOwned::full(2, 1, K::from_f64(3.0));
        let target = MatrixOwned::full(2, 1, K::from_f64(6.0));
        let mut inputs = HashMap::new();
        inputs.insert(input_id, input.as_view());
        inputs.insert(target_id, target.as_view());
        let data = Data::new(inputs);
        let mut engine = Engine::new(graph, prediction_id, cost_id, 0);
        engine.fit(
            &data,
            FitCfg {
                n_epochs: 1,
                batch_size: 2,
                params_initializer: Box::new(|params| params.fill(K::one())),
                learning_rate: K::from_f64(0.1),
            },
        );
        let prediction = engine.predict(&data).expect("engine was fitted");
        assert!((prediction.at(0, 0).to_f64() - 8.4).abs() < 1e-5);
    }

    #[test]
    fn fit_and_predict_support_f32_and_f64() {
        fit_predict_float::<f32>();
        fit_predict_float::<f64>();
    }

    #[test]
    fn relu_forward_and_backward_handle_zero() {
        let mut graph = Graph::default();
        let input = graph.add_input(crate::graph::Shape::fixed(1, 4));
        let output = graph.add_op(crate::graph::Operation::ReLU(input));
        let mut state = FitState::new(&graph, 1);
        {
            let mut values = state.get_mut_val(input);
            for (column, value) in [-2.0, 0.0, 3.0, 5.0].into_iter().enumerate() {
                *values.get_mut(0, column) = value;
            }
        }
        let crate::graph::Node::Operation(operation) =
            graph.traverse(Some(output)).last().unwrap().1
        else {
            unreachable!()
        };
        operation.apply(&mut state, output);
        assert_eq!(
            (0..4)
                .map(|column| state.get_val(output).at(0, column))
                .collect::<Vec<_>>(),
            vec![0.0, 0.0, 3.0, 5.0]
        );

        state.get_mut_grad(output).fill(1.0);
        operation.grad(&mut state, output);
        assert_eq!(
            (0..4)
                .map(|column| state.get_grad(input).at(0, column))
                .collect::<Vec<_>>(),
            vec![0.0, 0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn cross_entropy_forward_uses_one_hot_labels() {
        let mut graph = Graph::default();
        let labels = graph.add_input(crate::graph::Shape::batch_rows(3));
        let logits = graph.add_input(crate::graph::Shape::batch_rows(3));
        let cost = graph.add_op(crate::graph::Operation::LossCrossEntropySoftMax(
            labels, logits,
        ));
        assert_eq!(graph.shape_of(cost), crate::graph::Shape::fixed(1, 1));

        let mut state = FitState::new(&graph, 2);
        {
            let mut label_values = state.get_mut_val(labels);
            *label_values.get_mut(0, 0) = 1.0;
            *label_values.get_mut(0, 1) = 0.0;
            *label_values.get_mut(0, 2) = 0.0;
            *label_values.get_mut(1, 0) = 0.0;
            *label_values.get_mut(1, 1) = 1.0;
            *label_values.get_mut(1, 2) = 0.0;
            let mut logit_values = state.get_mut_val(logits);
            for j in 0..3 {
                *logit_values.get_mut(0, j) = 0.0;
            }
            *logit_values.get_mut(1, 0) = 2.0;
            *logit_values.get_mut(1, 1) = 1.0;
            *logit_values.get_mut(1, 2) = 0.0;
        }

        let crate::graph::Node::Operation(operation) = graph.traverse(Some(cost)).last().unwrap().1
        else {
            unreachable!()
        };
        operation.apply(&mut state, cost);
        let expected_loss = (3.0_f64.ln() + (2.0_f64.exp() + 1.0_f64.exp() + 1.0).ln() - 1.0) / 2.0;
        assert!((state.get_val(cost).at(0, 0) - expected_loss).abs() < 1e-12);
    }
}
