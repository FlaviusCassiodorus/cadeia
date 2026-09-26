use std::collections::{HashMap, HashSet};

use crate::graph::{
    Graph, Node, NodeId, NodeStorage, NodeStorageGradView, NodeStorageParamView,
    NodeStorageValueView,
};
use crate::matrix::{
    Element, MatrixOwned, MatrixView, MatrixViewMut, MatrixViewRead, MatrixViewWrite,
};

pub type ParamsInitializer = Box<dyn FnMut(&mut MatrixViewMut<'_, f64>)>;

pub struct FitCfg {
    pub n_epochs: u32,
    pub batch_size: u32,
    pub params_initializer: ParamsInitializer,
    pub learning_rate: f64,
}
pub struct Engine {
    graph: Graph,
    node_out: NodeId,
    node_cost: NodeId,
    state: Option<FitState>,
}

pub struct Data<'a> {
    inputs: HashMap<NodeId, MatrixView<'a, f64>>,
    n: usize,
}
impl<'a> Data<'a> {
    pub fn new(inputs: HashMap<NodeId, MatrixView<'a, f64>>) -> Self {
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
    pub fn from_input(node: NodeId, input: MatrixView<'a, f64>) -> Self {
        let mut inputs = HashMap::new();
        inputs.insert(node, input);
        Self::new(inputs)
    }
    pub fn rows(&self) -> usize {
        self.n
    }
    pub fn slice(&self, start: usize, end: usize) -> Data<'a> {
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

struct Storage {
    values: Vec<MatrixOwned<f64>>,
    grads: Vec<MatrixOwned<f64>>,
}

struct FitState {
    storage: Storage,
    positions: HashMap<NodeId, usize>,
    ops: HashSet<NodeId>,
    params: Vec<bool>,
    batch_size: usize,
}
impl FitState {
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
    fn params_init(&mut self, mut initializer: ParamsInitializer) {
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

impl NodeStorage for FitState {
    type K = f64;
    fn get_grad(&self, node_id: NodeId) -> MatrixView<'_, f64> {
        self.storage.grads[self.positions[&node_id]].as_view()
    }
    fn get_mut_grad(&mut self, node_id: NodeId) -> MatrixViewMut<'_, f64> {
        let position = self.positions[&node_id];
        self.storage.grads[position].as_view_mut()
    }
    fn get_val(&self, node_id: NodeId) -> MatrixView<'_, f64> {
        self.storage.values[self.positions[&node_id]].as_view()
    }
    fn get_mut_val(&mut self, node_id: NodeId) -> MatrixViewMut<'_, f64> {
        let position = self.positions[&node_id];
        self.storage.values[position].as_view_mut()
    }
    fn get_mut_val_and_grad(&mut self, node_id: NodeId) -> NodeStorageParamView<'_, f64> {
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

impl Engine {
    pub fn new(graph: Graph, node_out: NodeId, node_cost: NodeId) -> Self {
        Self {
            graph,
            node_out,
            node_cost,
            state: None,
        }
    }
    fn forward_pass(&mut self, data: &Data<'_>, out_id: NodeId) {
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

    fn apply_grad(&mut self, learning_rate: f64) {
        let state = self.state.as_mut().unwrap();
        for (node_id, node) in self.graph.traverse(None) {
            if matches!(node, Node::Param(_)) {
                let (mut parameter, gradient) = state.get_mut_val_and_grad(*node_id);
                parameter.sub_assign_scaled(&gradient, learning_rate);
            }
        }
    }
    pub fn fit(&mut self, input: &Data<'_>, cfg: FitCfg) {
        assert!(cfg.batch_size > 0, "batch size must be greater than zero");
        let mut state = FitState::new(&self.graph, cfg.batch_size as usize);
        state.params_init(cfg.params_initializer);
        self.state = Some(state);
        for _ in 0..cfg.n_epochs {
            for (start, end) in batch_indices(input.n, cfg.batch_size as usize) {
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
                // Perform the barckward pass FROM the cost node -- fills gradient buffers
                self.backward_pass(self.node_cost);
                // Change the params based on the gradients
                self.apply_grad(cfg.learning_rate);
            }
        }
    }
    pub fn predict(&mut self, input: &Data<'_>) -> Option<MatrixView<'_, f64>> {
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
        let mut engine = Engine::new(graph, input_id, input_id);
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
        let cost_id = graph.add_op(crate::graph::Operation::L1(output_id, 0.0));

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
        let mut engine = Engine::new(graph, output_id, cost_id);
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
}
