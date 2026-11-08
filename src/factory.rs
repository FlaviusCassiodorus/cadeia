use crate::graph::{Graph, NodeId, Operation, Shape};

/// The graph and standard node handles produced by an architecture factory.
pub struct ModelFactoryResult {
    pub graph: Graph,
    pub input: NodeId,
    pub target: NodeId,
    pub prediction: NodeId,
    pub cost: NodeId,
}

/// Builds a graph architecture from its configuration.
pub trait ModelFactory {
    fn build(&self) -> ModelFactoryResult;
}

#[derive(Clone, Copy, Debug)]
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

// So that we dont have the bring the trait into scope to use this
impl LassoCfg {
    pub fn build(&self) -> ModelFactoryResult {
        <Self as ModelFactory>::build(self)
    }
}

impl ModelFactory for LassoCfg {
    fn build(&self) -> ModelFactoryResult {
        let mut graph = Graph::default();
        let input = graph.add_input(Shape::batch_rows(self.n_features));
        let weights = graph.add_param(Shape::fixed(self.n_features, 1));
        let product = graph.add_op(Operation::MatMul(input, weights));
        let prediction = if self.has_intercept {
            let bias = graph.add_param(Shape::fixed(1, 1));
            graph.add_op(Operation::Add(product, bias))
        } else {
            product
        };
        let target = graph.add_input(Shape::batch_rows(1));
        let fit_loss = graph.add_op(Operation::LossL2(prediction, target));
        let penalty = graph.add_op(Operation::NormL1(weights, self.lambda));
        let cost = graph.add_op(Operation::Add(fit_loss, penalty));
        ModelFactoryResult {
            graph,
            input,
            target,
            prediction,
            cost,
        }
    }
}

/// Configuration for a fully connected network with batch-row inputs and outputs.
#[derive(Clone, Debug)]
pub struct MlpCfg {
    pub n_inputs: usize,
    pub n_output: usize,
    pub hidden_layers: Vec<usize>,
}

impl MlpCfg {
    pub fn build(&self) -> ModelFactoryResult {
        <Self as ModelFactory>::build(self)
    }
}

impl ModelFactory for MlpCfg {
    fn build(&self) -> ModelFactoryResult {
        let mut graph = Graph::default();
        let input = graph.add_input(Shape::batch_rows(self.n_inputs));
        let mut activation = input;
        let mut width = self.n_inputs;

        for &next_width in &self.hidden_layers {
            let weights = graph.add_param(Shape::fixed(width, next_width));
            let bias = graph.add_param(Shape::fixed(1, next_width));
            let product = graph.add_op(Operation::MatMul(activation, weights));
            let affine = graph.add_op(Operation::Add(product, bias));
            activation = graph.add_op(Operation::ReLU(affine));
            width = next_width;
        }

        let weights = graph.add_param(Shape::fixed(width, self.n_output));
        let bias = graph.add_param(Shape::fixed(1, self.n_output));
        let product = graph.add_op(Operation::MatMul(activation, weights));
        let prediction = graph.add_op(Operation::Add(product, bias));
        let target = graph.add_input(Shape::batch_rows(self.n_output));
        let cost = graph.add_op(Operation::LossCrossEntropySoftMax(target, prediction));

        ModelFactoryResult {
            graph,
            input,
            target,
            prediction,
            cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_shapes_cover_hidden_and_direct_networks() {
        let model = MlpCfg {
            n_inputs: 4,
            n_output: 2,
            hidden_layers: vec![5, 3],
        }
        .build();
        assert_eq!(model.graph.shape_of(model.prediction).rows_cols(7), (7, 2));
        assert_eq!(model.graph.shape_of(model.target).rows_cols(7), (7, 2));
        assert_eq!(model.graph.shape_of(model.cost), Shape::fixed(1, 1));

        let model = MlpCfg {
            n_inputs: 4,
            n_output: 2,
            hidden_layers: vec![],
        }
        .build();
        assert_eq!(model.graph.shape_of(model.prediction).rows_cols(7), (7, 2));
    }

    #[test]
    fn lasso_factory_preserves_default_graph() {
        let model = LassoCfg::default().build();
        assert_eq!(model.graph.shape_of(model.prediction).rows_cols(8), (8, 1));
        assert_eq!(model.graph.shape_of(model.target).rows_cols(8), (8, 1));
        assert_eq!(model.graph.shape_of(model.cost), Shape::fixed(1, 1));
    }
}
