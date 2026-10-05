use super::ops::Operation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    BATCH,
    N(usize),
}
impl Dim {
    fn concrete(&self, batch_size: usize) -> usize {
        match *self {
            Self::BATCH => batch_size,
            Self::N(n) => n,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub(super) rows: Dim,
    pub(super) cols: Dim,
}

impl Shape {
    pub fn fixed(rows: usize, cols: usize) -> Self {
        Self {
            rows: Dim::N(rows),
            cols: Dim::N(cols),
        }
    }
    pub fn batch_rows(cols: usize) -> Self {
        Self {
            rows: Dim::BATCH,
            cols: Dim::N(cols),
        }
    }
    pub fn rows_cols(&self, batch_size: usize) -> (usize, usize) {
        (
            self.rows.concrete(batch_size),
            self.cols.concrete(batch_size),
        )
    }
}

pub enum Node {
    Input(Shape),
    Param(Shape),
    Operation(Operation),
}

#[derive(Default)]
pub struct Graph {
    /// INVARIANT: the nodes are in topological order because to build one we need the NodeIds of the previous
    nodes: Vec<Node>,
    ids: Vec<NodeId>,
}

impl Graph {
    pub fn add_node(&mut self, node: Node) -> NodeId {
        self.shape_of_node(&node);
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        self.ids.push(id);
        id
    }
    pub fn add_param(&mut self, shape: Shape) -> NodeId {
        self.add_node(Node::Param(shape))
    }
    pub fn add_input(&mut self, shape: Shape) -> NodeId {
        self.add_node(Node::Input(shape))
    }
    pub fn add_op(&mut self, op: Operation) -> NodeId {
        self.add_node(Node::Operation(op))
    }

    pub(crate) fn shape_of(&self, node_id: NodeId) -> Shape {
        self.nodes.get(node_id.0).map_or_else(
            || {
                panic!(
                    "node id {} does not refer to a node in this graph",
                    node_id.0
                )
            },
            |node| self.shape_of_node(node),
        )
    }

    fn shape_of_node(&self, node: &Node) -> Shape {
        match node {
            Node::Input(shape) | Node::Param(shape) => *shape,
            Node::Operation(operation) => {
                let inputs = match operation {
                    Operation::MatMul(lhs, rhs)
                    | Operation::Add(lhs, rhs)
                    | Operation::LossL2(lhs, rhs) => vec![self.shape_of(*lhs), self.shape_of(*rhs)],
                    Operation::L1(input, _) => vec![self.shape_of(*input)],
                };
                operation.shape(&inputs)
            }
        }
    }

    pub(crate) fn traverse(
        &self,
        out_id: Option<NodeId>,
    ) -> impl Iterator<Item = (&NodeId, &Node)> {
        let limit = match out_id {
            Some(id) => self
                .ids
                .iter()
                .position(|&node_id| node_id == id)
                .map_or(0, |index| index + 1),
            None => self.nodes.len(),
        };

        self.ids.iter().zip(&self.nodes).take(limit)
    }

    pub(crate) fn reverse(
        &self,
        start_id: Option<NodeId>,
    ) -> impl Iterator<Item = (&NodeId, &Node)> {
        let mut index = match start_id {
            Some(id) => self.ids.iter().position(|&x| x == id),
            None => Some(self.nodes.len().saturating_sub(1)),
        };

        std::iter::from_fn(move || {
            let i = index?;

            index = i.checked_sub(1);

            Some((&self.ids[i], &self.nodes[i]))
        })
    }
}

pub(super) fn broadcast_dim(lhs: Dim, rhs: Dim) -> Dim {
    if lhs == rhs {
        lhs
    } else if lhs == Dim::N(1) {
        rhs
    } else if rhs == Dim::N(1) {
        lhs
    } else {
        panic!("Add dimensions are not broadcast-compatible")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

#[cfg(test)]
mod tests {
    use crate::graph::Operation::LossL2;

    use super::*;

    #[test]
    fn test_build() {
        let n_features = 2;
        let mut graph: Graph = Graph::default();
        let target = graph.add_input(Shape::batch_rows(1));
        let input = graph.add_input(Shape::batch_rows(n_features));
        let weight = graph.add_param(Shape::fixed(2, 1));
        let bias = graph.add_param(Shape::fixed(1, 1));
        let mul = graph.add_op(Operation::MatMul(input, weight));
        let out = graph.add_op(Operation::Add(mul, bias));
        let cost = graph.add_op(LossL2(out, target));

        assert_eq!(graph.traverse(Some(out)).count(), 6);
        assert_eq!(graph.reverse(Some(cost)).count(), 7);
        assert_eq!(graph.shape_of(out).rows_cols(8), (8, 1));
        assert_eq!(graph.shape_of(cost), Shape::fixed(1, 1));
    }

    #[test]
    #[should_panic(expected = "MatMul inner dimensions must match")]
    fn test_build_rejects_invalid_matmul_shape() {
        let mut graph = Graph::default();
        let lhs = graph.add_input(Shape::batch_rows(2));
        let rhs = graph.add_param(Shape::fixed(3, 1));

        graph.add_op(Operation::MatMul(lhs, rhs));
    }

    #[test]
    #[should_panic(expected = "LossL2 inputs must have the same shape")]
    fn test_build_rejects_invalid_loss_shape() {
        let mut graph = Graph::default();
        let prediction = graph.add_input(Shape::batch_rows(1));
        let target = graph.add_input(Shape::fixed(1, 1));

        graph.add_op(LossL2(prediction, target));
    }
}
