use crate::matrix::{
    Element, MatrixView, MatrixViewMut, MatrixViewRead, MatrixViewWrite, add_broadcast_assign,
    sum_broadcast_assign,
};

use super::graph_impl::{NodeId, Shape, broadcast_dim};

pub(crate) const MAX_OPERATION_INPUTS: usize = 2;

fn softmax_row_stats<K: Element>(logits: &MatrixView<'_, K>, row: usize) -> (f64, f64) {
    let cols = logits.layout().cols();
    // e^z / sum_j(e^z_j) remains unchanged if we subtract m from each z_j
    // since e^(z-m) = e^z * e^{-m}
    // and this we way we make all exponents less than 1 and prevent overflow
    let max_logit = (0..cols)
        .map(|j| logits.at(row, j).to_f64())
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_sum = (0..cols)
        .map(|j| (logits.at(row, j).to_f64() - max_logit).exp())
        .sum();
    (max_logit, exp_sum)
}

#[derive(Debug)]
pub enum Operation {
    MatMul(NodeId, NodeId),
    Add(NodeId, NodeId),
    LossL2(NodeId, NodeId),
    NormL1(NodeId, f64),
    ReLU(NodeId),
    LossCrossEntropySoftMax(NodeId, NodeId),
}

impl Operation {
    pub(super) fn shape(&self, inputs: &[Shape]) -> Shape {
        match *self {
            Operation::MatMul(_, _) => {
                let [lhs_shape, rhs_shape] = inputs else {
                    unreachable!()
                };
                assert_eq!(
                    lhs_shape.cols, rhs_shape.rows,
                    "MatMul inner dimensions must match"
                );
                Shape {
                    rows: lhs_shape.rows,
                    cols: rhs_shape.cols,
                }
            }
            Operation::Add(_, _) => {
                let [lhs_shape, rhs_shape] = inputs else {
                    unreachable!()
                };
                Shape {
                    rows: broadcast_dim(lhs_shape.rows, rhs_shape.rows),
                    cols: broadcast_dim(lhs_shape.cols, rhs_shape.cols),
                }
            }
            Operation::LossL2(_, _) => {
                let [lhs_shape, rhs_shape] = inputs else {
                    unreachable!()
                };
                assert_eq!(
                    lhs_shape, rhs_shape,
                    "LossL2 inputs must have the same shape"
                );
                Shape::fixed(1, 1)
            }
            Operation::LossCrossEntropySoftMax(_, _) => {
                let [labels_shape, logits_shape] = inputs else {
                    unreachable!()
                };
                assert_eq!(
                    labels_shape, logits_shape,
                    "cross entropy labels and logits must have the same shape"
                );
                assert!(matches!(logits_shape.cols, super::graph_impl::Dim::N(n) if n > 0));
                Shape::fixed(1, 1)
            }
            Operation::NormL1(_, _) => {
                assert_eq!(inputs.len(), 1);
                Shape::fixed(1, 1)
            }
            Operation::ReLU(_) => {
                assert_eq!(inputs.len(), 1);
                inputs[0]
            }
        }
    }
}

pub(crate) trait NodeStorage {
    type K: Element;
    // This function makes sense to exist given the current API
    #[allow(dead_code)]
    fn get_grad(&self, node_id: NodeId) -> MatrixView<'_, Self::K>;
    fn get_mut_grad(&mut self, node_id: NodeId) -> MatrixViewMut<'_, Self::K>;
    fn get_val(&self, node_id: NodeId) -> MatrixView<'_, Self::K>;
    fn get_mut_val(&mut self, node_id: NodeId) -> MatrixViewMut<'_, Self::K>;
    fn get_mut_val_and_grad(&mut self, node_id: NodeId) -> NodeStorageParamView<'_, Self::K>;
    fn get_val_output_and_inputs(
        &mut self,
        output: NodeId,
        inputs: &[NodeId],
    ) -> NodeStorageValueView<'_, Self::K>;
    // We write into input_grad based on out_grad and the input values
    // We get out_grad and inputs so we get 1 + n_inputs read only values
    fn get_grad_and_vals(
        &mut self,
        input_grad: NodeId,
        out_grad: NodeId,
        inputs: &[NodeId],
    ) -> NodeStorageGradView<'_, Self::K>;
}

pub(crate) type NodeStorageValueView<'a, K> = (
    MatrixViewMut<'a, K>,
    [Option<MatrixView<'a, K>>; MAX_OPERATION_INPUTS],
);
pub(crate) type NodeStorageGradView<'a, K> = (
    MatrixViewMut<'a, K>,
    MatrixView<'a, K>,
    [Option<MatrixView<'a, K>>; MAX_OPERATION_INPUTS],
);
pub(crate) type NodeStorageParamView<'a, K> = (MatrixViewMut<'a, K>, MatrixView<'a, K>);

impl Operation {
    pub(crate) fn grad<S>(&self, storage: &mut S, out_id: NodeId)
    where
        S: NodeStorage,
    {
        match *self {
            Operation::MatMul(a, b) => {
                // dA = dUp @ B^T
                let (mut grad_a, grad_up, inputs) = storage.get_grad_and_vals(a, out_id, &[b]);
                grad_a.matmul_assign(&grad_up, &inputs[0].as_ref().unwrap().transpose());
                // dB = A^T @ dUp
                let (mut grad_b, grad_up, inputs) = storage.get_grad_and_vals(b, out_id, &[a]);
                grad_b.matmul_assign(&inputs[0].as_ref().unwrap().transpose(), &grad_up);
            }
            Operation::Add(a, b) => {
                let (mut grad_a, grad_up, _) = storage.get_grad_and_vals(a, out_id, &[]);
                sum_broadcast_assign(&mut grad_a, &grad_up);
                let (mut grad_b, grad_up, _) = storage.get_grad_and_vals(b, out_id, &[]);
                sum_broadcast_assign(&mut grad_b, &grad_up);
            }
            Operation::LossL2(a, b) => {
                // dA = 2 / n * (A - B) * dUp
                let (mut grad_a, grad_up, inputs) = storage.get_grad_and_vals(a, out_id, &[a, b]);
                let a_values = inputs[0].as_ref().unwrap();
                let b_values = inputs[1].as_ref().unwrap();
                assert!(a_values.same_shape(b_values));
                let scale =
                    S::K::from_f64(2.0) / S::K::from_usize(a_values.size()) * grad_up.at(0, 0);
                for i in 0..a_values.layout().rows() {
                    for j in 0..a_values.layout().cols() {
                        *grad_a.get_mut(i, j) += (a_values.at(i, j) - b_values.at(i, j)) * scale;
                    }
                }

                // dB = 2 / n * (B - A) * dUp
                let (mut grad_b, grad_up, inputs) = storage.get_grad_and_vals(b, out_id, &[a, b]);
                let a_values = inputs[0].as_ref().unwrap();
                let b_values = inputs[1].as_ref().unwrap();
                let scale =
                    S::K::from_f64(2.0) / S::K::from_usize(a_values.size()) * grad_up.at(0, 0);
                for i in 0..b_values.layout().rows() {
                    for j in 0..b_values.layout().cols() {
                        *grad_b.get_mut(i, j) += (b_values.at(i, j) - a_values.at(i, j)) * scale;
                    }
                }
            }
            Operation::LossCrossEntropySoftMax(labels, logits) => {
                // Let y be labels, z the logits, L the cross entropy loss
                // L = - sum_i y_i * log(p_i)
                // softmax is p_i = e^z_i / sum_j e^z_j
                // So log(p_i) becomes z_i - log ( sum_j e^z_j)
                // and the sum becomes
                // - sum_i z_i y_i + sum_i y_i * log ( sum_j e^z_j ) =
                // - sum_i z_i y_i + log ( sum_j e^j )
                // where we take the j sum out since it does not depend on i and
                // use the fact that sum y_i = 1
                // dz_i = - y_i + e^z_i / sum_j e^z_j = p_i - y_i
                // since we average across rows:
                // dL/dz_ij = (p_ij - y_ij) / n_rows * dUp
                // We dont need dy since y is not a param and is a leaf
                let (mut grad_z, grad_up, inputs) =
                    storage.get_grad_and_vals(logits, out_id, &[labels, logits]);
                let y = inputs[0].as_ref().unwrap();
                let z = inputs[1].as_ref().unwrap();
                let nrows = y.layout().rows();
                let scale = S::K::from_f64(1.0 / nrows as f64) * grad_up.at(0, 0);
                for i in 0..nrows {
                    let (max_logit, exp_sum) = softmax_row_stats(z, i);
                    for j in 0..y.layout().cols() {
                        let p = S::K::from_f64((z.at(i, j).to_f64() - max_logit).exp() / exp_sum);
                        *grad_z.get_mut(i, j) += (p - y.at(i, j)) * scale;
                    }
                }
            }
            Operation::NormL1(a, coefficient) => {
                // dA = coefficient * sign(A) * dUp, with sign(0) = 0
                let (mut grad_a, grad_up, inputs) = storage.get_grad_and_vals(a, out_id, &[a]);
                let a_values = inputs[0].as_ref().unwrap();
                let scale = S::K::from_f64(coefficient) * grad_up.at(0, 0);
                let zero = S::K::zero();
                for i in 0..a_values.layout().rows() {
                    for j in 0..a_values.layout().cols() {
                        let value = a_values.at(i, j);
                        let sign = if value == zero {
                            zero
                        } else {
                            value / value.abs()
                        };
                        *grad_a.get_mut(i, j) += sign * scale;
                    }
                }
            }
            Operation::ReLU(a) => {
                let (mut grad_a, grad_up, inputs) = storage.get_grad_and_vals(a, out_id, &[a]);
                let values = inputs[0].as_ref().unwrap();
                let zero = S::K::zero();
                for i in 0..values.layout().rows() {
                    for j in 0..values.layout().cols() {
                        if values.at(i, j) > zero {
                            *grad_a.get_mut(i, j) += grad_up.at(i, j);
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn apply<S>(&self, storage: &mut S, out_id: NodeId)
    where
        S: NodeStorage,
    {
        match *self {
            Operation::MatMul(a, b) => {
                let (mut out, inputs) = storage.get_val_output_and_inputs(out_id, &[a, b]);
                out.matmul_assign(inputs[0].as_ref().unwrap(), inputs[1].as_ref().unwrap());
            }
            Operation::Add(a, b) => {
                let (mut out, inputs) = storage.get_val_output_and_inputs(out_id, &[a, b]);
                add_broadcast_assign(&mut out, inputs[0].as_ref().unwrap());
                add_broadcast_assign(&mut out, inputs[1].as_ref().unwrap());
            }
            Operation::LossL2(a, b) => {
                let (mut out, inputs) = storage.get_val_output_and_inputs(out_id, &[a, b]);
                let a = inputs[0].as_ref().unwrap();
                let b = inputs[1].as_ref().unwrap();
                assert!(a.same_shape(b));
                let n = S::K::from_usize(a.size());
                let mut loss = a.sum_with(b, |x, y| {
                    let difference = x - y;
                    difference * difference
                });
                loss /= n;
                assert_eq!(out.layout().shape(), (1, 1));
                *out.get_mut(0, 0) = loss;
            }
            Operation::LossCrossEntropySoftMax(labels, logits) => {
                let (mut out, inputs) =
                    storage.get_val_output_and_inputs(out_id, &[labels, logits]);
                let labels = inputs[0].as_ref().unwrap();
                let logits = inputs[1].as_ref().unwrap();
                assert!(
                    labels.same_shape(logits),
                    "cross entropy labels and logits must have the same shape"
                );
                assert!(logits.layout().cols() > 0);
                assert!(
                    logits.layout().rows() > 0,
                    "cross entropy batch must not be empty"
                );
                let n_classes = logits.layout().cols();
                let mut loss = 0.0;

                for i in 0..logits.layout().rows() {
                    let mut target_logit = 0.0;
                    for j in 0..n_classes {
                        target_logit += labels.at(i, j).to_f64() * logits.at(i, j).to_f64();
                    }
                    let (max_logit, exp_sum) = softmax_row_stats(logits, i);
                    let log_sum_exp = max_logit + exp_sum.ln();
                    loss += log_sum_exp - target_logit;
                }
                loss /= logits.layout().rows() as f64;
                *out.get_mut(0, 0) = S::K::from_f64(loss);
            }
            Operation::NormL1(a, coefficient) => {
                let (mut out, inputs) = storage.get_val_output_and_inputs(out_id, &[a]);
                let a = inputs[0].as_ref().unwrap();
                let mut loss = S::K::zero();
                for i in 0..a.layout().rows() {
                    for j in 0..a.layout().cols() {
                        loss += a.at(i, j).abs();
                    }
                }
                loss *= S::K::from_f64(coefficient);
                *out.get_mut(0, 0) = loss;
            }
            Operation::ReLU(a) => {
                let (mut out, inputs) = storage.get_val_output_and_inputs(out_id, &[a]);
                let input = inputs[0].as_ref().unwrap();
                let zero = S::K::zero();
                for i in 0..input.layout().rows() {
                    for j in 0..input.layout().cols() {
                        let value = input.at(i, j);
                        *out.get_mut(i, j) = if value > zero { value } else { zero };
                    }
                }
            }
        }
    }
}
