use crate::matrix::{Element, MatrixLayout, MatrixViewRead, MatrixViewWrite};

fn validate_matmut_shape(out: &MatrixLayout, a: &MatrixLayout, b: &MatrixLayout) {
    assert!(
        a.cols == b.rows && a.rows == out.rows && b.cols == out.cols,
        "MatMul shape mismatch"
    );
}

pub fn matmul_naive_ijk<K: Element>(
    out: &mut (impl MatrixViewWrite<K = K> + ?Sized),
    a: &(impl MatrixViewRead<K = K> + ?Sized),
    b: &(impl MatrixViewRead<K = K> + ?Sized),
) {
    validate_matmut_shape(out.layout(), a.layout(), b.layout());
    let (a_rows, a_cols, b_cols) = (a.layout().rows, a.layout().cols, b.layout().cols);
    for i in 0..a_rows {
        for j in 0..b_cols {
            for k in 0..a_cols {
                *out.get_mut(i, j) += a.at(i, k) * b.at(k, j);
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum LoopOrder {
    Ijk,
    Ikj,
    Jik,
    Jki,
    Kij,
    Kji,
}
impl LoopOrder {
    // Heuristic
    fn best_loop_order(out: &MatrixLayout, a: &MatrixLayout, b: &MatrixLayout) -> Self {
        use LoopOrder::*;

        let orders = [Ijk, Ikj, Jik, Jki, Kij, Kji];

        fn score(
            order: LoopOrder,
            out: &MatrixLayout,
            a: &MatrixLayout,
            b: &MatrixLayout,
        ) -> (usize, usize) {
            // Stride of the innermost dimension.
            let (a_inner, b_inner, c_inner) = match order {
                Ijk | Jik => (a.col_stride, b.row_stride, 0),
                Ikj | Kij => (0, b.col_stride, out.col_stride),
                Jki | Kji => (a.row_stride, 0, out.row_stride),
            };

            // Primary objective: make A and B accesses contiguous.
            // C gets a somewhat smaller weight because the reduction
            // dimension (k) lets C stay in a register for IJK-style loops.
            // Using a square-ish penalty rather than raw stride makes
            // large-stride accesses much more expensive.
            let primary = a_inner.saturating_mul(a_inner)
                + b_inner.saturating_mul(b_inner)
                + c_inner.saturating_mul(c_inner) / 2;

            // Small tie-breaker: prefer reuse in the middle loop.
            let secondary = match order {
                Ijk => b.col_stride + out.col_stride,
                Ikj => a.col_stride + b.row_stride,
                Jik => a.row_stride + out.row_stride,
                Jki => a.col_stride + b.row_stride,
                Kij => a.row_stride + out.row_stride,
                Kji => b.col_stride + out.col_stride,
            };

            (primary, secondary)
        }

        orders
            .into_iter()
            .min_by_key(|&order| {
                let (primary, secondary) = score(order, out, a, b);
                (primary, secondary)
            })
            .unwrap()
    }
}

#[derive(Copy, Clone)]
enum RowColMajor {
    RowMajor,
    ColMajor,
}

/// Uses a fixed loop order and literal standard-layout indexing for each layout tuple.
fn matmul_row_column_major<K: Element>(
    out: &mut (impl MatrixViewWrite<K = K> + ?Sized),
    a: &(impl MatrixViewRead<K = K> + ?Sized),
    b: &(impl MatrixViewRead<K = K> + ?Sized),
    out_order: RowColMajor,
    a_order: RowColMajor,
    b_order: RowColMajor,
) -> LoopOrder {
    let (a_rows, a_cols) = a.layout().shape();
    let (b_rows, b_cols) = b.layout().shape();
    let (out_rows, out_cols) = out.layout().shape();
    let a_data = a.data();
    let b_data = b.data();
    let out_data = out.data_mut();

    let out_row_major = matches!(out_order, RowColMajor::RowMajor);
    let a_row_major = matches!(a_order, RowColMajor::RowMajor);
    let b_row_major = matches!(b_order, RowColMajor::RowMajor);

    match (out_row_major, a_row_major, b_row_major) {
        (true, true, true) => {
            for i in 0..a_rows {
                for k in 0..a_cols {
                    let a_ik = a_data[i * a_cols + k];
                    for j in 0..b_cols {
                        out_data[i * out_cols + j] += a_ik * b_data[k * b_cols + j];
                    }
                }
            }
            LoopOrder::Ikj
        }
        (true, true, false) => {
            for i in 0..a_rows {
                for j in 0..b_cols {
                    let mut sum = K::zero();
                    for k in 0..a_cols {
                        sum += a_data[i * a_cols + k] * b_data[j * b_rows + k];
                    }
                    out_data[i * out_cols + j] += sum;
                }
            }
            LoopOrder::Ijk
        }
        (true, false, true) => {
            for k in 0..a_cols {
                for i in 0..a_rows {
                    let a_ik = a_data[k * a_rows + i];
                    for j in 0..b_cols {
                        out_data[i * out_cols + j] += a_ik * b_data[k * b_cols + j];
                    }
                }
            }
            LoopOrder::Kij
        }
        (true, false, false) => {
            for j in 0..b_cols {
                for k in 0..a_cols {
                    let b_kj = b_data[j * b_rows + k];
                    for i in 0..a_rows {
                        out_data[i * out_cols + j] += a_data[k * a_rows + i] * b_kj;
                    }
                }
            }
            LoopOrder::Jki
        }
        (false, true, true) => {
            for i in 0..a_rows {
                for k in 0..a_cols {
                    let a_ik = a_data[i * a_cols + k];
                    for j in 0..b_cols {
                        out_data[j * out_rows + i] += a_ik * b_data[k * b_cols + j];
                    }
                }
            }
            LoopOrder::Ikj
        }
        (false, true, false) => {
            for j in 0..b_cols {
                for i in 0..a_rows {
                    let mut sum = K::zero();
                    for k in 0..a_cols {
                        sum += a_data[i * a_cols + k] * b_data[j * b_rows + k];
                    }
                    out_data[j * out_rows + i] += sum;
                }
            }
            LoopOrder::Jik
        }
        (false, false, true) => {
            for k in 0..a_cols {
                for j in 0..b_cols {
                    let b_kj = b_data[k * b_cols + j];
                    for i in 0..a_rows {
                        out_data[j * out_rows + i] += a_data[k * a_rows + i] * b_kj;
                    }
                }
            }
            LoopOrder::Kji
        }
        (false, false, false) => {
            for j in 0..b_cols {
                for k in 0..a_cols {
                    let b_kj = b_data[j * b_rows + k];
                    for i in 0..a_rows {
                        out_data[j * out_rows + i] += a_data[k * a_rows + i] * b_kj;
                    }
                }
            }
            LoopOrder::Jki
        }
    }
}

pub fn matmul_optimized_loop_order<K: Element>(
    out: &mut (impl MatrixViewWrite<K = K> + ?Sized),
    a: &(impl MatrixViewRead<K = K> + ?Sized),
    b: &(impl MatrixViewRead<K = K> + ?Sized),
) {
    validate_matmut_shape(out.layout(), a.layout(), b.layout());
    // The more complex logic is only worth if the matrix is big enough
    if a.size() < 10 && b.size() < 10 {
        matmul_naive_ijk(out, a, b);
        return;
    }
    let (out_layout, a_layout, b_layout) = (out.layout(), a.layout(), b.layout());

    // When the stride is 1, the compiler might not perform loop versioning to allow SIMD.
    // We give it a nudge by treating the row/column major cases explicitly
    fn helper_layout(m: &MatrixLayout) -> Option<RowColMajor> {
        if m.is_row_major() {
            if m.is_column_major() {
                debug_assert!(m.rows == 1 && m.cols == 1);
            }
            Some(RowColMajor::RowMajor)
        } else if m.is_column_major() {
            Some(RowColMajor::ColMajor)
        } else {
            None
        }
    }

    if let (Some(out_order), Some(a_order), Some(b_order)) = (
        helper_layout(out_layout),
        helper_layout(a_layout),
        helper_layout(b_layout),
    ) {
        matmul_row_column_major(out, a, b, out_order, a_order, b_order);
        return;
    }

    let loop_order = LoopOrder::best_loop_order(out_layout, a_layout, b_layout);
    match loop_order {
        LoopOrder::Ijk => {
            for i in 0..a.layout().rows {
                for j in 0..b.layout().cols {
                    let mut sum = K::zero();
                    for k in 0..a.layout().cols {
                        sum += a.at(i, k) * b.at(k, j);
                    }
                    *out.get_mut(i, j) += sum;
                }
            }
        }

        LoopOrder::Ikj => {
            for i in 0..a.layout().rows {
                for k in 0..a.layout().cols {
                    let a_ik = a.at(i, k);
                    for j in 0..b.layout().cols {
                        *out.get_mut(i, j) += a_ik * b.at(k, j);
                    }
                }
            }
        }

        LoopOrder::Jik => {
            for j in 0..b.layout().cols {
                for i in 0..a.layout().rows {
                    let mut sum = K::zero();
                    for k in 0..a.layout().cols {
                        sum += a.at(i, k) * b.at(k, j);
                    }
                    *out.get_mut(i, j) += sum;
                }
            }
        }

        LoopOrder::Jki => {
            for j in 0..b.layout().cols {
                for k in 0..a.layout().cols {
                    let b_kj = b.at(k, j);
                    for i in 0..a.layout().rows {
                        *out.get_mut(i, j) += a.at(i, k) * b_kj;
                    }
                }
            }
        }

        LoopOrder::Kij => {
            for k in 0..a.layout().cols {
                for i in 0..a.layout().rows {
                    let a_ik = a.at(i, k);
                    for j in 0..b.layout().cols {
                        *out.get_mut(i, j) += a_ik * b.at(k, j);
                    }
                }
            }
        }

        LoopOrder::Kji => {
            for k in 0..a.layout().cols {
                for j in 0..b.layout().cols {
                    let b_kj = b.at(k, j);
                    for i in 0..a.layout().rows {
                        *out.get_mut(i, j) += a.at(i, k) * b_kj;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{MatrixOwned, MatrixView, MatrixViewMut};
    use super::*;
    fn test_methods<K: Element>(a: &MatrixView<K>, b: &MatrixView<K>, expected: &MatrixView<K>) {
        let mut out_mat = MatrixOwned::zeros(3, 3);
        let mut out = out_mat.as_view_mut();

        matmul_naive_ijk(&mut out, a, b);
        assert!(out.eq_matrix(expected));
        out.fill_zero();
        matmul_optimized_loop_order(&mut out, a, b);
        assert!(out.eq_matrix(expected));
    }

    #[test]
    fn test_matmul_ijk() {
        let mat = MatrixOwned::from_vec(vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3);
        let view = mat.as_view();

        // 3x3 · 3x3
        let a = view.submatrix(0..3, 0..3);
        let b = view.submatrix(0..3, 0..3);

        let mat_expected =
            MatrixOwned::from_vec(vec![30u8, 36, 42, 66, 81, 96, 102, 126, 150], 3, 3);
        let expected = mat_expected.as_view();
        test_methods(&a, &b, &expected);

        // 3x2 · 2x3
        let a = view.submatrix(0..3, 0..2);
        let b = view.submatrix(0..2, 0..3);

        let mat_expected = MatrixOwned::from_vec(vec![9u8, 12, 15, 24, 33, 42, 39, 54, 69], 3, 3);
        let expected = mat_expected.as_view();
        test_methods(&a, &b, &expected);

        // 3x1 · 1x3
        let a = view.sub_col(0);
        let b = view.sub_row(0);

        let mat_expected = MatrixOwned::from_vec(vec![1u8, 2, 3, 4, 8, 12, 7, 14, 21], 3, 3);
        let expected = mat_expected.as_view();
        test_methods(&a, &b, &expected);
    }

    #[test]
    fn test_matmul_mixed_layouts_accumulate_into_output() {
        let expected_data = [30u8, 24, 18, 84, 69, 54, 138, 114, 90];

        for a_column_major in [false, true] {
            for b_column_major in [false, true] {
                for out_column_major in [false, true] {
                    let a_data = if a_column_major {
                        [1u8, 4, 7, 2, 5, 8, 3, 6, 9]
                    } else {
                        [1u8, 2, 3, 4, 5, 6, 7, 8, 9]
                    };
                    let b_data = if b_column_major {
                        [9u8, 6, 3, 8, 5, 2, 7, 4, 1]
                    } else {
                        [9u8, 8, 7, 6, 5, 4, 3, 2, 1]
                    };
                    let a_layout = if a_column_major {
                        MatrixLayout::column_major(3, 3)
                    } else {
                        MatrixLayout::row_major(3, 3)
                    };
                    let b_layout = if b_column_major {
                        MatrixLayout::column_major(3, 3)
                    } else {
                        MatrixLayout::row_major(3, 3)
                    };
                    let a = MatrixView::new(&a_data, a_layout);
                    let b = MatrixView::new(&b_data, b_layout);
                    let mut out_mat = MatrixOwned::full(3, 3, 1u8);
                    let mut out = if out_column_major {
                        out_mat.as_view_mut().transpose()
                    } else {
                        out_mat.as_view_mut()
                    };

                    matmul_optimized_loop_order(&mut out, &a, &b);

                    for i in 0..3 {
                        for j in 0..3 {
                            assert_eq!(
                                out.at(i, j),
                                expected_data[i * 3 + j] + 1,
                                "unexpected result for layouts: A column={a_column_major}, B column={b_column_major}, out column={out_column_major}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn standard_layout_data(rows: usize, cols: usize, column_major: bool) -> Vec<f32> {
        (0..rows * cols)
            .map(|index| {
                let (row, col) = if column_major {
                    (index % rows, index / rows)
                } else {
                    (index / cols, index % cols)
                };
                (row * 3 + col + 1) as f32
            })
            .collect()
    }

    #[test]
    fn test_standard_layout_matmul_f32_square_and_rectangular() {
        for (rows, inner, cols) in [(4, 4, 4), (4, 5, 6)] {
            for a_column_major in [false, true] {
                for b_column_major in [false, true] {
                    for out_column_major in [false, true] {
                        let a_data = standard_layout_data(rows, inner, a_column_major);
                        let b_data = standard_layout_data(inner, cols, b_column_major);
                        let a_layout = if a_column_major {
                            MatrixLayout::column_major(rows, inner)
                        } else {
                            MatrixLayout::row_major(rows, inner)
                        };
                        let b_layout = if b_column_major {
                            MatrixLayout::column_major(inner, cols)
                        } else {
                            MatrixLayout::row_major(inner, cols)
                        };
                        let out_layout = if out_column_major {
                            MatrixLayout::column_major(rows, cols)
                        } else {
                            MatrixLayout::row_major(rows, cols)
                        };
                        let a = MatrixView::new(&a_data, a_layout);
                        let b = MatrixView::new(&b_data, b_layout);

                        let mut expected_data = vec![1.0_f32; rows * cols];
                        let mut expected = MatrixViewMut::new(
                            &mut expected_data,
                            MatrixLayout::row_major(rows, cols),
                        );
                        matmul_naive_ijk(&mut expected, &a, &b);

                        let mut out_data = vec![1.0_f32; rows * cols];
                        let mut out = MatrixViewMut::new(&mut out_data, out_layout.clone());
                        matmul_optimized_loop_order(&mut out, &a, &b);

                        if rows == inner && inner == cols {
                            let mut order_out_data = vec![1.0_f32; rows * cols];
                            let mut order_out =
                                MatrixViewMut::new(&mut order_out_data, out_layout.clone());
                            let selected_order = matmul_row_column_major(
                                &mut order_out,
                                &a,
                                &b,
                                if out_column_major {
                                    RowColMajor::ColMajor
                                } else {
                                    RowColMajor::RowMajor
                                },
                                if a_column_major {
                                    RowColMajor::ColMajor
                                } else {
                                    RowColMajor::RowMajor
                                },
                                if b_column_major {
                                    RowColMajor::ColMajor
                                } else {
                                    RowColMajor::RowMajor
                                },
                            );
                            assert_eq!(
                                selected_order,
                                LoopOrder::best_loop_order(&out_layout, a.layout(), b.layout()),
                                "loop order mismatch for square layout tuple: output column={out_column_major}, A column={a_column_major}, B column={b_column_major}"
                            );
                        }

                        let actual = MatrixView::new(&out_data, out_layout);
                        let expected =
                            MatrixView::new(&expected_data, MatrixLayout::row_major(rows, cols));
                        assert!(
                            actual.eq_matrix(&expected),
                            "result mismatch for shape {rows}x{inner}x{cols}, layouts A column={a_column_major}, B column={b_column_major}, output column={out_column_major}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_best_loop_order_layouts() {
        let n = 16;
        let row = MatrixLayout::row_major(n, n);
        let col = MatrixLayout::column_major(n, n);
        // RR: A row-major, B row-major
        assert_eq!(LoopOrder::best_loop_order(&row, &row, &row), LoopOrder::Ikj);
        // CC: A column-major, B column-major, row-major output
        assert_eq!(LoopOrder::best_loop_order(&row, &col, &col), LoopOrder::Jki);
        //RC: A row-major, B column-major
        assert_eq!(LoopOrder::best_loop_order(&row, &row, &col), LoopOrder::Ijk);
        // CR: A column-major, B row-major
        assert_eq!(LoopOrder::best_loop_order(&row, &col, &row), LoopOrder::Kij);
    }
}
