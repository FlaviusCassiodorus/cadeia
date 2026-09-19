use super::matmul;
use super::{DefaultElement, Element};
use std::ops::Range;

#[derive(Clone)]
pub struct MatrixLayout {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) row_stride: usize,
    pub(super) col_stride: usize,
}

fn is_valid_layout(layout: &MatrixLayout, data_len: usize) -> bool {
    if layout.rows == 0 || layout.cols == 0 {
        return true;
    }

    // It is valid if the maximum element can be reached
    let max = (layout.rows - 1) * layout.row_stride + (layout.cols - 1) * layout.col_stride;
    max < data_len
}

impl MatrixLayout {
    pub fn row_major(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_stride: cols,
            col_stride: 1,
        }
    }
    pub fn column_major(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_stride: 1,
            col_stride: rows,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /*
    fn is_row_major(&self) -> bool {
        self.col_stride == 1
    }
    fn is_column_major(&self) -> bool {
        self.row_stride == 1
    }
    */
}

// Invariant ensures data is valid
// We have a reference to the data so it will remain valid
// All methods that build/change MatrixView must ensure that
pub trait MatrixViewRead {
    type K: Element;
    fn layout(&self) -> &MatrixLayout;
    fn data(&self) -> &[Self::K];

    #[inline]
    fn same_shape<M>(&self, other: &M) -> bool
    where
        M: MatrixViewRead<K = Self::K>,
    {
        self.layout().rows == other.layout().rows && self.layout().cols == other.layout().cols
    }

    #[inline]
    fn _idx(&self, i: usize, j: usize) -> usize {
        i * self.layout().row_stride + j * self.layout().col_stride
    }
    #[inline]
    fn _check_idx(&self, i: usize, j: usize) -> bool {
        i < self.layout().rows && j < self.layout().cols
    }
    #[inline]
    fn _prefer_row_major_iteration<M>(&self, other: &M) -> bool
    where
        M: MatrixViewRead<K = Self::K>,
    {
        let self_layout = self.layout();
        let other_layout = other.layout();
        let row_major_score = self_layout.col_stride + other_layout.col_stride;
        let column_major_score = self_layout.row_stride + other_layout.row_stride;
        row_major_score <= column_major_score
    }

    #[inline]
    fn try_at(&self, i: usize, j: usize) -> Option<Self::K> {
        if i < self.layout().rows && j < self.layout().cols {
            Some(self.data()[self._idx(i, j)])
        } else {
            None
        }
    }

    #[inline]
    fn at(&self, i: usize, j: usize) -> Self::K {
        assert!(i < self.layout().rows && j < self.layout().cols);
        self.data()[self._idx(i, j)]
    }
    fn eq_matrix<M>(&self, other: &M) -> bool
    where
        M: MatrixViewRead<K = Self::K>,
    {
        self.same_shape(other)
            && (0..self.layout().rows)
                .all(|i| (0..self.layout().cols).all(|j| self.at(i, j) == other.at(i, j)))
    }
}

// Test operations between row-major and column-major matrix views.

pub trait MatrixViewWrite: MatrixViewRead {
    fn data_mut(&mut self) -> &mut [Self::K];
    fn get_mut(&mut self, i: usize, j: usize) -> &mut Self::K {
        assert!(self._check_idx(i, j));
        let idx = self._idx(i, j);
        self.data_mut().get_mut(idx).unwrap()
    }

    fn fill_with(&mut self, mut f: impl FnMut() -> Self::K) {
        let (rows, cols) = self.layout().shape();
        for i in 0..rows {
            for j in 0..cols {
                *self.get_mut(i, j) = f();
            }
        }
    }

    fn add_assign(&mut self, other: &impl MatrixViewRead<K = <Self>::K>) {
        assert!(self.same_shape(other), "Shape mismatch");
        let layout = other.layout();
        let is_row_major_iter = self._prefer_row_major_iteration(other);
        // Inner loop on the more packed data
        if is_row_major_iter {
            for i in 0..layout.rows {
                for j in 0..layout.cols {
                    *self.get_mut(i, j) += other.at(i, j);
                }
            }
        } else {
            for j in 0..layout.cols {
                for i in 0..layout.rows {
                    *self.get_mut(i, j) += other.at(i, j);
                }
            }
        }
    }

    fn matmul_assign(
        &mut self,
        a: &impl MatrixViewRead<K = <Self>::K>,
        b: &impl MatrixViewRead<K = <Self>::K>,
    ) {
        // matmul::matmul_naive_ijk(self, a, b);
        matmul::matmul_optimized_loop_order(self, a, b);
    }
}

pub struct MatrixView<'a, K = DefaultElement> {
    pub(super) data: &'a [K],
    pub(super) layout: MatrixLayout,
}
impl<'a, K: Element> MatrixView<'a, K> {
    pub fn new(data: &'a [K], layout: MatrixLayout) -> Self {
        assert!(is_valid_layout(&layout, data.len()));
        Self { data, layout }
    }
    pub fn transpose(&self) -> Self {
        let layout = MatrixLayout {
            rows: self.layout.cols,
            cols: self.layout.rows,
            row_stride: self.layout.col_stride,
            col_stride: self.layout.row_stride,
        };
        Self {
            data: self.data,
            layout,
        }
    }
    pub fn submatrix(&self, row_range: Range<usize>, col_range: Range<usize>) -> Self {
        assert!(
            row_range.start <= row_range.end && row_range.end <= self.layout.rows,
            "Row range out of bounds"
        );
        assert!(
            col_range.start <= col_range.end && col_range.end <= self.layout.cols,
            "Column range out of bounds"
        );

        let layout = MatrixLayout {
            rows: row_range.end - row_range.start,
            cols: col_range.end - col_range.start,
            row_stride: self.layout.row_stride,
            col_stride: self.layout.col_stride,
        };

        if layout.rows == 0 || layout.cols == 0 {
            return Self::new(&self.data[..0], layout);
        }

        // Positive strides so we can make the slice we view in smaller
        let first_element = self._idx(row_range.start, col_range.start);
        let last_element = self._idx(row_range.end - 1, col_range.end - 1);
        let data = &self.data[first_element..=last_element];
        Self::new(data, layout)
    }
    pub fn sub_rows(&self, range: Range<usize>) -> Self {
        self.submatrix(range, 0..self.layout.cols)
    }

    pub fn sub_cols(&self, range: Range<usize>) -> Self {
        self.submatrix(0..self.layout.rows, range)
    }

    pub fn sub_row(&self, row: usize) -> Self {
        self.submatrix(row..row + 1, 0..self.layout.cols)
    }

    pub fn sub_col(&self, col: usize) -> Self {
        self.submatrix(0..self.layout.rows, col..col + 1)
    }
}

impl<'a, K: Element> MatrixViewRead for MatrixView<'a, K> {
    type K = K;
    fn data(&self) -> &[K] {
        self.data
    }
    fn layout(&self) -> &MatrixLayout {
        &self.layout
    }
}

impl<'a, K: Element> PartialEq for MatrixView<'a, K> {
    fn eq(&self, other: &Self) -> bool {
        self.eq_matrix(other)
    }
}

pub struct MatrixViewMut<'a, K = DefaultElement> {
    pub(super) data: &'a mut [K],
    pub(super) layout: MatrixLayout,
}

impl<'a, K: Element> MatrixViewMut<'a, K> {
    pub fn new(data: &'a mut [K], layout: MatrixLayout) -> Self {
        assert!(is_valid_layout(&layout, data.len()));
        Self { data, layout }
    }
    // Reborrow: MatrixView cannot outlife self, self cannot outlive data
    pub fn as_view(&self) -> MatrixView<'_, K> {
        MatrixView {
            data: self.data,
            layout: self.layout.clone(),
        }
    }
    pub fn transpose(self) -> Self {
        let layout = MatrixLayout {
            rows: self.layout.cols,
            cols: self.layout.rows,
            row_stride: self.layout.col_stride,
            col_stride: self.layout.row_stride,
        };
        Self {
            data: self.data,
            layout,
        }
    }
    pub fn fill(&mut self, value: K) {
        for i in 0..self.layout.rows {
            for j in 0..self.layout.cols {
                *self.get_mut(i, j) = value;
            }
        }
    }
    pub fn fill_zero(&mut self) {
        self.fill(K::zero());
    }
}

impl<'a, K: Element> MatrixViewRead for MatrixViewMut<'a, K> {
    type K = K;
    fn data(&self) -> &[K] {
        self.data
    }
    fn layout(&self) -> &MatrixLayout {
        &self.layout
    }
}
impl<'a, K: Element> MatrixViewWrite for MatrixViewMut<'a, K> {
    fn data_mut(&mut self) -> &mut [Self::K] {
        self.data
    }
}
impl<'a, K: Element> PartialEq for MatrixViewMut<'a, K> {
    fn eq(&self, other: &Self) -> bool {
        self.eq_matrix(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_at_and_try_at() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let view = MatrixView {
            data: &data,
            layout: MatrixLayout::row_major(2, 3),
        };

        assert_eq!(view.at(0, 0), 1.0);
        assert_eq!(view.at(1, 2), 6.0);

        assert_eq!(view.try_at(0, 1), Some(2.0));
        assert_eq!(view.try_at(1, 0), Some(4.0));

        assert_eq!(view.try_at(2, 0), None);
        assert_eq!(view.try_at(0, 3), None);
    }

    #[test]
    #[should_panic]
    fn test_at_panics_out_of_bounds() {
        let data = [1.0, 2.0, 3.0, 4.0];

        let view = MatrixView {
            data: &data,
            layout: MatrixLayout::row_major(2, 2),
        };

        view.at(2, 0);
    }
    #[test]
    fn test_add_into() {
        let mut a_data = [1.0, 2.0, 3.0, 4.0];
        let b_data = [10.0, 20.0, 30.0, 40.0];
        let mut a = MatrixViewMut {
            data: &mut a_data,
            layout: MatrixLayout::row_major(2, 2),
        };
        let b = MatrixView {
            data: &b_data,
            layout: MatrixLayout::row_major(2, 2),
        };
        a.add_assign(&b);
        assert_eq!(a.at(0, 0), 11.0);
        assert_eq!(a.at(0, 1), 22.0);
        assert_eq!(a.at(1, 0), 33.0);
        assert_eq!(a.at(1, 1), 44.0);
    }

    #[test]
    fn test_is_valid_layout() {
        let cases = [
            // Valid contiguous layout
            (
                MatrixLayout {
                    rows: 2,
                    cols: 3,
                    row_stride: 3,
                    col_stride: 1,
                },
                6,
                true,
            ),
            // Max element is out of bounds
            (
                MatrixLayout {
                    rows: 2,
                    cols: 3,
                    row_stride: 3,
                    col_stride: 1,
                },
                5,
                false,
            ),
            // Valid non-contiguous layout
            (
                MatrixLayout {
                    rows: 3,
                    cols: 2,
                    row_stride: 5,
                    col_stride: 2,
                },
                13,
                true,
            ),
            // Non-contiguous layout is out of bounds
            (
                MatrixLayout {
                    rows: 3,
                    cols: 2,
                    row_stride: 5,
                    col_stride: 2,
                },
                12,
                false,
            ),
        ];

        for (layout, data_len, expected) in cases {
            assert_eq!(
                is_valid_layout(&layout, data_len),
                expected,
                "unexpected result for layout: {:?}, data_len: {data_len}",
                (
                    layout.rows,
                    layout.cols,
                    layout.row_stride,
                    layout.col_stride
                )
            );
        }
    }

    #[test]
    fn zero_sized_views_are_valid() {
        let empty: [u8; 0] = [];

        let rows_zero = MatrixView::new(&empty, MatrixLayout::row_major(0, 3));
        assert_eq!(rows_zero.layout().shape(), (0, 3));

        let cols_zero = MatrixView::new(&empty, MatrixLayout::column_major(3, 0));
        assert_eq!(cols_zero.layout().shape(), (3, 0));
    }

    #[test]
    fn fill_only_writes_logical_elements() {
        let mut data = [1, 2, 99, 3, 4];
        let layout = MatrixLayout {
            rows: 2,
            cols: 2,
            row_stride: 3,
            col_stride: 1,
        };
        {
            let mut view = MatrixViewMut::new(&mut data, layout);
            view.fill_zero();
        }

        assert_eq!(data, [0, 0, 99, 0, 0]);
    }

    #[test]
    fn fill_with_writes_values_in_logical_order() {
        let mut data = [0, 0, 99, 0, 0];
        let layout = MatrixLayout {
            rows: 2,
            cols: 2,
            row_stride: 3,
            col_stride: 1,
        };
        let mut next = 1;

        {
            let mut view = MatrixViewMut::new(&mut data, layout);
            view.fill_with(|| {
                let value = next;
                next += 1;
                value
            });
        }

        assert_eq!(data, [1, 2, 99, 3, 4]);
    }

    #[test]
    fn as_view_creates_readonly_view_and_allows_mutation_afterward() {
        let mut data = vec![1, 2, 3, 4];

        let layout = MatrixLayout::row_major(2, 2); // adjust to your API
        let view_mut = MatrixViewMut::<u8>::new(&mut data, layout);

        {
            let readonly = view_mut.as_view();

            // This wont work because the reborrow freezes the mutable reference inside the view
            // view.data[0] = 123;
            assert_eq!(readonly.at(0, 0), 1);
            assert_eq!(readonly.at(0, 1), 2);
            assert_eq!(readonly.at(1, 0), 3);
            assert_eq!(readonly.at(1, 1), 4);
        }

        // The shared reborrow has ended, so mutable access is possible again.
        // The `view`` has a mutable reference inside so itself does not need to be mutable
        // for us to change it
        view_mut.data[0] = 123;

        assert_eq!(view_mut.data[0], 123);
    }

    #[test]
    fn column_major_indexing() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let layout = MatrixLayout::column_major(2, 3);
        let view = MatrixView::<u8>::new(&data, layout);

        // Column-major layout:
        //
        // [1, 3, 5]
        // [2, 4, 6]
        //
        // The data is stored as:
        // [1, 2, 3, 4, 5, 6]

        assert_eq!(view.at(0, 0), 1);
        assert_eq!(view.at(1, 0), 2);
        assert_eq!(view.at(0, 1), 3);
        assert_eq!(view.at(1, 1), 4);
        assert_eq!(view.at(0, 2), 5);
        assert_eq!(view.at(1, 2), 6);

        assert_eq!(view.try_at(0, 0), Some(1));
        assert_eq!(view.try_at(1, 0), Some(2));
        assert_eq!(view.try_at(0, 1), Some(3));
        assert_eq!(view.try_at(1, 1), Some(4));
        assert_eq!(view.try_at(0, 2), Some(5));
        assert_eq!(view.try_at(1, 2), Some(6));

        // Out-of-bounds accesses.
        assert_eq!(view.try_at(2, 0), None);
        assert_eq!(view.try_at(0, 3), None);
    }

    fn fixtures() -> impl Iterator<Item = (Vec<u8>, MatrixLayout)> {
        let shapes = [(3, 3), (4, 3), (3, 4), (3, 1), (1, 3)];
        let layouts = [true, false]; // true = row major

        shapes
            .into_iter()
            .flat_map(move |(rows, cols)| {
                layouts
                    .into_iter()
                    .map(move |is_row_major| (rows, cols, is_row_major))
            })
            .map(move |(rows, cols, is_row_major)| {
                let layout = if is_row_major {
                    MatrixLayout::row_major(rows, cols)
                } else {
                    MatrixLayout::column_major(rows, cols)
                };

                let data = (1..=(rows * cols)).map(|x| x as u8).collect();

                (data, layout)
            })
    }

    #[test]
    fn test_transpose() {
        for (data, layout) in fixtures() {
            let mut data = data;
            let view_mut = MatrixViewMut::<u8>::new(&mut data, layout.clone());
            let view = view_mut.as_view();

            let view_t = view.transpose();
            assert_eq!(view_t.layout.rows, layout.cols);
            assert_eq!(view_t.layout.cols, layout.rows);

            // We need this because as long as the mutable reference is alive we cant read the view
            let mut tape = Vec::with_capacity(layout.rows * layout.cols);
            for i in 0..layout.rows {
                for j in 0..layout.cols {
                    let ori_ij = view.at(i, j);
                    assert_eq!(view_t.at(j, i), ori_ij);
                    tape.push(ori_ij);
                }
            }

            let view_mut_t = view_mut.transpose();
            assert_eq!(view_mut_t.layout.rows, layout.cols);
            assert_eq!(view_mut_t.layout.cols, layout.rows);

            let mut c = 0;
            for i in 0..layout.rows {
                for j in 0..layout.cols {
                    assert_eq!(view_mut_t.at(j, i), tape[c]);
                    c += 1;
                }
            }
        }
    }

    #[test]
    fn test_submatrix_views() {
        for (data, layout) in fixtures() {
            let mut data = data;
            let view_mut = MatrixViewMut::<u8>::new(&mut data, layout.clone());
            let view = view_mut.as_view();

            let rows = layout.rows;
            let cols = layout.cols;

            // Single row at beginning, middle and end.
            for row in [0, rows / 2, rows - 1] {
                let sub = view.sub_row(row);

                assert_eq!(sub.layout.rows, 1);
                assert_eq!(sub.layout.cols, cols);

                for j in 0..cols {
                    assert_eq!(sub.at(0, j), view.at(row, j));
                }
            }

            // Single column at beginning, middle and end.
            for col in [0, cols / 2, cols - 1] {
                let sub = view.sub_col(col);

                assert_eq!(sub.layout.rows, rows);
                assert_eq!(sub.layout.cols, 1);

                for i in 0..rows {
                    assert_eq!(sub.at(i, 0), view.at(i, col));
                }
            }

            // The `interior submatrix` only makes sense if there are more than 2
            fn ranges(n: usize) -> Vec<Range<usize>> {
                match n {
                    0 | 1 => vec![],
                    #[allow(clippy::single_range_in_vec_init)]
                    2 => vec![0..2],
                    n => vec![0..2, 1..n - 1, n - 2..n],
                }
            }

            // Row ranges: beginning, middle and end.
            for row_range in ranges(rows) {
                let sub = view.sub_rows(row_range.clone());

                assert_eq!(sub.layout.rows, row_range.end - row_range.start);
                assert_eq!(sub.layout.cols, cols);

                for i in 0..sub.layout.rows {
                    for j in 0..cols {
                        assert_eq!(sub.at(i, j), view.at(row_range.start + i, j),);
                    }
                }
            }

            // Column ranges: beginning, middle and end.
            for col_range in ranges(cols) {
                let sub = view.sub_cols(col_range.clone());

                assert_eq!(sub.layout.rows, rows);
                assert_eq!(sub.layout.cols, col_range.end - col_range.start);

                for i in 0..rows {
                    for j in 0..sub.layout.cols {
                        assert_eq!(sub.at(i, j), view.at(i, col_range.start + j),);
                    }
                }
            }

            let row_ranges = ranges(rows);
            let col_ranges = ranges(cols);
            for row_range in row_ranges {
                for col_range in col_ranges.clone() {
                    let sub = view.submatrix(row_range.clone(), col_range.clone());

                    assert_eq!(sub.layout.rows, row_range.end - row_range.start);
                    assert_eq!(sub.layout.cols, col_range.end - col_range.start);

                    for i in 0..sub.layout.rows {
                        for j in 0..sub.layout.cols {
                            assert_eq!(
                                sub.at(i, j),
                                view.at(row_range.start + i, col_range.start + j),
                            );
                        }
                    }
                }
            }
        }
    }
}
