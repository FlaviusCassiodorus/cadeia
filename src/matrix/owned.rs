use crate::matrix::{MatrixViewMut, view::MatrixLayout};

use super::{DefaultElement, Element, MatrixView};

pub struct MatrixOwned<K = DefaultElement> {
    data: Vec<K>,
    rows: usize,
    cols: usize,
}

impl<K: Element> MatrixOwned<K> {
    pub fn zeros(rows: usize, cols: usize) -> Self
    where
        K: Element,
    {
        let data = vec![K::zero(); rows * cols];
        Self { data, rows, cols }
    }

    pub fn full(rows: usize, cols: usize, value: K) -> Self
    where
        K: Element,
    {
        let data = vec![value; rows * cols];
        Self { data, rows, cols }
    }

    pub fn full_with(rows: usize, cols: usize, f: impl FnMut() -> K) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        data.resize_with(rows * cols, f);
        Self { data, rows, cols }
    }

    pub fn as_view(&self) -> MatrixView<'_, K> {
        MatrixView::new(&self.data, MatrixLayout::row_major(self.rows, self.cols))
    }
    pub fn as_view_mut(&mut self) -> MatrixViewMut<'_, K> {
        MatrixViewMut::new(
            &mut self.data,
            MatrixLayout::row_major(self.rows, self.cols),
        )
    }

    pub fn from_vec(data: Vec<K>, rows: usize, cols: usize) -> Self {
        assert!(data.len() == rows * cols);
        Self { data, rows, cols }
    }
}

impl<'a, K: Element> From<&'a MatrixOwned<K>> for MatrixView<'a, K> {
    fn from(value: &'a MatrixOwned<K>) -> Self {
        value.as_view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let matrix: MatrixOwned = MatrixOwned::zeros(2, 3);

        // Usual path
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.data.len(), 6);

        assert!(matrix.data.iter().all(|&x| x == 0.0));

        // Accepts zeros size
        assert_eq!(MatrixOwned::<f64>::zeros(0, 1).rows, 0);
        assert_eq!(MatrixOwned::<f64>::zeros(1, 0).cols, 0);
        assert_eq!(MatrixOwned::<f64>::zeros(0, 0).cols, 0);
    }

    #[test]
    fn test_full() {
        let matrix = MatrixOwned::full(2, 3, 42.0);

        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.data.len(), 6);
        assert!(matrix.data.iter().all(|&x| x == 42.0));
    }

    #[test]
    fn test_full_with() {
        let mut value = 0;
        let matrix = MatrixOwned::full_with(2, 3, || {
            value += 1;
            value
        });

        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.data, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn matrix_owned_can_convert_to_view() {
        let matrix = MatrixOwned::<i32>::zeros(2, 3);
        let _: MatrixView<'_, i32> = (&matrix).into();
        _ = matrix.as_view();
    }
}
