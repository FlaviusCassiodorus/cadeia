use super::{Element, MatrixView, MatrixViewMut, MatrixViewRead};
use std::fmt;

// Debug: useful for seeing the actual structure of the view.
macro_rules! impl_fmt_debug {
    ($t:ident,$name:literal ) => {
        impl<'a, K: fmt::Debug + Element> fmt::Debug for $t<'a, K> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct($name)
                    .field("rows", &self.layout().rows)
                    .field("cols", &self.layout().cols)
                    .field("row_stride", &self.layout().row_stride)
                    .field("col_stride", &self.layout().col_stride)
                    .field("data", &DebugSlice(&self.data))
                    .finish()
            }
        }
    };
}

struct DebugSlice<'a, T>(&'a [T]);
impl<T: fmt::Debug> fmt::Debug for DebugSlice<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const N: usize = 3;

        let data = self.0;

        f.write_str("[")?;
        // We print N elements in head and N in tail so total 2N elements at most
        // In this case there is no truncation
        if data.len() <= 2 * N {
            for (i, x) in data.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                x.fmt(f)?;
            }
        // If we have more than 2N elements we truncate in the middle
        } else {
            for (i, x) in data[..N].iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                x.fmt(f)?;
            }

            f.write_str(", ..., ")?;

            for (i, x) in data[data.len() - N..].iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                x.fmt(f)?;
            }
        }

        f.write_str("]")
    }
}

impl_fmt_debug!(MatrixView, "MatrixView");
impl_fmt_debug!(MatrixViewMut, "MatrixViewMut");
