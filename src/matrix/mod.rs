mod format;
mod owned;
mod view;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

pub mod matmul;
pub use owned::MatrixOwned;
pub use view::{MatrixLayout, MatrixView, MatrixViewMut, MatrixViewRead, MatrixViewWrite};
pub type DefaultElement = f64;

pub trait Element:
    Copy
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + PartialEq
{
    fn zero() -> Self;
}
macro_rules! impl_element_float {
    ($t:ty) => {
        impl Element for $t {
            fn zero() -> $t {
                0.0
            }
        }
    };
}
macro_rules! impl_element_int {
    ($t:ty) => {
        impl Element for $t {
            fn zero() -> $t {
                0
            }
        }
    };
}
impl_element_float!(f32);
impl_element_float!(f64);

impl_element_int!(i8);
impl_element_int!(i16);
impl_element_int!(i32);
impl_element_int!(i64);
impl_element_int!(i128);
impl_element_int!(isize);
impl_element_int!(u8);
impl_element_int!(u16);
impl_element_int!(u32);
impl_element_int!(u64);
impl_element_int!(u128);
impl_element_int!(usize);
