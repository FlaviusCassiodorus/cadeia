mod format;
mod owned;
mod view;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

pub mod matmul;
pub use owned::MatrixOwned;
pub use view::{
    MatrixLayout, MatrixView, MatrixViewMut, MatrixViewRead, MatrixViewWrite, add_broadcast_assign,
    sum_broadcast_assign,
};
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
    fn one() -> Self;
    fn abs(self) -> Self;
    fn from_usize(value: usize) -> Self;
    fn from_f64(value: f64) -> Self;
}
macro_rules! impl_element_float {
    ($t:ty) => {
        impl Element for $t {
            fn zero() -> $t {
                0.0
            }
            fn one() -> $t {
                1.0
            }
            fn abs(self) -> $t {
                <$t>::abs(self)
            }
            fn from_usize(value: usize) -> $t {
                value as $t
            }
            fn from_f64(value: f64) -> $t {
                value as $t
            }
        }
    };
}
macro_rules! impl_element_signed {
    ($t:ty) => {
        impl Element for $t {
            fn zero() -> $t {
                0
            }
            fn one() -> $t {
                1
            }
            fn abs(self) -> $t {
                <$t>::abs(self)
            }
            fn from_usize(value: usize) -> $t {
                value as $t
            }
            fn from_f64(value: f64) -> $t {
                value as $t
            }
        }
    };
}
macro_rules! impl_element_unsigned {
    ($t:ty) => {
        impl Element for $t {
            fn zero() -> $t {
                0
            }
            fn one() -> $t {
                1
            }
            fn abs(self) -> $t {
                self
            }
            fn from_usize(value: usize) -> $t {
                value as $t
            }
            fn from_f64(value: f64) -> $t {
                value as $t
            }
        }
    };
}
impl_element_float!(f32);
impl_element_float!(f64);

impl_element_signed!(i8);
impl_element_signed!(i16);
impl_element_signed!(i32);
impl_element_signed!(i64);
impl_element_signed!(i128);
impl_element_signed!(isize);
impl_element_unsigned!(u8);
impl_element_unsigned!(u16);
impl_element_unsigned!(u32);
impl_element_unsigned!(u64);
impl_element_unsigned!(u128);
impl_element_unsigned!(usize);
