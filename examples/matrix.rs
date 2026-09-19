use cadeia::matrix::{MatrixOwned, MatrixViewRead, MatrixViewWrite};

fn main() {
    // We start by creating a owned matrix (owns the buffer)
    let mut next = 1.0;
    let matrix = MatrixOwned::full_with(2, 3, || {
        let value = next;
        next += 1.0;
        value
    });

    // This is a readonly view into the matrix
    let view = matrix.as_view();

    // Some examples of operations we can do, still refering to the same buffer
    println!("matrix: {view:?}");
    println!("shape: {:?}", view.layout().shape());
    println!("matrix[1, 2]: {}", view.at(1, 2));
    println!("matrix[2, 0]: {:?}", view.try_at(2, 0));

    let transposed = view.transpose();
    println!("transpose: {transposed:?}");

    let submatrix = view.submatrix(0..2, 1..3);
    println!("last two columns: {submatrix:?}");

    // Now we add the previous matrix into this one
    let mut sum = MatrixOwned::full(2, 3, 10.0);
    {
        let mut sum_view = sum.as_view_mut();
        sum_view.add_assign(&view);
    }
    println!("10 + matrix: {:?}", sum.as_view());

    // We can also multiply
    let mut mul = MatrixOwned::zeros(2, 2);
    let mut mul_view = mul.as_view_mut();
    mul_view.matmul_assign(&view, &transposed);
    println!("matrix * matrix_transpose: {mul_view:?}");
}
