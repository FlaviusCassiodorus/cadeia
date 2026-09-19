use cadeia::matrix::matmul::{matmul_naive_ijk, matmul_optimized_loop_order};
use cadeia::matrix::{MatrixLayout, MatrixView, MatrixViewMut};
use criterion::{Criterion, criterion_group, criterion_main};

#[derive(Copy, Clone)]
enum LayoutKind {
    RowMajor,
    ColumnMajor,
    Transposed,
}

impl LayoutKind {
    fn label(self) -> &'static str {
        match self {
            Self::RowMajor => "row",
            Self::ColumnMajor => "column",
            Self::Transposed => "transpose",
        }
    }
}

#[derive(Copy, Clone)]
struct MatmulCase {
    name: &'static str,
    rows: usize,
    inner: usize,
    cols: usize,
    a_layout: LayoutKind,
    b_layout: LayoutKind,
    out_layout: LayoutKind,
}

fn read_view<'a>(
    data: &'a [f32],
    rows: usize,
    cols: usize,
    layout: LayoutKind,
) -> MatrixView<'a, f32> {
    match layout {
        LayoutKind::RowMajor => MatrixView::new(data, MatrixLayout::row_major(rows, cols)),
        LayoutKind::ColumnMajor => MatrixView::new(data, MatrixLayout::column_major(rows, cols)),
        LayoutKind::Transposed => {
            MatrixView::new(data, MatrixLayout::row_major(cols, rows)).transpose()
        }
    }
}

fn write_view<'a>(
    data: &'a mut [f32],
    rows: usize,
    cols: usize,
    layout: LayoutKind,
) -> MatrixViewMut<'a, f32> {
    match layout {
        LayoutKind::RowMajor => MatrixViewMut::new(data, MatrixLayout::row_major(rows, cols)),
        LayoutKind::ColumnMajor => MatrixViewMut::new(data, MatrixLayout::column_major(rows, cols)),
        LayoutKind::Transposed => {
            MatrixViewMut::new(data, MatrixLayout::row_major(cols, rows)).transpose()
        }
    }
}

fn bench_case(c: &mut Criterion, case: MatmulCase) {
    let mut group = c.benchmark_group(format!(
        "matmul/{}/{}-{}-{}/{}-{}-{}",
        case.name,
        case.rows,
        case.inner,
        case.cols,
        case.a_layout.label(),
        case.b_layout.label(),
        case.out_layout.label(),
    ));

    let a_data: Vec<f32> = (0..case.rows * case.inner).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (0..case.inner * case.cols).map(|x| x as f32).collect();
    let mut out_data = vec![0.0; case.rows * case.cols];

    let a = read_view(&a_data, case.rows, case.inner, case.a_layout);
    let b = read_view(&b_data, case.inner, case.cols, case.b_layout);
    let mut out = write_view(&mut out_data, case.rows, case.cols, case.out_layout);

    group.bench_function("naive_ijk", |bench| {
        bench.iter(|| {
            out.fill_zero();
            matmul_naive_ijk(&mut out, &a, &b);
            std::hint::black_box(&out);
        });
    });

    group.bench_function("optimized_loop_order", |bench| {
        bench.iter(|| {
            out.fill_zero();
            matmul_optimized_loop_order(&mut out, &a, &b);
            std::hint::black_box(&out);
        });
    });

    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    for case in [
        MatmulCase {
            name: "square",
            rows: 10,
            inner: 10,
            cols: 10,
            a_layout: LayoutKind::RowMajor,
            b_layout: LayoutKind::RowMajor,
            out_layout: LayoutKind::RowMajor,
        },
        MatmulCase {
            name: "square",
            rows: 100,
            inner: 100,
            cols: 100,
            a_layout: LayoutKind::RowMajor,
            b_layout: LayoutKind::RowMajor,
            out_layout: LayoutKind::RowMajor,
        },
        MatmulCase {
            name: "square",
            rows: 200,
            inner: 200,
            cols: 200,
            a_layout: LayoutKind::RowMajor,
            b_layout: LayoutKind::RowMajor,
            out_layout: LayoutKind::RowMajor,
        },
        MatmulCase {
            name: "rectangular",
            rows: 32,
            inner: 64,
            cols: 24,
            a_layout: LayoutKind::RowMajor,
            b_layout: LayoutKind::RowMajor,
            out_layout: LayoutKind::RowMajor,
        },
        MatmulCase {
            name: "column_a",
            rows: 32,
            inner: 64,
            cols: 24,
            a_layout: LayoutKind::ColumnMajor,
            b_layout: LayoutKind::RowMajor,
            out_layout: LayoutKind::RowMajor,
        },
        MatmulCase {
            name: "column_b",
            rows: 32,
            inner: 64,
            cols: 24,
            a_layout: LayoutKind::RowMajor,
            b_layout: LayoutKind::ColumnMajor,
            out_layout: LayoutKind::RowMajor,
        },
        MatmulCase {
            name: "column_all",
            rows: 32,
            inner: 64,
            cols: 24,
            a_layout: LayoutKind::ColumnMajor,
            b_layout: LayoutKind::ColumnMajor,
            out_layout: LayoutKind::ColumnMajor,
        },
        MatmulCase {
            name: "skinny",
            rows: 128,
            inner: 16,
            cols: 64,
            a_layout: LayoutKind::Transposed,
            b_layout: LayoutKind::ColumnMajor,
            out_layout: LayoutKind::Transposed,
        },
    ] {
        bench_case(c, case);
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_matmul
);
criterion_main!(benches);
