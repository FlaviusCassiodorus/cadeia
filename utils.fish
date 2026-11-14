# Run all Criterion benchmarks, save a baseline and archive the report.
function bench-save
    set -l name $argv[1]

    if test -z "$name"
        echo "Usage: bench-save <baseline-name>"
        return 1
    end

    set -l commit (git rev-parse --short HEAD)
    or return 1

    echo "Saving baseline: $name"
    cargo bench --bench matmul_bench --bench mlp_bench -- --save-baseline "$name"
    or return 1

    set -l archive "benchmark-reports/base/$name/$commit"
    mkdir -p "$archive"
    or return 1

    cp -a target/criterion/. "$archive/"
    or return 1

    echo "$commit" > "$archive/commit.txt"
    or return 1

    echo "Baseline archived at $archive/"
end


# Run all Criterion benchmarks against a baseline and archive the comparison.
function bench-compare
    set -l name $argv[1]

    if test -z "$name"
        echo "Usage: bench-compare <baseline-name>"
        return 1
    end

    set -l commit (git rev-parse --short HEAD)
    or return 1

    echo "Comparing against baseline: $name"

    cargo bench --bench matmul_bench --bench mlp_bench -- --baseline "$name"
    or return 1

    set -l archive "benchmark-reports/compare/$name/$commit"
    mkdir -p "$archive"
    or return 1

    cp -a target/criterion/. "$archive/"
    or return 1

    echo "$commit" > "$archive/commit.txt"
    or return 1

    echo "Comparison archived at $archive/"
end
