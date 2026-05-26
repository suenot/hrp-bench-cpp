# HRP Benchmark — C++17

A C++17 port of the C reference HRP (Hierarchical Risk Parity) benchmark. It is
a faithful, algorithm-identical translation used for cross-language performance
comparison.

## Build

```sh
make build          # clang++ -O3 -std=c++17 -o hrp_bench bench.cpp
# or directly:
clang++ -O3 -std=c++17 -o hrp_bench bench.cpp
```

## Run

```sh
make run            # builds then runs ./hrp_bench
# or:
./hrp_bench
```

## What it measures

For each universe size `N` in `{10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000}`,
the program generates 365 daily synthetic prices per asset (via a 64-bit LCG;
generation is **not** timed) and runs the HRP pipeline. Five stages are timed
with `std::chrono::steady_clock` (microseconds) and summed into `TOTAL`:

1. **LogRet**  — log returns
2. **Cov**     — sample covariance (divide by T-1)
3. **Linkage** — average-linkage hierarchical clustering
4. **QuasiD**  — quasi-diagonalization (reorder by leaf order)
5. **Weights** — HRP recursive bisection weights

Correlation and distance matrices are computed but, matching the C reference,
are **not** timed.

Results print as a table on STDOUT. A `verify N=10:` line with the first three
HRP weights and their sum is printed to STDERR for cross-language checking.

## Complexity note

Average linkage uses the **O(n²)** nearest-neighbour-chain algorithm (Müllner
2011), exactly as in the C reference. Clustering is therefore no longer the
bottleneck; for large `N` the runtime is dominated by the O(n²·T) covariance
stage.
