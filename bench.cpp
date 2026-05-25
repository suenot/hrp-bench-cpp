/**
 * HRP Benchmark — C++17 implementation
 * Compile: clang++ -O3 -std=c++17 -o hrp_bench bench.cpp fastcluster.cpp
 *
 * Faithful port of the C reference (../c/bench.c), except that the
 * average-linkage stage uses Daniel Müllner's fastcluster library
 * (the O(n²) NN-chain algorithm, the same one SciPy uses) instead of a
 * hand-rolled O(n³) linkage. All other stages (RNG, log returns,
 * covariance, correlation, distance, leaf order, quasi-diag, HRP weights)
 * are unchanged.
 */

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cstring>
#include <chrono>

#include "fastcluster.h"

// ── Timing ──

static double now_us() {
    using namespace std::chrono;
    auto t = steady_clock::now().time_since_epoch();
    return duration_cast<duration<double, std::micro>>(t).count();
}

// ── Generate synthetic prices ──

static void generate_prices(std::vector<double> &prices, int n, int days) {
    uint64_t seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 6364136223846793005ULL + 1;
        double start = 100.0 + (seed % 900);
        prices[i * days] = start;
        double vol = 0.01 + (seed % 50) * 0.001;
        for (int t = 1; t < days; t++) {
            seed = seed * 6364136223846793005ULL + 1;
            double u = ((double)seed / (double)0xFFFFFFFFFFFFFFFFULL) - 0.5;
            double ret = vol * u * 0.816;
            prices[i * days + t] = prices[i * days + t - 1] * std::exp(ret);
        }
    }
}

// ── Log returns ──

static void log_returns(const std::vector<double> &prices, std::vector<double> &rets,
                        int n, int days) {
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < days - 1; t++) {
            rets[i * (days - 1) + t] =
                std::log(prices[i * days + t + 1] / prices[i * days + t]);
        }
    }
}

// ── Covariance matrix ──

static void cov_matrix(const std::vector<double> &rets, std::vector<double> &cov,
                       int n, int T) {
    std::vector<double> means(n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < T; t++) means[i] += rets[i * T + t];
        means[i] /= T;
    }
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double s = 0;
            for (int t = 0; t < T; t++)
                s += (rets[i * T + t] - means[i]) * (rets[j * T + t] - means[j]);
            cov[i * n + j] = cov[j * n + i] = s / (T - 1);
        }
    }
}

// ── Correlation & Distance ──

static void corr_matrix(const std::vector<double> &cov, std::vector<double> &corr, int n) {
    std::vector<double> stds(n);
    for (int i = 0; i < n; i++) stds[i] = std::sqrt(cov[i * n + i]);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            corr[i * n + j] = (stds[i] > 0 && stds[j] > 0)
                                  ? cov[i * n + j] / (stds[i] * stds[j])
                                  : 0;
}

static void dist_matrix(const std::vector<double> &corr, std::vector<double> &dist, int n) {
    for (int i = 0; i < n * n; i++) {
        double v = (1.0 - corr[i]) / 2.0;
        dist[i] = std::sqrt(v > 0 ? v : 0);
    }
}

// ── Average linkage (fastcluster, Müllner NN-chain) ──

struct LinkageRow { int i, j; double dist; int size; };

// Build the condensed (upper-triangular, no diagonal) distance vector that
// fastcluster's hclust_fast expects: length n*(n-1)/2. NOT timed — matches
// how the C reference times only the linkage core.
static void condensed_distance(const std::vector<double> &dist_in,
                               std::vector<double> &condensed, int n) {
    size_t k = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            condensed[k++] = dist_in[(size_t)i * n + j];
}

// Convert fastcluster's R-format merge/height output into the SciPy-style
// linkage matrix the leaf_order / quasi-diag stages expect.
//
// fastcluster (R hclust convention), for merge step s (0-based):
//   child = merge[s] (and merge[(n-1)+s] for the second child)
//     negative -x  -> singleton leaf with 0-based id (x-1)
//     positive  c  -> the cluster formed at merge step (c-1)
//   height[s] = merge distance, steps are in non-decreasing height order.
//
// SciPy convention used downstream:
//   leaves are node ids 0..n-1; the cluster formed at step s is node n+s;
//   children stored with i < j; size is the leaf count of the merged cluster.
static void merge_to_linkage(const int *merge, const double *height,
                             std::vector<LinkageRow> &Z, int n) {
    std::vector<int> csize(n - 1, 0);  // leaf count of cluster formed at each step
    for (int s = 0; s < n - 1; s++) {
        int m1 = merge[s];
        int m2 = merge[(n - 1) + s];
        int a = (m1 < 0) ? (-m1 - 1) : (n + (m1 - 1));
        int b = (m2 < 0) ? (-m2 - 1) : (n + (m2 - 1));
        int sa = (m1 < 0) ? 1 : csize[m1 - 1];
        int sb = (m2 < 0) ? 1 : csize[m2 - 1];
        csize[s] = sa + sb;
        int lo = a < b ? a : b, hi = a < b ? b : a;
        Z[s] = LinkageRow{ lo, hi, height[s], csize[s] };
    }
}

// ── Leaf order (iterative) ──

static void leaf_order(const std::vector<LinkageRow> &Z, int n,
                       std::vector<int> &order, int &out_len) {
    std::vector<int> stack(2 * n);
    int sp = 0, cnt = 0;
    stack[sp++] = n + (n - 2); // root
    while (sp > 0) {
        int node = stack[--sp];
        if (node < n) { order[cnt++] = node; continue; }
        const LinkageRow &r = Z[node - n];
        stack[sp++] = r.j;
        stack[sp++] = r.i;
    }
    out_len = cnt;
}

// ── Quasi-diag ──

static void quasi_diag(const std::vector<double> &cov, const std::vector<int> &order,
                       std::vector<double> &qd, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            qd[i * n + j] = cov[order[i] * n + order[j]];
}

// ── HRP weights ──

static double cluster_var(const std::vector<double> &cov, const int *idx, int m, int n) {
    double v = 0;
    for (int a = 0; a < m; a++)
        for (int b = 0; b < m; b++)
            v += cov[idx[a] * n + idx[b]];
    return v / ((double)m * m);
}

static void hrp_weights(const std::vector<double> &covQ, int n, std::vector<double> &w) {
    for (int i = 0; i < n; i++) w[i] = 1.0;

    // BFS-style bisection
    struct Segment { std::vector<int> idx; };
    std::vector<Segment> queue;
    queue.reserve(2 * n);

    Segment all;
    all.idx.resize(n);
    for (int i = 0; i < n; i++) all.idx[i] = i;
    queue.push_back(std::move(all));

    size_t qh = 0;
    while (qh < queue.size()) {
        Segment seg = std::move(queue[qh++]);
        int len = (int)seg.idx.size();
        if (len <= 1) continue;
        int mid = len / 2;

        double vL = cluster_var(covQ, seg.idx.data(), mid, n);
        double vR = cluster_var(covQ, seg.idx.data() + mid, len - mid, n);
        double alpha = (1.0 / vL) / (1.0 / vL + 1.0 / vR);

        for (int i = 0; i < mid; i++) w[seg.idx[i]] *= alpha;
        for (int i = mid; i < len; i++) w[seg.idx[i]] *= (1.0 - alpha);

        Segment left, right;
        left.idx.assign(seg.idx.begin(), seg.idx.begin() + mid);
        right.idx.assign(seg.idx.begin() + mid, seg.idx.end());

        queue.push_back(std::move(left));
        queue.push_back(std::move(right));
    }

    double sum = 0;
    for (int i = 0; i < n; i++) sum += w[i];
    for (int i = 0; i < n; i++) w[i] /= sum;
}

// ── Format time ──

static void fmt_time(double us, char *buf) {
    if (us < 1000) snprintf(buf, 32, "%6.0fµs", us);
    else if (us < 1e6) snprintf(buf, 32, "%6.1fms", us / 1e3);
    else snprintf(buf, 32, "%6.2fs ", us / 1e6);
}

// ── Bench ──

static void bench(int n, int days, bool verify) {
    std::vector<double> prices((size_t)n * days);
    generate_prices(prices, n, days);

    int T = days - 1;
    std::vector<double> rets((size_t)n * T);
    double t0 = now_us();
    log_returns(prices, rets, n, days);
    double t_ret = now_us() - t0;

    std::vector<double> cov((size_t)n * n);
    t0 = now_us();
    cov_matrix(rets, cov, n, T);
    double t_cov = now_us() - t0;

    std::vector<double> corr((size_t)n * n);
    corr_matrix(cov, corr, n);

    std::vector<double> dist((size_t)n * n);
    dist_matrix(corr, dist, n);

    // Condensed-distance prep is OUTSIDE the timer (the C reference times
    // only the linkage core; fastcluster consumes a condensed vector).
    std::vector<LinkageRow> Z(n - 1);
    std::vector<double> condensed((size_t)n * (n - 1) / 2);
    condensed_distance(dist, condensed, n);
    std::vector<int> merge(2 * (n - 1));
    std::vector<double> height(n - 1);

    t0 = now_us();
    hclust_fast(n, condensed.data(), HCLUST_METHOD_AVERAGE, merge.data(), height.data());
    double t_link = now_us() - t0;

    merge_to_linkage(merge.data(), height.data(), Z, n);

    std::vector<int> order(n);
    int olen;
    leaf_order(Z, n, order, olen);

    std::vector<double> qd((size_t)n * n);
    t0 = now_us();
    quasi_diag(cov, order, qd, n);
    double t_qd = now_us() - t0;

    std::vector<double> w(n);
    t0 = now_us();
    hrp_weights(qd, n, w);
    double t_w = now_us() - t0;

    double total = t_ret + t_cov + t_link + t_qd + t_w;

    char s1[32], s2[32], s3[32], s4[32], s5[32], s6[32];
    fmt_time(t_ret, s1); fmt_time(t_cov, s2); fmt_time(t_link, s3);
    fmt_time(t_qd, s4); fmt_time(t_w, s5); fmt_time(total, s6);
    printf("  %6d │ %s │ %s │ %s │ %s │ %s │ %s\n", n, s1, s2, s3, s4, s5, s6);

    if (verify) {
        fprintf(stderr, "verify N=%d: w0=%.8g w1=%.8g w2=%.8g sum=%.8g\n",
                n, w[0], w[1], w[2], w[0] + w[1] + w[2]);
    }
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║          HRP Benchmark — C++ (clang++ -O3)                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    printf("  365 daily observations per asset\n\n");
    printf("  %6s │ %8s │ %8s │ %8s │ %8s │ %8s │ %8s\n",
        "N", "LogRet", "Cov", "Linkage", "QuasiD", "Weights", "TOTAL");
    printf("  ───────────────────────────────────────────────────────────────────\n");

    int sizes[] = {10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000};
    int count = sizeof(sizes) / sizeof(sizes[0]);
    for (int i = 0; i < count; i++) bench(sizes[i], 365, sizes[i] == 10);
    return 0;
}
