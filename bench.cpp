/**
 * HRP Benchmark — C++17 implementation
 * Compile: clang++ -O3 -std=c++17 -o hrp_bench bench.cpp
 *
 * Faithful port of the C reference (../c/bench.c). The algorithm is kept
 * bit-identical; only idiomatic C++ containers (std::vector) and timing
 * (std::chrono) are used.
 */

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cstring>
#include <chrono>

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

// ── Average linkage ──

struct LinkageRow { int i, j; double dist; int size; };

static void average_linkage(const std::vector<double> &dist_in,
                            std::vector<LinkageRow> &Z, int n) {
    int cap = 2 * n;
    std::vector<double> D((size_t)cap * cap, 1e18);
    std::vector<char> active(cap, 0);
    std::vector<int> sizes(cap, 0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) D[i * cap + j] = dist_in[i * n + j];
        active[i] = 1;
        sizes[i] = 1;
    }

    for (int step = 0; step < n - 1; step++) {
        double minD = 1e18;
        int mi = 0, mj = 0;
        for (int i = 0; i < n + step; i++) {
            if (!active[i]) continue;
            for (int j = i + 1; j < n + step; j++) {
                if (!active[j]) continue;
                if (D[i * cap + j] < minD) { minD = D[i * cap + j]; mi = i; mj = j; }
            }
        }
        int nid = n + step;
        sizes[nid] = sizes[mi] + sizes[mj];
        Z[step] = LinkageRow{ mi, mj, minD, sizes[nid] };

        for (int k = 0; k < nid; k++) {
            if (!active[k] || k == mi || k == mj) continue;
            double nd = (D[mi * cap + k] * sizes[mi] + D[mj * cap + k] * sizes[mj]) / sizes[nid];
            D[nid * cap + k] = nd;
            D[k * cap + nid] = nd;
        }
        D[nid * cap + nid] = 0;
        active[mi] = 0;
        active[mj] = 0;
        active[nid] = 1;
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

    std::vector<LinkageRow> Z(n - 1);
    t0 = now_us();
    average_linkage(dist, Z, n);
    double t_link = now_us() - t0;

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
