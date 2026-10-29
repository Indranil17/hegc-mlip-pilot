// Stand-alone tests for hegc_core.hpp: brute-force reference over periodic images, no Python needed.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "hegc_core.hpp"

static int failures = 0;
#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL  %s  (%s:%d)\n", msg, __FILE__, __LINE__);         \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

// Reference: every unordered pair and every image in a generous shift range, same acceptance rule as the kernel.
static std::vector<double> brute_force(const std::vector<double>& pos, int n, const double* cell, double cutoff, int span) {
    std::vector<double> d;
    for (int a = 0; a < n; ++a)
        for (int b = a; b < n; ++b)
            for (int sx = -span; sx <= span; ++sx)
                for (int sy = -span; sy <= span; ++sy)
                    for (int sz = -span; sz <= span; ++sz) {
                        if (b == a && !(sx > 0 || (sx == 0 && (sy > 0 || (sy == 0 && sz > 0))))) continue;
                        double v[3];
                        for (int c = 0; c < 3; ++c)
                            v[c] = pos[3 * b + c] - pos[3 * a + c] + sx * cell[c] + sy * cell[3 + c] + sz * cell[6 + c];
                        const double r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                        if (r <= cutoff && !(a == b && sx == 0 && sy == 0 && sz == 0)) d.push_back(r);
                    }
    std::sort(d.begin(), d.end());
    return d;
}

static void test_random_cells() {
    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double cells[3][9] = {
        {9.0, 0, 0, 0, 10.0, 0, 0, 0, 11.0},                        // orthorhombic
        {8.0, 0, 0, 3.0, 9.0, 0, 1.0, 2.0, 10.0},                   // triclinic
        {6.0, 0, 0, -2.5, 6.5, 0, 1.5, -1.0, 7.0},                  // small triclinic, cutoff above half the cell
    };
    const double cutoffs[3] = {3.1, 4.4, 8.5};
    for (int t = 0; t < 3; ++t) {
        const int n = 30;
        std::vector<double> pos(3 * n);
        for (int a = 0; a < n; ++a) {
            // random fractional coordinates, including values outside [0, 1) to test wrapping
            const double f[3] = {u(gen) * 3.0 - 1.0, u(gen) * 3.0 - 1.0, u(gen) * 3.0 - 1.0};
            for (int c = 0; c < 3; ++c) pos[3 * a + c] = f[0] * cells[t][c] + f[1] * cells[t][3 + c] + f[2] * cells[t][6 + c];
        }
        const hegc::Box box = hegc::make_box(cells[t]);
        // wrap the reference positions the same way (a lattice translation does not change any distance set)
        const hegc::Pairs p = hegc::neighbor_pairs(pos.data(), n, box, cutoffs[t]);
        std::vector<double> got(p.d.begin(), p.d.end());
        std::sort(got.begin(), got.end());
        const std::vector<double> ref = brute_force(pos, n, cells[t], cutoffs[t], 6);
        char msg[160];
        std::snprintf(msg, sizeof msg, "cell %d: pair count %zu vs brute force %zu", t, got.size(), ref.size());
        CHECK(got.size() == ref.size(), msg);
        if (got.size() == ref.size()) {
            double worst = 0.0;
            for (std::size_t q = 0; q < got.size(); ++q) worst = std::max(worst, std::fabs(got[q] - ref[q]));
            std::snprintf(msg, sizeof msg, "cell %d: distances differ by up to %.3e", t, worst);
            CHECK(worst < 1e-10, msg);
        }
        std::printf("cell %d  cutoff %.1f  pairs %zu\n", t, cutoffs[t], got.size());
    }
}

static void test_histogram_conserves_pairs() {
    const double cell[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> u(0.0, 10.0);
    const int n = 200;
    std::vector<double> pos(3 * n);
    std::vector<int32_t> types(n);
    for (int a = 0; a < n; ++a) {
        for (int c = 0; c < 3; ++c) pos[3 * a + c] = u(gen);
        types[a] = a % 3;
    }
    const hegc::Box box = hegc::make_box(cell);
    const auto h = hegc::pair_histogram(pos.data(), types.data(), n, box, 4.0, 40, 3);
    long long total = 0;
    for (long long v : h) total += v;
    const hegc::Pairs p = hegc::neighbor_pairs(pos.data(), n, box, 4.0);
    long long inside = 0;
    for (double d : p.d) inside += (d > 0.0 && d < 4.0);
    CHECK(total == inside, "histogram total equals the number of pairs below rmax");
    // only t1 <= t2 may be filled
    long long lower = 0;
    for (int t1 = 0; t1 < 3; ++t1)
        for (int t2 = 0; t2 < t1; ++t2)
            for (int k = 0; k < 40; ++k) lower += h[(t1 * 3 + t2) * 40 + k];
    CHECK(lower == 0, "no counts below the type-pair diagonal");
}

static void test_connectivity_chain() {
    // Four Si in a row bridged by three O, then a gap, then one isolated Si with its own O. Cubic cell, no wrap-around.
    const double cell[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
    const int32_t Si = 0, O = 1;
    std::vector<double> pos;
    std::vector<int32_t> types;
    auto add = [&](double x, double y, double z, int32_t t) {
        pos.insert(pos.end(), {x, y, z});
        types.push_back(t);
    };
    for (int k = 0; k < 4; ++k) add(3.2 * k, 0, 0, Si);
    for (int k = 0; k < 3; ++k) add(3.2 * k + 1.6, 0, 0, O);
    add(20, 20, 20, Si);
    add(21.6, 20, 20, O);
    const hegc::Box box = hegc::make_box(cell);
    const auto c = hegc::connectivity(pos.data(), types.data(), static_cast<int>(types.size()), box, {2.0, 0.0}, O);
    CHECK(c.cluster_size.size() == 2, "two clusters");
    std::vector<int> sz(c.cluster_size.begin(), c.cluster_size.end());
    std::sort(sz.begin(), sz.end());
    CHECK(sz.size() == 2 && sz[0] == 1 && sz[1] == 4, "cluster sizes 1 and 4");
    int bridging = 0;
    for (std::size_t a = 0; a < types.size(); ++a) bridging += (types[a] == O && c.n_centres[a] == 2);
    CHECK(bridging == 3, "three bridging oxygens");
    CHECK(c.cn[0] == 1 && c.cn[1] == 2, "coordination numbers along the chain end and middle");
}

int main() {
    test_random_cells();
    test_histogram_conserves_pairs();
    test_connectivity_chain();
    if (failures == 0) std::printf("ALL PASSED\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
