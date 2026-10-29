// hegc_core.hpp -- periodic cell-list kernels for analysing MLIP trajectories of oxide glasses.
//
// Header-only, C++17, no dependencies. The stand-alone tests (tests/test_core.cpp) include this file.
//
// What it does
//   * neighbor_pairs: every unordered pair of atoms closer than a cutoff in a fully periodic,
//     possibly triclinic cell, each periodic image counted once. Cost is O(N) for a fixed cutoff.
//   * pair_histogram: partial pair-distance histograms by atom type (the input of a g(r)).
//   * connectivity: centre atoms (e.g. Si, B, Al, Mo) joined through shared ligand atoms (O),
//     grouped with union-find; also the number of centres bonded to each ligand.
//
// Conventions
//   * cell rows are the lattice vectors; positions are Cartesian; all three directions periodic.
//   * A pair (i, j, shift) is reported once. For i < j every image within the cutoff is reported.
//     For i == j only one of each +/- pair of self-images is reported (the lexicographically
//     positive shift), so a cutoff larger than the cell does not double count.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace hegc {

struct Box {
    double h[3][3];      // lattice vectors as rows
    double hinv[3][3];   // inverse of h, so that fractional = cartesian * hinv
    double height[3];    // distance between the pair of faces spanned by the two other vectors
    double volume;
};

inline Box make_box(const double* cell9) {
    Box b{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) b.h[r][c] = cell9[3 * r + c];
    const auto& m = b.h;
    const double c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    const double c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    const double c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    const double det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    if (std::fabs(det) < 1e-12) throw std::invalid_argument("hegc: singular cell");
    b.volume = std::fabs(det);
    const double id = 1.0 / det;
    b.hinv[0][0] = c00 * id;
    b.hinv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id;
    b.hinv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id;
    b.hinv[1][0] = c01 * id;
    b.hinv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id;
    b.hinv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id;
    b.hinv[2][0] = c02 * id;
    b.hinv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id;
    b.hinv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id;
    for (int k = 0; k < 3; ++k) {
        const int a = (k + 1) % 3, c = (k + 2) % 3;
        const double cx = m[a][1] * m[c][2] - m[a][2] * m[c][1];
        const double cy = m[a][2] * m[c][0] - m[a][0] * m[c][2];
        const double cz = m[a][0] * m[c][1] - m[a][1] * m[c][0];
        const double area = std::sqrt(cx * cx + cy * cy + cz * cz);
        b.height[k] = b.volume / area;
    }
    return b;
}

struct Pairs {
    std::vector<int32_t> i, j;
    std::vector<double> d;
    std::size_t size() const { return d.size(); }
};

inline int floor_div(int a, int n) { return a >= 0 ? a / n : -((-a + n - 1) / n); }

// All pairs closer than `cutoff`, each periodic image once (see header comment).
inline Pairs neighbor_pairs(const double* pos, int n, const Box& box, double cutoff) {
    if (cutoff <= 0.0) throw std::invalid_argument("hegc: cutoff must be positive");
    Pairs out;
    if (n <= 0) return out;

    // wrapped fractional coordinates
    std::vector<double> f(3 * static_cast<std::size_t>(n));
    for (int a = 0; a < n; ++a) {
        for (int c = 0; c < 3; ++c) {
            double v = pos[3 * a] * box.hinv[0][c] + pos[3 * a + 1] * box.hinv[1][c] + pos[3 * a + 2] * box.hinv[2][c];
            v -= std::floor(v);
            f[3 * a + c] = (v >= 1.0) ? 0.0 : v;
        }
    }

    int nb[3], reach[3];
    for (int k = 0; k < 3; ++k) {
        nb[k] = std::max(1, static_cast<int>(std::floor(box.height[k] / cutoff)));
        const double width = box.height[k] / nb[k];
        reach[k] = static_cast<int>(std::ceil(cutoff / width));
    }
    const std::size_t nbins = static_cast<std::size_t>(nb[0]) * nb[1] * nb[2];

    // counting sort of atoms into bins
    std::vector<int> bin_of(n);
    std::vector<int> start(nbins + 1, 0);
    for (int a = 0; a < n; ++a) {
        int ix = std::min(nb[0] - 1, static_cast<int>(f[3 * a] * nb[0]));
        int iy = std::min(nb[1] - 1, static_cast<int>(f[3 * a + 1] * nb[1]));
        int iz = std::min(nb[2] - 1, static_cast<int>(f[3 * a + 2] * nb[2]));
        bin_of[a] = (ix * nb[1] + iy) * nb[2] + iz;
        ++start[bin_of[a] + 1];
    }
    for (std::size_t b = 0; b < nbins; ++b) start[b + 1] += start[b];
    std::vector<int> order(n), fill(start.begin(), start.end() - 1);
    for (int a = 0; a < n; ++a) order[fill[bin_of[a]]++] = a;

    const double cut2 = cutoff * cutoff;
    for (int a = 0; a < n; ++a) {
        const int ba = bin_of[a];
        const int iz = ba % nb[2], iy = (ba / nb[2]) % nb[1], ix = ba / (nb[1] * nb[2]);
        for (int dx = -reach[0]; dx <= reach[0]; ++dx) {
            const int tx = ix + dx, sx = floor_div(tx, nb[0]), wx = tx - sx * nb[0];
            for (int dy = -reach[1]; dy <= reach[1]; ++dy) {
                const int ty = iy + dy, sy = floor_div(ty, nb[1]), wy = ty - sy * nb[1];
                for (int dz = -reach[2]; dz <= reach[2]; ++dz) {
                    const int tz = iz + dz, sz = floor_div(tz, nb[2]), wz = tz - sz * nb[2];
                    const int tb = (wx * nb[1] + wy) * nb[2] + wz;
                    for (int q = start[tb]; q < start[tb + 1]; ++q) {
                        const int b = order[q];
                        if (b < a) continue;
                        if (b == a && !(sx > 0 || (sx == 0 && (sy > 0 || (sy == 0 && sz > 0))))) continue;
                        const double fx = f[3 * b] + sx - f[3 * a];
                        const double fy = f[3 * b + 1] + sy - f[3 * a + 1];
                        const double fz = f[3 * b + 2] + sz - f[3 * a + 2];
                        const double x = fx * box.h[0][0] + fy * box.h[1][0] + fz * box.h[2][0];
                        const double y = fx * box.h[0][1] + fy * box.h[1][1] + fz * box.h[2][1];
                        const double z = fx * box.h[0][2] + fy * box.h[1][2] + fz * box.h[2][2];
                        const double r2 = x * x + y * y + z * z;
                        if (r2 <= cut2) {
                            out.i.push_back(a);
                            out.j.push_back(b);
                            out.d.push_back(std::sqrt(r2));
                        }
                    }
                }
            }
        }
    }
    return out;
}

// Partial pair-distance histograms. counts[(t1 * ntypes + t2) * nbins + k] holds the number of
// unordered pairs whose types are {t1, t2} with t1 <= t2 and whose distance falls in bin k of
// [0, rmax). Pairs at distance exactly 0 are ignored, as in a g(r) that excludes self pairs.
inline std::vector<int64_t> pair_histogram(const double* pos, const int32_t* types, int n, const Box& box,
                                           double rmax, int nbins, int ntypes) {
    std::vector<int64_t> counts(static_cast<std::size_t>(ntypes) * ntypes * nbins, 0);
    const Pairs p = neighbor_pairs(pos, n, box, rmax);
    const double inv = nbins / rmax;
    for (std::size_t q = 0; q < p.size(); ++q) {
        if (p.d[q] <= 0.0 || p.d[q] >= rmax) continue;
        int t1 = types[p.i[q]], t2 = types[p.j[q]];
        if (t1 > t2) std::swap(t1, t2);
        if (t1 < 0 || t2 >= ntypes) throw std::out_of_range("hegc: atom type out of range");
        const int k = std::min(nbins - 1, static_cast<int>(p.d[q] * inv));
        ++counts[(static_cast<std::size_t>(t1) * ntypes + t2) * nbins + k];
    }
    return counts;
}

struct DisjointSet {
    std::vector<int> parent, size;
    explicit DisjointSet(int n) : parent(n), size(n, 1) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (size[a] < size[b]) std::swap(a, b);
        parent[b] = a;
        size[a] += size[b];
    }
};

struct Connectivity {
    std::vector<int32_t> cluster;      // per atom: cluster id of a centre atom, -1 for every other atom
    std::vector<int32_t> cluster_size; // per cluster id: number of centre atoms
    std::vector<int32_t> n_centres;    // per atom: number of centre atoms bonded to this ligand atom, 0 for non-ligands
    std::vector<int32_t> cn;           // per atom: number of ligand atoms bonded to this centre atom, 0 for non-centres
};

// Centres are atoms whose type has centre_cutoff[type] > 0; a ligand is an atom of type ligand_type.
// A centre and a ligand are bonded when closer than the centre's cutoff. Two centres belong to the
// same cluster when they share at least one ligand atom (for Si-O-Si this is a bridging oxygen).
inline Connectivity connectivity(const double* pos, const int32_t* types, int n, const Box& box,
                                 const std::vector<double>& centre_cutoff, int ligand_type) {
    double cmax = 0.0;
    for (double c : centre_cutoff) cmax = std::max(cmax, c);
    Connectivity out;
    out.cluster.assign(n, -1);
    out.n_centres.assign(n, 0);
    out.cn.assign(n, 0);
    if (cmax <= 0.0) return out;
    const int ntypes = static_cast<int>(centre_cutoff.size());
    const Pairs p = neighbor_pairs(pos, n, box, cmax);

    DisjointSet ds(n);
    std::vector<int> first_centre(n, -1);  // first centre seen bonded to each ligand
    for (std::size_t q = 0; q < p.size(); ++q) {
        int a = p.i[q], b = p.j[q];
        int c = -1, l = -1;
        const int ta = types[a], tb = types[b];
        if (ta < ntypes && centre_cutoff[ta] > 0.0 && tb == ligand_type && p.d[q] <= centre_cutoff[ta]) { c = a; l = b; }
        else if (tb < ntypes && centre_cutoff[tb] > 0.0 && ta == ligand_type && p.d[q] <= centre_cutoff[tb]) { c = b; l = a; }
        if (c < 0) continue;
        ++out.cn[c];
        ++out.n_centres[l];
        if (first_centre[l] < 0) first_centre[l] = c;
        else ds.unite(first_centre[l], c);
    }
    std::vector<int> id_of_root(n, -1);
    for (int a = 0; a < n; ++a) {
        const int t = types[a];
        if (t < ntypes && centre_cutoff[t] > 0.0) {
            const int r = ds.find(a);
            if (id_of_root[r] < 0) {
                id_of_root[r] = static_cast<int>(out.cluster_size.size());
                out.cluster_size.push_back(0);
            }
            out.cluster[a] = id_of_root[r];
            ++out.cluster_size[id_of_root[r]];
        }
    }
    return out;
}

}  // namespace hegc
