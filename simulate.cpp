// Exact insertion-position simulation of the coupled LCD and PORT profiles.
// Usage: simulate REPLICATES SEED OUTPUT.csv CONTROL_MEANS.csv FACTORS.csv
// Checkpoints are LCD global sizes; the coupled PORT has n+1 vertices.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

struct Moments {
    std::uint64_t count = 0;
    double mean = 0.0, m2 = 0.0;
    void add(double x) {
        ++count;
        double delta = x - mean;
        mean += delta / count;
        m2 += delta * (x - mean);
    }
    double se() const { return std::sqrt(m2 / (count - 1) / count); }
};

// Rejection sampling avoids the modulo bias of rng()%bound. The specified
// engine plus this sampler reproduces the integer stream across C++ libraries.
std::uint64_t bounded(std::mt19937_64& rng, std::uint64_t bound) {
    const std::uint64_t threshold = -bound % bound;
    for (;;) {
        auto r = rng();
        if (r >= threshold) return r % bound;
    }
}

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("usage: simulate REPLICATES SEED OUTPUT.csv CONTROL_MEANS.csv FACTORS.csv");
        const auto reps = std::stoull(argv[1]), seed = std::stoull(argv[2]);
        if (reps < 2) throw std::runtime_error("at least two replicates required");
        const std::array<int, 4> sizes{100, 1000, 10000, 100000};
        const int nmax = sizes.back();
        std::ofstream factors(argv[5]);
        if (!factors) throw std::runtime_error("cannot open factor output file");
        factors << "replicate,n,root_weight,branch_root_weight\n";
        using Key = std::tuple<int, int, int>;
        std::map<Key, Moments> stats, excess_stats;
        std::map<std::pair<int,int>, std::vector<double>> controls;
        std::ifstream control_file(argv[4]);
        if (!control_file) throw std::runtime_error("cannot open mean control file");
        std::string line;
        std::getline(control_file, line); // header
        while (std::getline(control_file, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream row(line);
            int n,H;
            row >> n >> H;
            std::vector<double> mu(H+1,0.0);
            for (int h=1; h<=H; ++h) row >> mu[h];
            if (!row) throw std::runtime_error("invalid mean control row");
            controls[{n,H}] = mu;
        }
        std::vector<std::uint32_t> slots(2 * nmax + 1);
        for (std::uint64_t rep = 0; rep < reps; ++rep) {
            std::mt19937_64 rng(seed + rep);
            // Code = 2*global_level + branch_flag. The root has two retained
            // positions; vertex 2 has one missing-branch position at level 1.
            slots[0] = 0; slots[1] = 0; slots[2] = 3;
            std::vector<int> retained(64, 0), missing(64, 0);
            retained[0] = 2; missing[1] = 1;
            int component_size = 1, checkpoint = 0;
            for (int current = 1; current < nmax; ++current) {
                const auto code = slots[bounded(rng, 2ULL * current + 1)];
                const auto depth = code / 2;
                auto& weights = (code & 1) ? missing : retained;
                if (depth + 1 >= weights.size()) weights.resize(depth + 2, 0);
                ++weights[depth]; ++weights[depth + 1];
                if (!(code & 1)) ++component_size;
                // Retain the chosen slot and append one at the parent and
                // one at the child: total attachment weight increases by two.
                slots[2 * current + 1] = code;
                slots[2 * current + 2] = code + 2;
                const int n = current + 1;
                if (n != sizes[checkpoint]) continue;
                const int sum_ret = std::accumulate(retained.begin(), retained.end(), 0);
                const int sum_miss = std::accumulate(missing.begin(), missing.end(), 0);
                if (sum_ret != 2 * component_size || sum_ret + sum_miss != 2 * n + 1 || missing[0])
                    throw std::runtime_error("profile invariant failed");
                const double L = std::log(n);
                std::set<int> cutoffs{1, 3, 6, std::max(1, int(std::floor(0.35 * L)))};
                for (int H : cutoffs) {
                    double scale = 1 / std::sqrt(n), sum = 0, squares = 0, maximum = 0;
                    const auto& mu = controls.at({n,H});
                    double mu_norm = 0, dot = 0, chosen_gap = 0;
                    int chosen = 1;
                    for (int h=1; h<=H; ++h) {
                        mu_norm += mu[h]*mu[h];
                        if (mu[h] > mu[chosen]) chosen = h;
                    }
                    mu_norm = std::sqrt(mu_norm);
                    for (int h = 1; h <= H; ++h) {
                        scale *= 2.0 * h / L;
                        const double gap = scale * missing[h];
                        sum += gap; squares += gap * gap; maximum = std::max(maximum, gap);
                        dot += gap*mu[h]/mu_norm;
                        if (h==chosen) chosen_gap = gap;
                    }
                    stats[{n,H,1}].add(sum);
                    stats[{n,H,2}].add(std::sqrt(squares));
                    stats[{n,H,0}].add(maximum); // 0 denotes infinity in the CSV.
                    double residual = std::sqrt(squares)-dot;
                    if (residual < -1e-10) throw std::runtime_error("negative residual beyond roundoff");
                    excess_stats[{n,H,1}].add(0.0);
                    excess_stats[{n,H,2}].add(std::max(0.0,residual));
                    excess_stats[{n,H,0}].add(maximum-chosen_gap);
                }
                ++checkpoint;
            }
            factors << rep << ',' << nmax << ',' << retained[0] << ',' << missing[1] << '\n';
            if ((rep + 1) % 4096 == 0) std::cerr << (rep + 1) << "/" << reps << " paths\n";
        }
        std::ofstream out(argv[3]);
        if (!out) throw std::runtime_error("cannot open output file");
        out << "n,H,p,replicates,mean,se,excess_mean,excess_se\n" << std::setprecision(17);
        for (const auto& [key, moments] : stats) {
            auto [n,H,p] = key;
            out << n << ',' << H << ',' << (p == 0 ? "inf" : std::to_string(p)) << ','
                << moments.count << ',' << moments.mean << ',' << moments.se() << ','
                << excess_stats.at(key).mean << ',' << excess_stats.at(key).se() << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
