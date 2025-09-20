/* MIT License
 *
 * Copyright (c) 2025 Andrew D Smith
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "CLI11.hpp"

#include <cassert>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <print>
#include <random>
#include <ranges>
#include <vector>

typedef std::mt19937 random_engine;

struct count_pair {
  std::uint32_t n_meth{};
  std::uint32_t n_reads{};
};

template <> struct std::formatter<count_pair> : std::formatter<std::string> {
  auto
  format(const count_pair &x, auto &ctx) const {
    return std::formatter<std::string>::format(
      std::format("{}\t{}", x.n_meth, x.n_reads), ctx);
  }
};

struct beta_binomial_sampler {
  std::gamma_distribution<double> gamma_a;
  std::gamma_distribution<double> gamma_b;
  std::poisson_distribution<std::uint32_t> n_reads_dist;

  beta_binomial_sampler(const double alpha, const double beta,
                        const double n_reads_lambda) :
    gamma_a{alpha, 1.0}, gamma_b{beta, 1.0}, n_reads_dist{n_reads_lambda} {}

  [[nodiscard]] auto
  operator()(random_engine &rng) -> count_pair {
    const auto n_reads = n_reads_dist(rng) + 1;
    const auto x = gamma_a(rng);
    const auto y = gamma_b(rng);
    const auto p_star = x / (x + y);
    std::binomial_distribution<std::uint32_t> n_meth_dist(n_reads, p_star);
    return {n_reads, n_meth_dist(rng)};
  }
};

[[nodiscard]] static auto
write_design_matrix(const std::string &filename, const auto &group_indicator) {
  std::ofstream out(filename);
  if (!out)
    throw std::runtime_error("failed to open design output file: " + filename);
  std::println(out, "{}\t{}", "intercept", "factor");
  for (auto i = 0u; i < std::size(group_indicator); ++i)
    std::println(out, "Sample{}\t1\t{}", i, group_indicator[i]);
}

[[nodiscard]] static auto
write_data_matrix_header(std::ofstream &out, const auto &n_individuals,
                         const std::string &colname_prefix) {
  std::uint32_t out_count = 0;
  const auto gen_label = [&out_count, &colname_prefix]() {
    const auto label = (out_count > 0 ? "\t" : "") + colname_prefix;
    return std::format("{}{}", label, std::to_string(out_count++));
  };
  std::ranges::generate_n(std::ostream_iterator<std::string>(out),
                          n_individuals, gen_label);
  std::println(out);
}

static auto
simulate_group(random_engine &rng, const double lambda_reads,
               const std::uint32_t n_individuals, const double p,
               const double phi, std::vector<count_pair> &data) {
  const double alpha = p * phi;
  const double beta = (1.0 - p) * phi;
  beta_binomial_sampler bss(alpha, beta, lambda_reads);
  const auto generator = [&]() { return bss(rng); };
  std::generate_n(std::back_inserter(data), n_individuals, generator);
}

int
main(int argc, char *argv[]) {  // NOLINT(*-c-arrays)
  static constexpr auto command = "radmethsim";
  static const auto usage = std::format("Usage: radmethsim [options]", command);

  std::string outfile;
  std::string design_file;
  std::string rowname = "X";

  // User parameters
  std::uint32_t n_group0 = 30;
  std::uint32_t n_group1 = 30;

  std::uint32_t n_rows = 1;

  double lambda_reads = 10.0;

  double p0 = 0.5;
  double p1 = 0.5;

  // Shared dispersion parameter
  double phi = 1.0;

  CLI::App app{};
  argv = app.ensure_utf8(argv);
  app.usage(usage);

  // clang-format off
  app.set_help_flag("-h,--help", "print a detailed help message and exit");
  app.add_option("--n-group0", n_group0, "size of group 1");
  app.add_option("--n-group1", n_group1, "size of group 2");
  app.add_option("--p0", p0, "p group 0");
  app.add_option("--p1", p1, "p group 1");
  app.add_option("--reads", lambda_reads, "mean reads per site");
  app.add_option("-r,--n-rows", n_rows, "number of rows to generate");
  app.add_option("-d,--dispersion", phi, "dispersion parameter");
  app.add_option("-o,--output", outfile, "data output file")->required();
  app.add_option("-D,--design", design_file, "design output file")->required();
  app.add_option("--row-prefix", rowname, "rowname prefix");
  // clang-format on

  if (argc < 2) {
    std::println("{}", app.help());
    return EXIT_SUCCESS;
  }
  CLI11_PARSE(app, argc, argv);

  std::random_device rd;
  random_engine rng(rd());

  std::ofstream out(outfile);
  if (!out)
    throw std::runtime_error("failed to open data output file: " + outfile);

  const auto n_individuals = n_group0 + n_group1;
  write_data_matrix_header(out, n_individuals, "Sample");

  auto data = std::vector<count_pair>(n_individuals);

  for (auto i = 0u; i < n_rows; ++i) {
    data.clear();
    simulate_group(rng, lambda_reads, n_group0, p0, phi, data);
    simulate_group(rng, lambda_reads, n_group1, p1, phi, data);

    std::print(out, "{}", rowname);
    for (auto j = 0u; j < std::size(data); ++j)
      std::print(out, "\t{}", data[j]);
    std::println(out);
  }

  std::vector<std::uint32_t> group_indicators(n_group0, 0);
  group_indicators.insert(std::end(group_indicators), n_group1, 1);

  write_design_matrix(design_file, group_indicators);

  return EXIT_SUCCESS;
}
