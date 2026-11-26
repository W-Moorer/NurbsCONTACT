#pragma once
#include <cstddef>

namespace nc {
enum class PairBackend { occt, mesh, brep };

struct PairSearchOptions {
    double tolerance_m{2e-6};
    std::size_t max_nodes{30'000};
};

class PairGeometry;
} // namespace nc
