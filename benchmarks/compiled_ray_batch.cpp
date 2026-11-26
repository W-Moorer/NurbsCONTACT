// Run identical FP64 rays through the production GPU CAD exit predicate.
// CUDA events time resident batches after a device warmup.
#include "../src/compiled_geometry.hpp"
#include "../src/cuda_columns.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
  using namespace nc::compiled;
  try {
    if (argc != 7)
      throw std::invalid_argument("expected part.ncpart rays.bin reach_mm "
                                  "repeats exits.bin rays_per_launch");
    const auto reach = std::stod(argv[3]);
    const int repeats = std::stoi(argv[4]);
    const auto batch = std::stoul(argv[6]);
    const auto bytes = std::filesystem::file_size(argv[2]);
    static_assert(sizeof(device::Ray) == 6 * sizeof(double));
    if (!(reach > 0) || !batch || repeats < 1 || !bytes ||
        bytes % sizeof(device::Ray))
      throw std::invalid_argument("invalid ray batch");
    std::vector<device::Ray> rays(bytes / sizeof(device::Ray));
    std::ifstream input(argv[2], std::ios::binary);
    if (!input.read(reinterpret_cast<char *>(rays.data()), bytes))
      throw std::runtime_error("failed reading rays");
    auto part = cuda_columns(*load_solid(argv[1]), false, reach);
    std::vector<device::Hit> hits;
    std::ofstream output(argv[5], std::ios::binary);
    std::cout<<std::setprecision(17)<<'[';
    for(int trial=0;trial<3;++trial) {
      if(trial) std::cout<<',';
      std::cout<<part->time_ray_batches(rays,reach,static_cast<int>(batch),repeats,hits);
    }
    std::cout<<"]\n";
    for(const auto& hit:hits) {
      const double row[]{double(hit.face),hit.t};
      output.write(reinterpret_cast<const char*>(row),sizeof(row));
    }
    if (!output)
      throw std::runtime_error("failed writing exits");
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
