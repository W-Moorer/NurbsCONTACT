// Time only HHT and graph scheduling using the coordinate loads from an existing scene.
#include "../src/cuda_columns.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    using namespace nc::compiled::device;
    try {
        if (argc != 4)
            throw std::invalid_argument("expected scene, step and step count");
        std::ifstream file(argv[1]);
        if (!file)
            throw std::runtime_error("cannot open scene");
        ResidentScene model;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream fields(line);
            std::string kind;
            fields >> kind;
            if (kind == "coordinate") {
                AxisCoordinate c{};
                if (!(fields >> c.mass >> c.q >> c.v >> c.k >> c.c >> c.rest >> c.load))
                    throw std::invalid_argument("invalid coordinate");
                model.coordinates.push_back(c);
            } else if (kind == "load") {
                size_t axis, n;
                if (!(fields >> axis >> n))
                    throw std::invalid_argument("invalid load table");
                auto &c = model.coordinates.at(axis);
                c.first = static_cast<int>(model.loads.size());
                for (size_t k = 0; k < n; ++k) {
                    if (!std::getline(file, line))
                        throw std::invalid_argument("incomplete load table");
                    std::istringstream row(line);
                    LoadPoint load;
                    if (!(row >> load.t >> load.value))
                        throw std::invalid_argument("invalid load");
                    model.loads.push_back(load);
                }
                c.end = static_cast<int>(model.loads.size());
            }
        }
        const auto dt = std::stod(argv[2]);
        const int steps = std::stoi(argv[3]);
        auto run = advance_resident(model, dt, steps);
        if (!std::isfinite(run.frames.back()))
            throw std::runtime_error("nonfinite coordinate response");
        std::cout << "measurement,steps,dt_s,online_s,geometry_queries,force_evaluations,p50_s,p99_s,max_s,"
                     "missed_deadlines\n"
                  << "HHT_without_contact," << steps << ',' << dt << ',' << run.online_s << ',' << run.queries
                  << ',' << run.forces << ',' << run.p50_s << ',' << run.p99_s << ',' << run.maximum_step_s
                  << ',' << run.missed_deadlines << '\n';
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
