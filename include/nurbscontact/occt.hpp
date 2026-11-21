#pragma once

#include "nurbscontact/geometry.hpp"

#include <TopoDS_Shape.hxx>

#include <filesystem>

namespace nc {

struct ImportOptions {
    double uv_join_tolerance{1e-5};
};

Model adapt_occt(const TopoDS_Shape& shape, const ImportOptions& options = {});
TopoDS_Shape read_step_shape(const std::filesystem::path& file);
Model read_step(const std::filesystem::path& file, const ImportOptions& options = {});

}  // namespace nc
