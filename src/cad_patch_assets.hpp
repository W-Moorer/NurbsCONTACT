#pragma once

#include <BinTools.hxx>
#include <OSD_Process.hxx>
#include <Standard_Version.hxx>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <vector>

namespace nc::cad {
// Each file contains the native shape identity and one versioned, fixed-row table.
// A missing or incomplete file simply regenerates the table in memory.
class PatchAssets {
    std::string identity_;
    std::filesystem::path prefix_;
    static std::uint64_t hash(const char* bytes, std::size_t n) {
        std::uint64_t h = 14695981039346656037ull;
        for (std::size_t i = 0; i < n; ++i)
            h = (h ^ static_cast<unsigned char>(bytes[i])) * 1099511628211ull;
        return h;
    }

  public:
    std::size_t loaded{}, built{};
    explicit PatchAssets(const TopoDS_Shape& shape) {
        std::error_code ec;
        const auto setting = std::getenv("NC_CAD_CACHE_DIR");
        auto dir = setting ? std::filesystem::path(setting)
                           : std::filesystem::temp_directory_path(ec) / "nurbscontact-cad";
        if (ec || dir.empty()) return;
        std::filesystem::create_directories(dir, ec);
        if (ec) return;
        try {
            std::ostringstream encoded(std::ios::out | std::ios::binary);
            BinTools::Write(shape, encoded, false, false, BinTools_FormatVersion_CURRENT);
            if (!encoded) return;
            identity_ = encoded.str();
            prefix_ = dir / ("patch-v3-" OCC_VERSION_COMPLETE "-" +
                             std::to_string(hash(identity_.data(), identity_.size())));
        } catch (const Standard_Failure&) {
        }
    }
    template <class Row, class Build>
    std::vector<Row> table(const std::string& name, std::size_t limit, Build build) {
        static_assert(std::is_trivially_copyable_v<Row>);
        const std::filesystem::path path = prefix_.string() + "-" + name + ".bin";
        if (!prefix_.empty()) {
            std::ifstream in(path, std::ios::binary | std::ios::ate);
            const auto end = in.tellg();
            const std::size_t header = identity_.size() + sizeof(std::uint64_t);
            if (in && end >= 0 && static_cast<std::uintmax_t>(end) >= header) {
                const auto bytes = static_cast<std::uintmax_t>(end) - header;
                if (bytes % sizeof(Row) == 0 && bytes / sizeof(Row) <= limit) {
                    in.seekg(0);
                    std::string key(identity_.size(), '\0');
                    std::uint64_t checksum{};
                    in.read(key.data(), key.size())
                        .read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
                    if (in && key == identity_) {
                        std::vector<Row> rows(bytes / sizeof(Row));
                        auto data = reinterpret_cast<char*>(rows.data());
                        if (bytes) in.read(data, bytes);
                        if (in && checksum == hash(data, bytes)) {
                            ++loaded;
                            return rows;
                        }
                    }
                }
            }
        }
        auto rows = build();
        ++built;
        if (prefix_.empty()) return rows;
        static std::atomic_uint sequence{};
        const auto temp = path.string() + "." + std::to_string(OSD_Process().ProcessId()) + "." +
                          std::to_string(sequence++) + ".tmp";
        const auto bytes = rows.size() * sizeof(Row);
        const auto data = reinterpret_cast<const char*>(rows.data());
        const auto checksum = hash(data, bytes);
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        out.write(identity_.data(), identity_.size())
            .write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
        if (bytes) out.write(data, bytes);
        out.close();
        std::error_code ec;
        if (out) std::filesystem::rename(temp, path, ec);
        std::filesystem::remove(temp, ec);
        return rows;
    }
};
} // namespace nc::cad
