#pragma once

#include <cstddef>
#include <functional>

namespace nc::contact_tasks {
// Zero selects the legacy pair dispatcher. All task ranges share one arena.
int concurrency();
void ranges(std::size_t count, std::size_t grain,
            const std::function<void(std::size_t, std::size_t)>& visit);
}
