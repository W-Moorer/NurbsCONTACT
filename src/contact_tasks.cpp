#include "contact_tasks.hpp"

#ifdef NC_WITH_TBB
#include <algorithm>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

namespace nc::contact_tasks {
int concurrency() {
#ifdef NC_WITH_TBB
    static const int limit = [] {
        const int available = oneapi::tbb::info::default_concurrency();
#ifdef _OPENMP
        return std::min(available, omp_get_max_threads());
#else
        return available;
#endif
    }();
    return limit;
#else
    return 0;
#endif
}

void ranges(std::size_t count, std::size_t grain,
            const std::function<void(std::size_t, std::size_t)>& visit) {
#ifdef NC_WITH_TBB
    if (count > grain && concurrency() > 1) {
        static oneapi::tbb::task_arena arena(concurrency());
        arena.execute([&] {
            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, count, grain),
                [&](const auto& range) { visit(range.begin(), range.end()); });
        });
        return;
    }
#else
    (void)grain;
#endif
    if (count) visit(0, count);
}
}
