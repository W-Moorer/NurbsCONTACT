// Synchronous controller benchmark. Each applied load is published only after
// the previous wrench has reached the host. The dynamics remain on the GPU.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#if defined(_WIN32)
#define NC_EXPORT extern "C" __declspec(dllexport)
#else
#define NC_EXPORT extern "C"
#endif

NC_EXPORT int exchange_steps(char *page, int begin, int end, int joints, int wrench_size,
                             const double *loads, double *received, std::int64_t *elapsed) {
    using Clock = std::chrono::steady_clock;
    std::atomic_ref<int> command(*reinterpret_cast<int *>(page));
    std::atomic_ref<int> reply(*reinterpret_cast<int *>(page + 64));
    double *input = reinterpret_cast<double *>(page + 128);
    double *wrench = reinterpret_cast<double *>(page + 384);
    for (int step = begin + 1; step <= end; ++step) {
        const auto start = Clock::now();
        std::memcpy(input, loads + step * joints, joints * sizeof(double));
        command.store(step, std::memory_order_release);
        for (unsigned polls = 0; reply.load(std::memory_order_acquire) != step; ++polls) {
            if (polls % 4096 == 0 && Clock::now() - start > std::chrono::milliseconds(500))
                return step;
        }
        std::memcpy(received + step * wrench_size, wrench, wrench_size * sizeof(double));
        elapsed[step] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    }
    return 0;
}
