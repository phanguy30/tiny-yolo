#pragma once
// Minimal persistent thread pool (std::thread only). The calling thread also
// participates, so a pool of N threads spawns N-1 workers.  Tasks are pulled
// dynamically from an atomic counter for load balance.  Workers spin briefly
// on an atomic generation counter before blocking on a condition variable, so
// back-to-back parallel regions (common in streamed conv chains) do not pay the
// sleep/wake-up latency, while idle threads still sleep between inferences.
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(int threads, int spin_iterations = 20000);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return threads_; }

    // fn(task_index, thread_index) for task_index in [0, count)
    template <class F>
    void parallel_for(int count, F&& fn) {
        if (count <= 0) return;
        if (threads_ == 1 || count == 1) {
            for (int i = 0; i < count; ++i) fn(i, 0);
            return;
        }
        run(count, &trampoline<F>, static_cast<void*>(&fn));
    }

private:
    using TaskFn = void (*)(void*, int, int);
    template <class F>
    static void trampoline(void* ctx, int task, int thread) { (*static_cast<F*>(ctx))(task, thread); }

    void run(int count, TaskFn fn, void* ctx);
    void worker(int index);
    void work(int thread_index);

    int threads_;
    int spin_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable start_cv_, done_cv_;
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<bool> stop_{false};
    std::atomic<int> active_{0};
    std::atomic<int> sleepers_{0};
    bool main_waiting_ = false;

    TaskFn fn_ = nullptr;
    void* ctx_ = nullptr;
    int count_ = 0;
    std::atomic<int> next_{0};
};
