#include "thread_pool.hpp"

ThreadPool::ThreadPool(int threads, int spin_iterations)
    : threads_(threads < 1 ? 1 : threads), spin_(spin_iterations) {
    for (int i = 1; i < threads_; ++i) workers_.emplace_back(&ThreadPool::worker, this, i);
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_.store(true);
    }
    start_cv_.notify_all();
    for (auto& t : workers_) t.join();
}

void ThreadPool::work(int thread_index) {
    for (;;) {
        const int task = next_.fetch_add(1, std::memory_order_relaxed);
        if (task >= count_) break;
        fn_(ctx_, task, thread_index);
    }
}

void ThreadPool::run(int count, TaskFn fn, void* ctx) {
    fn_ = fn;
    ctx_ = ctx;
    count_ = count;
    next_.store(0, std::memory_order_relaxed);
    active_.store(static_cast<int>(workers_.size()), std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(mutex_);  // pairs with sleeping workers' predicate check
        generation_.fetch_add(1, std::memory_order_release);
    }
    if (sleepers_.load(std::memory_order_acquire) > 0) start_cv_.notify_all();
    work(0);
    // wait for workers: spin first, then block
    for (int i = 0; i < spin_; ++i)
        if (active_.load(std::memory_order_acquire) == 0) return;
    std::unique_lock<std::mutex> lock(mutex_);
    main_waiting_ = true;
    done_cv_.wait(lock, [this] { return active_.load(std::memory_order_acquire) == 0; });
    main_waiting_ = false;
}

void ThreadPool::worker(int index) {
    std::uint64_t seen = 0;
    for (;;) {
        bool got = false;
        for (int i = 0; i < spin_ && !got; ++i) {
            if (generation_.load(std::memory_order_acquire) != seen || stop_.load(std::memory_order_relaxed)) got = true;
            else if ((i & 1023) == 1023) std::this_thread::yield();
        }
        if (!got) {
            std::unique_lock<std::mutex> lock(mutex_);
            sleepers_.fetch_add(1, std::memory_order_acq_rel);
            start_cv_.wait(lock, [&] { return stop_.load() || generation_.load(std::memory_order_acquire) != seen; });
            sleepers_.fetch_sub(1, std::memory_order_acq_rel);
        }
        if (stop_.load()) return;
        seen = generation_.load(std::memory_order_acquire);
        work(index);
        if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (main_waiting_) done_cv_.notify_one();
        }
    }
}
