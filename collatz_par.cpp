#include <cstdint>
#include <hpc_helpers.hpp>
#include <optional>
#include <string_view>
#include <thread>
#include <threadPool.hpp>
#include <vector>

/**
 * @brief Alias for unsigned long long for brevity
 */
using ull = unsigned long long;

/**
 * @brief Configuration options for the Collatz calculator
 */
struct Options {
    bool dynamic_scheduling{false};  ///< Flag to use dynamic scheduling for parallel processing
    uint32_t n_threads{16};          ///< Number of threads to use
    ull task_size{1ULL};             ///< Chunk size for task distribution

    /**
     * @brief Prints the program configuration
     */
    void print() const noexcept {
        std::cout << "Configuration: \n"
                  << "  - Scheduling: " << (dynamic_scheduling ? "dynamic" : "static") << "\n"
                  << "  - Threads: " << n_threads << "\n"
                  << "  - Task size: " << task_size << "\n";
    }
};

/**
 * @brief Represents a range of numbers to process
 */
struct Range {
    const ull lower_bound;        ///< Lower bound of the range (inclusive)
    const ull upper_bound;        ///< Upper bound of the range (inclusive)
    const ull size;               ///< Size of the range (calculated at construction)
    ull collatz_max_steps{0ULL};  ///< Max number of steps in the Collatz sequence for a number in
                                  ///< the range

    /**
     * @brief Construct a new Range object
     *
     * @param low_bound Lower bound of the range (inclusive)
     * @param up_bound Upper bound of the range (inclusive)
     */
    explicit Range(ull low_bound, ull up_bound)
        : lower_bound(low_bound),
          upper_bound(up_bound),
          size((low_bound <= up_bound) ? (up_bound - low_bound + 1) : 0) {
    }

    /**
     * @brief Check if the range is valid
     *
     * @return true if valid, false otherwise
     */
    [[nodiscard]] bool is_valid() const noexcept {
        return lower_bound <= upper_bound;
    }
};

/**
 * @brief Parses a string in the format "start-end" into a Range
 *
 * @param arg String containing the range specification
 * @return Optional Range, empty if parsing fails
 */
[[nodiscard]] std::optional<Range> parse_range(const std::string_view arg) {
    // Find the position of the dash
    auto dash_position = arg.find('-');

    if (dash_position == std::string_view::npos) {
        std::cout << "Invalid range format " << arg << ". Expected 'start-end'." << std::endl;
        return std::nullopt;
    }
    // Extract the lower bound
    ull lower_bound = std::stoull(arg.substr(0, dash_position).data());
    // Extract the upper bound
    ull upper_bound = std::stoull(arg.substr(dash_position + 1).data());

    Range range{lower_bound, upper_bound};
    if (!range.is_valid()) {
        std::cout << "Invalid range: " << range.lower_bound << " is less than " << range.upper_bound
                  << std::endl;
        return std::nullopt;
    }

    return range;
}

/**
 * @brief Parses command line arguments
 *
 * @param args Vector of command line arguments
 * @param opt Options structure to populate
 * @return Optional Vector of Range objects, empty if parsing fails
 */
[[nodiscard]] std::optional<std::vector<Range>> parse_args(
    const std::vector<std::string_view>& args, Options& opt) {
    // Vector to store the ranges
    std::vector<Range> ranges;
    size_t n_args = args.size();
    size_t i{0};

    // Check if options are set
    for (; i < n_args; i++) {
        if (args[i] == "-d") {
            opt.dynamic_scheduling = true;

        } else if (args[i] == "-n" && i + 1 < n_args) {
            // Extract the number of threads
            auto n_threads = static_cast<uint32_t>(std::stoi(args[++i].data()));

            // Check the number of threads
            if (n_threads <= 0) {
                std::cout << "Invalid number of threads: " << n_threads
                          << "The number must be positive." << std::endl;
                return std::nullopt;
            }
            // Store in the struct
            opt.n_threads = n_threads;

        } else if (args[i] == "-c" && i + 1 < n_args) {
            // Extract the task size
            auto task_size = static_cast<uint32_t>(std::stoi(args[++i].data()));

            // Check the task size
            if (task_size <= 0) {
                std::cout << "Invalid task size: " << task_size << "The number must be positive."
                          << std::endl;
                return std::nullopt;
            }

            opt.task_size = task_size;
        } else
            // No options found, must be ranges or invalid argument
            break;
    }

    // Iterate on the ranges
    for (; i < n_args; i++) {
        // Store if parsing doesn't fail
        if (auto range = parse_range(args[i]))
            ranges.emplace_back(range.value());
        else
            return std::nullopt;
    }
    return ranges;
}

/**
 * @brief Prints usage information
 *
 * @param program_name Name of the program
 */
void print_usage(std::string_view program_name) {
    std::cout << "Usage" << program_name
              << "<start>-<end> [<start>-<end> ...]\n"
                 "\n"
                 "Options:\n"
                 "  -d       Enable dynamic scheduling for parallel processing\n"
                 "  -n N     Set number of threads (default: 16)\n"
                 "  -c N     Set task chunk size (default: 1)\n";
}

/**
 * @brief Prints the max steps of the Collatz sequence for every range
 *
 * @param ranges Vector of ranges
 */
void print_results(const std::vector<Range>& ranges) {
    for (const auto& range : ranges)
        std::cout << "Start: " << range.lower_bound << " \t End: " << range.upper_bound
                  << "\t Max Steps: " << range.collatz_max_steps << std::endl;
}

/**
 * @brief Calculates the number of steps in the Collatz sequence for a given number
 *
 * The Collatz conjecture states that for any positive integer n:
 * - If n is even, divide it by 2
 * - If n is odd, multiply by 3 and add 1
 * - Repeat until n = 1
 *
 * @param n The starting number
 * @return Number of steps to reach 1
 */
constexpr ull collatz_steps(ull n) {
    ull steps{0ULL};
    while (n != 1ULL) {
        n = (n % 2ULL == 0ULL) ? n / 2ULL : 3ULL * n + 1ULL;
        ++steps;
    }
    return steps;
}

/**
 * @brief Computes the maximum Collatz sequence length within a given range sequentially
 *
 * @param range Range of values to process
 */
void collatz_seq(std::vector<Range>& ranges) {
    for (auto& range : ranges) {
        ull max_steps{0ULL};

        for (auto i = range.lower_bound; i <= range.upper_bound; i++) {
            auto steps = collatz_steps(i);
            max_steps = std::max(max_steps, steps);
        }
        range.collatz_max_steps = max_steps;
    }
}

/**
 * @brief Computes the maximum Collatz sequence length within a given range using threads
 *
 * This function divides the workload using a block-cyclic distribution pattern.
 * Each thread processes a subset of numbers and tracks its local maximum and,
 * when it acquires the lock, update the global max for the range
 *
 * @param range Range of values to process
 * @param opt Configuration options including thread count and task size
 */
void collatz_static(std::vector<Range>& ranges, const Options& opt) {
    // Mutex to access the range_max variable
    std::mutex max_mutex;

    /**
     * @brief Function executed by each thread
     *
     * 1. Starts at its designated offset
     * 2. Processes a block of size task_size
     * 3. Jumps ahead by padding to skip blocks assigned to other threads
     * 4. Update the max when the thread acquires the lock
     *
     * @param thread_id ID of the current thread
     */
    auto block_cyclic = [&opt, &max_mutex, &ranges](int thread_id) {
        for (auto& range : ranges) {
            // Compute the starting point for this thread
            const ull offset = thread_id * opt.task_size + range.lower_bound;
            // Calculate the jump to the next chunk assigned to this thread
            const ull padding = opt.n_threads * opt.task_size;
            // Max steps tracked by this thread
            ull thread_max_steps{0ULL};

            // Process chunks
            for (auto i = offset; i <= range.upper_bound; i += padding) {
                // Compute the upper bound of this chunk
                const ull upper_bound_block = std::min(i + opt.task_size - 1, range.upper_bound);

                // Process each number in the current block
                for (auto j = i; j <= upper_bound_block; j++) {
                    auto steps = collatz_steps(j);
                    thread_max_steps = std::max(thread_max_steps, steps);
                }
            }
            // Scope of the lock to update the max
            {
                std::lock_guard<std::mutex> lock(max_mutex);
                range.collatz_max_steps = std::max(thread_max_steps, range.collatz_max_steps);
            }
        }
    };

    std::vector<std::thread> threads;

    // Launch the threads
    for (uint32_t i = 0; i < opt.n_threads; i++) {
        threads.emplace_back(block_cyclic, i);
    }

    // Wait for all the threads
    for (auto& thread : threads) thread.join();
}

/**
 * @brief Computes Collatz maximum sequence length using dynamic thread scheduling
 *
 * This function uses a thread pool to dynamically assign tasks to threads.
 * Each range is divided into smaller chunks of size task_size.
 * Max reduction is done in parallel.
 *
 * @param ranges Vector of Range objects to process
 * @param opt Configuration options including thread count and task size
 */
void collatz_dyn(std::vector<Range>& ranges, Options& opt) {
    // Create thread pool with specified number of threads
    ThreadPool tp(opt.n_threads);

    // Process each range
    for (auto& range : ranges) {
        const ull upper_bound = range.upper_bound;
        const ull lower_bound = range.lower_bound;

        // Shared max steps for this range
        std::mutex max_mutex;

        // Vector to store futures for tracking task completion
        std::vector<std::future<void>> futures;

        // Divide range into tasks of size task_size
        for (ull offset = lower_bound; offset <= upper_bound; offset += opt.task_size) {
            /**
             * @brief Function executed by each thread in the thread pool
             *
             * 1. Starts at its designated offset
             * 2. Processes a block of size task_size
             * 3. Update the max when it acquires the lock
             */
            auto thread_task = [task_size = opt.task_size, upper_bound, &max_mutex,
                                &range](const ull offset) {
                // Calculate actual upper bound for this task
                const ull upper = std::min(offset + task_size - 1, upper_bound);
                ull thread_max_steps{0ULL};

                // Process each number in the current chunk
                for (auto j = offset; j <= upper; ++j) {
                    auto steps = collatz_steps(j);
                    thread_max_steps = std::max(thread_max_steps, steps);
                }

                // Update the maximum steps with mutex protection
                {
                    std::lock_guard<std::mutex> lock(max_mutex);
                    range.collatz_max_steps = std::max(range.collatz_max_steps, thread_max_steps);
                }
            };

            // Submit task to thread pool and store the future
            futures.push_back(tp.enqueue(thread_task, offset));
        }

        // Wait for all tasks for this range to complete
        for (auto& future : futures) {
            future.wait();
        }
    }
}

/**
 * @brief Computes Collatz maximum sequence length using dynamic thread scheduling
 *
 * This function uses a thread pool to dynamically assign tasks to threads.
 * Each range is divided into smaller chunks of size task_size.
 * Max reduction is done sequentially.
 *
 * @param ranges Vector of Range objects to process
 * @param opt Configuration options including thread count and task size
 */
void collatz_dyn_no_red(std::vector<Range>& ranges, Options& opt) {
    // Each cell contains a vector in which are stored all the local max computed by threads on a
    // single range
    std::vector<std::vector<std::future<ull>>> results(ranges.size());
    // Create thread pool with specified number of threads
    ThreadPool tp(opt.n_threads);

    // Process each range
    for (auto i = 0ULL; i < ranges.size(); i++) {
        const ull upper_bound = ranges[i].upper_bound;
        const ull lower_bound = ranges[i].lower_bound;

        // Divide range into tasks of size task_size
        for (ull offset = lower_bound; offset <= upper_bound; offset += opt.task_size) {
            /**
             * @brief Function executed by each thread in the thread pool
             *
             * 1. Starts at its designated offset
             * 2. Processes a block of size task_size
             * 3. Update the max when it acquires the lock
             */
            auto thread_task = [task_size = opt.task_size, upper_bound](const ull offset) {
                // Calculate actual upper bound for this task
                const ull upper = std::min(offset + task_size, upper_bound);
                ull thread_max_steps{0ULL};

                // Process each number in the current chunk
                for (auto j = offset; j <= upper; ++j) {
                    auto steps = collatz_steps(j);
                    thread_max_steps = std::max(thread_max_steps, steps);
                }
                return thread_max_steps;
            };
            results[i].emplace_back(tp.enqueue(thread_task, offset));
        }
    }

    // Compute the max for every range
    for (auto i = 0ULL; i < ranges.size(); i++) {
        ull range_max_steps{0ULL};
        for (auto& result : results[i]) range_max_steps = std::max(range_max_steps, result.get());
        ranges[i].collatz_max_steps = range_max_steps;
    }
}

/**
 * @brief Main function to control program execution flow
 *
 * Parses command line arguments, configures options, and executes
 * Collatz computation based on configuration.
 * Times the executions and prints results.
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return Program exit code
 */
int main(int argc, char* argv[]) {
    if (argc == 1) {
        print_usage(argv[0]);
        return 0;
    }

    // Wrap the raw C-strings in a vector
    // string_view avoids to copy the characters and is read-only
    std::vector<std::string_view> args(argv + 1, argv + argc);

    // Options passed to the program by command line
    Options opt;

    // Parse the arguments from the command line
    auto opt_ranges = parse_args(args, opt);

    // Error occurred
    if (!opt_ranges.has_value()) return 0;

    // Extract the ranges from the optional
    auto ranges = opt_ranges.value();

    // Print configurations
    opt.print();

    // Execute sequential version
    TIMERSTART(coll_seq)
    collatz_seq(ranges);
    TIMERSTOP(coll_seq)

    print_results(ranges);
    // Reset results for parallel execution
    for (auto& range : ranges) range.collatz_max_steps = 0;

    // Execute either static or dynamic parallel version based on configuration
    if (!opt.dynamic_scheduling) {
        // Static scheduling with block-cyclic distribution
        TIMERSTART(coll_static)
        collatz_static(ranges, opt);
        TIMERSTOP(coll_static)
    } else {
#if 0
        // Dynamic scheduling with thread pool
        TIMERSTART(coll_dynamic)
        collatz_dyn(ranges, opt);
        TIMERSTOP(coll_dynamic)
        // Reset results for parallel execution
        for (auto& range : ranges) range.collatz_max_steps = 0;
#endif
        TIMERSTART(coll_dynamic_no_red)
        collatz_dyn_no_red(ranges, opt);
        TIMERSTOP(coll_dynamic_no_red)
    }

    print_results(ranges);
}
