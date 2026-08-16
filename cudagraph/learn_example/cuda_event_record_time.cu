#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <cstdlib>

// CUDA kernel to sleep for n seconds using __nanosleep
__global__ void sleepKernel(int n) {
    unsigned long long total_ns = static_cast<unsigned long long>(n) * 1000000000ULL;
    unsigned long long max_ns = 1000000ULL; // 1 millisecond
    unsigned long long iterations = total_ns / max_ns;
    unsigned long long remaining_ns = total_ns % max_ns;

    for (unsigned long long i = 0; i < iterations; ++i) {
        __nanosleep(max_ns);
    }
    if (remaining_ns > 0) {
        __nanosleep(remaining_ns);
    }
}

int main(int argc, char *argv[]) {
    // Default sleep times
    float default_sleep_time = 0.5f;
    float first_sleep_seconds = default_sleep_time;
    float second_sleep_seconds = default_sleep_time;

    // Parse command-line arguments if provided
    if (argc >= 2) {
        first_sleep_seconds = std::atof(argv[1]);
    }
    if (argc >= 3) {
        second_sleep_seconds = std::atof(argv[2]);
    }

    // Initialize CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the first kernel to sleep for 1 second
    sleepKernel<<<1, 1>>>(1);

    // Host sleep for the first sleep time
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(first_sleep_seconds * 1000)));

    // Record the start event
    cudaEventRecord(start);

    // Launch the second kernel to sleep for 1 second
    sleepKernel<<<1, 1>>>(1);

    // Host sleep for the second sleep time
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(second_sleep_seconds * 1000)));

    // Record the stop event
    cudaEventRecord(stop);

    cudaEventSynchronize(stop); // Ensure the second kernel completes

    // Calculate the elapsed time for the second kernel
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print the elapsed time
    std::cout << "Elapsed time for the second sleep: " << elapsedTime / 1000.0f << " seconds" << std::endl;

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}